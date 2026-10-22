import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import scipy.sparse as sparse

from cinerec.retrieval.base import BaseRetriever
from cinerec.ranking.base import BaseRanker
from cinerec.retrieval.als_retriever import ALSRetriever

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TOP_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]
DEFAULT_USER_STATS = {"user_avg_rating": 3.5, "user_num_ratings": 0}
DEFAULT_MOVIE_STATS = {"movie_avg_rating": 3.5, "movie_num_ratings": 0}
MIN_RATINGS_FOR_PERSONALIZATION = 5


class TwoStageRecommender:
    """Двухэтапный рекомендательный пайплайн (Retrieval + Ranking) для v2.0."""

    def __init__(self, retriever: BaseRetriever = None, ranker: BaseRanker = None) -> None:
        logger.info("Инициализация пайплайна...")
        
        self.retriever = retriever if retriever else ALSRetriever()
        
        if ranker is None:
            logger.info("Загрузка дефолтного ранжировщика (LGBMRanker-LTR)...")
            with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
                self.ranker = ranker if ranker else pickle.load(f)

        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Загружает предвычисленные статистики и метаданные из офлайн-пайплайна."""
        logger.info("Загрузка справочников...")
        # Эти данные мы предвычисляли ранее и они нужны для фичей ранжировщика
        self.movie_stats = pd.read_parquet(PROCESSED_DIR / "movie_stats.parquet").to_dict(orient="index")
        self.popular_movie_ids = pd.read_parquet(PROCESSED_DIR / "popular_movies.parquet").index.tolist()

        movies = pd.read_csv(DATA_DIR / "movies.csv")
        movies["genres"] = movies["genres"].fillna("")
        for genre in TOP_GENRES:
            movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)
            
        self.movie_genres = movies.set_index("movieId")[[f"is_{g}" for g in TOP_GENRES]].to_dict(orient="index")
        self.movie_titles = movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

    def _get_popular_fallback(self, top_n: int) -> List[Dict[str, Any]]:
        """Возвращает витрину популярных фильмов (Cold Start Fallback)."""
        results = []
        for movie_id in self.popular_movie_ids[:top_n]:
            meta = self.movie_titles.get(movie_id, {"title": "Unknown", "genres": ""})
            stats = self.movie_stats.get(movie_id, {"movie_avg_rating": 0.0})
            results.append({
                "movieId": int(movie_id),
                "title": meta["title"],
                "genres": meta["genres"],
                "score": float(stats["movie_avg_rating"] / 5.0)
            })
        return results

    def _calculate_user_features(self, user_ratings: Dict[int, float]) -> Dict[str, float]:
        """Рассчитывает признаки пользователя на лету на основе его оценок из БД."""
        if not user_ratings:
            return {**DEFAULT_USER_STATS, **{f"user_avg_{g}": 3.5 for g in TOP_GENRES}}

        avg_rating = float(np.mean(list(user_ratings.values())))
        num_ratings = len(user_ratings)
        
        features = {
            "user_avg_rating": avg_rating,
            "user_num_ratings": num_ratings
        }

        # Считаем средний рейтинг по каждому жанру для этого пользователя
        for genre in TOP_GENRES:
            genre_movies = [
                mid for mid in user_ratings.keys() 
                if self.movie_genres.get(mid, {}).get(f"is_{genre}", 0) == 1
            ]
            if genre_movies:
                genre_ratings = [user_ratings[mid] for mid in genre_movies]
                features[f"user_avg_{genre}"] = float(np.mean(genre_ratings))
            else:
                features[f"user_avg_{genre}"] = avg_rating

        return features

    def recommend(
        self, 
        user_ratings: Dict[int, float], 
        top_n: int = 10, 
        retrieval_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Генерирует топ-N рекомендаций на основе оценок пользователя (из БД).
        
        Args:
            user_ratings: Словарь {movie_id: score} с оценками пользователя.
            top_n: Количество рекомендаций.
            retrieval_k: Количество кандидатов для этапа ранжирования.
        """
        # 1. Проверка на холодный старт (если оценок мало)
        if len(user_ratings) < MIN_RATINGS_FOR_PERSONALIZATION:
            logger.info("Cold start: user has %d ratings. Returning popular items.", len(user_ratings))
            return self._get_popular_fallback(top_n)

        # 2. RETRIEVAL (Онлайн-обновление)
        # Нам нужно получить кандидатов на основе оценок пользователя
        candidates = self.retriever.recommend_for_ratings(user_ratings, top_k=retrieval_k)
        if not candidates:
            return self._get_popular_fallback(top_n)

        # 3. RANKING
        user_feats = self._calculate_user_features(user_ratings)
        features = []

        for item in candidates:
            movie_id = item["movieId"]
            retrieval_score = item["score"]
            
            m_stats = self.movie_stats.get(movie_id, DEFAULT_MOVIE_STATS)
            m_genres = self.movie_genres.get(movie_id, {f"is_{g}": 0 for g in TOP_GENRES})
            
            # Формируем полный вектор фичей для LightGBM
            feature_row = {
                "retrieval_score": retrieval_score,
                "movie_avg_rating": m_stats["movie_avg_rating"],
                "movie_num_ratings": m_stats["movie_num_ratings"],
                "user_avg_rating": user_feats["user_avg_rating"],
                "user_num_ratings": user_feats["user_num_ratings"],
                "rating_diff": user_feats["user_avg_rating"] - m_stats["movie_avg_rating"],
                "popularity_score": m_stats["movie_num_ratings"] * m_stats["movie_avg_rating"],
                **m_genres,
            }
            # Добавляем фичи по жанрам
            for genre in TOP_GENRES:
                feature_row[f"user_avg_{genre}"] = user_feats[f"user_avg_{genre}"]
                
            features.append(feature_row)

        feature_cols = ["retrieval_score", "movie_avg_rating", "movie_num_ratings", 
                        "user_avg_rating", "user_num_ratings", "rating_diff", "popularity_score"] \
                       + [f"is_{g}" for g in TOP_GENRES] + [f"user_avg_{g}" for g in TOP_GENRES]
        
        X_rank = pd.DataFrame(features).reindex(columns=feature_cols, fill_value=0)
        
        # Инференс модели (в зависимости от того, LTR или Classifier)
        scores = self.ranker.predict(X_rank)

        # 4. Формирование итогового ответа
        results = []
        for item, score in zip(candidates, scores):
            movie_id = item["movieId"]
            meta = self.movie_titles.get(movie_id, {"title": "Unknown", "genres": ""})
            results.append({
                "movieId": movie_id,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": float(score)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]