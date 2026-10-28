import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cinerec.ranking.base import BaseRanker
from cinerec.retrieval.als import ALSRetriever
from cinerec.retrieval.base import BaseRetriever

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

    def __init__(
        self, 
        retriever: BaseRetriever | None = None, 
        ranker: BaseRanker | None = None
    ) -> None:
        logger.info("Инициализация пайплайна...")
        
        # Dependency Injection для Retriever
        self.retriever = retriever if retriever is not None else ALSRetriever()
        
        # ИСПРАВЛЕНИЕ: Правильная логика для Ranker
        if ranker is not None:
            logger.info("Использование переданного ранжировщика: %s", type(ranker).__name__)
            self.ranker = ranker
        else:
            logger.info("Загрузка дефолтного ранжировщика (LGBMRanker-LTR)...")
            with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
                self.ranker = pickle.load(f)

        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Загружает предвычисленные статистики и метаданные из офлайн-пайплайна."""
        logger.info("Загрузка справочников...")
        self.movie_stats = pd.read_parquet(PROCESSED_DIR / "movie_stats.parquet").to_dict(orient="index")
        self.popular_movie_ids = pd.read_parquet(PROCESSED_DIR / "popular_movies.parquet").index.tolist()

        movies = pd.read_csv(DATA_DIR / "movies.csv")
        movies["genres"] = movies["genres"].fillna("")
        for genre in TOP_GENRES:
            movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)
            
        self.movie_genres = movies.set_index("movieId")[[f"is_{g}" for g in TOP_GENRES]].to_dict(orient="index")
        self.movie_titles = movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

    def _get_popular_fallback(self, top_n: int) -> list[dict[str, Any]]:
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

    def _calculate_user_features(self, user_ratings: dict[int, float]) -> dict[str, float]:
        """Рассчитывает признаки пользователя на лету на основе его оценок из БД."""
        if not user_ratings:
            return {**DEFAULT_USER_STATS, **{f"user_avg_{g}": 3.5 for g in TOP_GENRES}}

        avg_rating = float(np.mean(list(user_ratings.values())))
        num_ratings = len(user_ratings)
        
        features = {
            "user_avg_rating": avg_rating,
            "user_num_ratings": num_ratings
        }

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
        user_ratings: dict[int, float], 
        top_n: int = 10, 
        retrieval_k: int = 50
    ) -> list[dict[str, Any]]:
        """
        Генерирует топ-N рекомендаций на основе оценок пользователя (из БД).
        """
        if len(user_ratings) < MIN_RATINGS_FOR_PERSONALIZATION:
            logger.info("Cold start: user has %d ratings. Returning popular items.", len(user_ratings))
            return self._get_popular_fallback(top_n)

        candidates = self.retriever.recommend_for_ratings(user_ratings, top_k=retrieval_k)
        if not candidates:
            return self._get_popular_fallback(top_n)

        user_feats = self._calculate_user_features(user_ratings)
        features = []

        for item in candidates:
            movie_id = item["movieId"]
            retrieval_score = item["score"]
            
            m_stats = self.movie_stats.get(movie_id, DEFAULT_MOVIE_STATS)
            m_genres = self.movie_genres.get(movie_id, {f"is_{g}": 0 for g in TOP_GENRES})
            
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
            for genre in TOP_GENRES:
                feature_row[f"user_avg_{genre}"] = user_feats[f"user_avg_{genre}"]
                
            features.append(feature_row)

        feature_cols = ["retrieval_score", "movie_avg_rating", "movie_num_ratings", 
                        "user_avg_rating", "user_num_ratings", "rating_diff", "popularity_score"] \
                       + [f"is_{g}" for g in TOP_GENRES] + [f"user_avg_{g}" for g in TOP_GENRES]
        
        X_rank = pd.DataFrame(features).reindex(columns=feature_cols, fill_value=0)
        
        # Теперь self.ranker гарантированно существует
        scores = self.ranker.predict(X_rank)

        results = []
        for item, score in zip(candidates, scores, strict=False):
            movie_id = item["movieId"]
            meta = self.movie_titles.get(movie_id, {"title": "Unknown", "genres": ""})
            results.append({
                "movieId": movie_id,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": float(score)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]