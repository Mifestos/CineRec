import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

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


class TwoStageRecommender:
    """Двухэтапный рекомендательный пайплайн (Retrieval + Ranking)."""

    def __init__(self, retriever: BaseRetriever = None, ranker: BaseRanker = None) -> None:
        logger.info("Инициализация пайплайна...")
        
        # Dependency Injection: если модель не передана, берем дефолтную
        self.retriever = retriever if retriever else ALSRetriever()
        
        if ranker is None:
            logger.info("Загрузка дефолтного ранжировщика (LGBMClassifier)...")
            with open(MODEL_DIR / "ranker_classifier.pkl", "rb") as f:
                self.ranker = pickle.load(f)
        else:
            self.ranker = ranker

        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Загружает предвычисленные статистики и метаданные для инференса."""
        logger.info("Загрузка справочников...")
        self.movie_stats = pd.read_parquet(PROCESSED_DIR / "movie_stats.parquet").to_dict(orient="index")
        self.user_stats = pd.read_parquet(PROCESSED_DIR / "user_stats.parquet").to_dict(orient="index")
        self.popular_movie_ids = pd.read_parquet(PROCESSED_DIR / "popular_movies.parquet").index.tolist()

        movies = pd.read_csv(DATA_DIR / "movies.csv")
        movies["genres"] = movies["genres"].fillna("")
        for genre in TOP_GENRES:
            movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)
            
        self.movie_genres = movies.set_index("movieId")[[f"is_{g}" for g in TOP_GENRES]].to_dict(orient="index")
        self.movie_titles = movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

        # Загружаем предвычисленную статистику по жанрам вместо чтения всего CSV
        self.user_genre_stats = {}
        for genre in TOP_GENRES:
            df = pd.read_parquet(PROCESSED_DIR / f"{genre}_stats.parquet")
            self.user_genre_stats[f"user_avg_{genre}"] = df[f"user_avg_{genre}"].to_dict()

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

        
    def recommend(self, user_id: int, top_n: int = 10, retrieval_k: int = 50, filter_already_liked: bool = True) -> List[Dict[str, Any]]:
        """Генерирует топ-N рекомендаций для пользователя."""
        
        if not self.retriever.has_user(user_id):
            logger.info("Cold start: user_id=%s. Returning popular items.", user_id)
            return self._get_popular_fallback(top_n)

        candidates = self.retriever.recommend(user_id, top_k=retrieval_k, filter_already_liked=filter_already_liked)
        if not candidates:
            return self._get_popular_fallback(top_n)

        u_stats = self.user_stats.get(user_id, DEFAULT_USER_STATS)
        features = []
        
        u_genre_features = {}
        for genre in TOP_GENRES:
            u_genre_features[f"user_avg_{genre}"] = self.user_genre_stats[f"user_avg_{genre}"].get(
                user_id, u_stats["user_avg_rating"]
            )

        for item in candidates:
            movie_id = item["movieId"]
            retrieval_score = item["score"]
            
            m_stats = self.movie_stats.get(movie_id, DEFAULT_MOVIE_STATS)
            m_genres = self.movie_genres.get(movie_id, {f"is_{g}": 0 for g in TOP_GENRES})
            
            features.append({
                "retrieval_score": retrieval_score,
                "movie_avg_rating": m_stats["movie_avg_rating"],
                "movie_num_ratings": m_stats["movie_num_ratings"],
                "user_avg_rating": u_stats["user_avg_rating"],
                "user_num_ratings": u_stats["user_num_ratings"],
                "rating_diff": u_stats["user_avg_rating"] - m_stats["movie_avg_rating"],
                "popularity_score": m_stats["movie_num_ratings"] * m_stats["movie_avg_rating"],
                **m_genres,
                **u_genre_features
            })

        feature_cols = ["retrieval_score", "movie_avg_rating", "movie_num_ratings", 
                        "user_avg_rating", "user_num_ratings", "rating_diff", "popularity_score"] \
                       + [f"is_{g}" for g in TOP_GENRES] + [f"user_avg_{g}" for g in TOP_GENRES]
        
        X_rank = pd.DataFrame(features).reindex(columns=feature_cols, fill_value=0)
        
        scores = self.ranker.predict(X_rank)

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