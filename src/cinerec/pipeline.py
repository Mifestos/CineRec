import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from implicit.als import AlternatingLeastSquares

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
TOP_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]
DEFAULT_USER_STATS = {"user_avg_rating": 3.5, "user_num_ratings": 0}
DEFAULT_MOVIE_STATS = {"movie_avg_rating": 3.5, "movie_num_ratings": 0}


class TwoStageRecommender:
    """Двухэтапная рекомендательная система (ALS Retrieval -> LightGBM Ranking)."""

    def __init__(self) -> None:
        logger.info("Инициализация пайплайна: загрузка моделей и справочников...")
        
        self.als_model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")
        
        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            # Принудительно приводим ключи к int для стабильного маппинга из API
            self.user_to_idx = {int(k): v for k, v in data["user_to_idx"].items()}
            self.idx_to_movie = {int(k): v for k, v in data["idx_to_movie"].items()}
            self.sparse_user_item = data["sparse_user_item"]
            
        with open(MODEL_DIR / "ranker.pkl", "rb") as f:
            self.ranker = pickle.load(f)

        self._load_reference_data()

    def _load_reference_data(self) -> None:
        """Загружает и подготавливает статистики пользователей и фильмов."""
        ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
        
        # Словари со статистикой
        self.movie_stats = ratings.groupby("movieId").agg(
            movie_avg_rating=("rating", "mean"),
            movie_num_ratings=("rating", "count")
        ).to_dict(orient="index")
        
        self.user_stats = ratings.groupby("userId").agg(
            user_avg_rating=("rating", "mean"),
            user_num_ratings=("rating", "count")
        ).to_dict(orient="index")

        # Метаданные фильмов
        movies = pd.read_csv(DATA_DIR / "movies.csv")
        movies["genres"] = movies["genres"].fillna("")
        for genre in TOP_GENRES:
            movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)
            
        self.movie_genres = movies.set_index("movieId")[ [f"is_{g}" for g in TOP_GENRES] ].to_dict(orient="index")
        self.movie_titles = movies.set_index("movieId")[["title", "genres"]].to_dict(orient="index")

    def recommend(self, user_id: int, top_n: int = 10, retrieval_k: int = 50) -> List[Dict[str, Any]]:
        """Возвращает топ-N рекомендаций для заданного пользователя."""
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in training data.")

        user_idx = self.user_to_idx[user_id]

        # 1. RETRIEVAL: получаем кандидатов
        item_ids, als_scores = self.als_model.recommend(
            user_idx, 
            self.sparse_user_item[user_idx], 
            N=retrieval_k, 
            filter_already_liked_items=True
        )
        
        # Приводим к int, чтобы избежать ошибок с numpy.int64 при поиске в словарях
        movie_ids = [int(self.idx_to_movie[idx]) for idx in item_ids]

        # 2. RANKING: генерация фичей
        u_stats = self.user_stats.get(user_id, DEFAULT_USER_STATS)
        features = []
        
        for movie_id, als_score in zip(movie_ids, als_scores):
            m_stats = self.movie_stats.get(movie_id, DEFAULT_MOVIE_STATS)
            m_genres = self.movie_genres.get(movie_id, {f"is_{g}": 0 for g in TOP_GENRES})
            
            features.append({
                "als_score": float(als_score),
                "movie_avg_rating": float(m_stats["movie_avg_rating"]),
                "movie_num_ratings": float(m_stats["movie_num_ratings"]),
                "user_avg_rating": float(u_stats["user_avg_rating"]),
                "user_num_ratings": float(u_stats["user_num_ratings"]),
                **m_genres
            })

        # 3. Инференс LightGBM
        # Явно указываем порядок колонок, чтобы LightGBM не ругался
        feature_cols = ["als_score", "movie_avg_rating", "movie_num_ratings", 
                        "user_avg_rating", "user_num_ratings"] + [f"is_{g}" for g in TOP_GENRES]
        
        X_rank = pd.DataFrame(features)[feature_cols]
        lgb_scores = self.ranker.predict_proba(X_rank)[:, 1]

        # 4. Формирование ответа
        results = []
        for movie_id, lgb_score in zip(movie_ids, lgb_scores):
            meta = self.movie_titles.get(movie_id, {"title": "Unknown", "genres": ""})
            results.append({
                "movieId": movie_id,
                "title": meta["title"],
                "genres": meta["genres"],
                "score": float(lgb_score)
            })

        # Сортировка по убыванию скора LightGBM
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]