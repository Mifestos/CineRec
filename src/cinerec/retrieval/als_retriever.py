import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

from cinerec.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")


class ALSRetriever(BaseRetriever):
    """Реализация Retrieval на базе матричной факторизации (ALS)."""

    def __init__(self) -> None:
        logger.info("Загрузка ALS-модели...")
        self.model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")

        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            # Приводим ключи к int для надежного поиска
            self.user_to_idx = {int(k): v for k, v in data["user_to_idx"].items()}
            self.idx_to_movie = {int(k): v for k, v in data["idx_to_movie"].items()}
            self.movie_to_idx = {int(k): v for k, v in data["movie_to_idx"].items()}
            self.sparse_user_item = data["sparse_user_item"]

    def has_user(self, user_id: int) -> bool:
        """Проверяет наличие пользователя в обученной матрице."""
        return user_id in self.user_to_idx

    def recommend(self, user_id: int, top_k: int, filter_already_liked: bool = True) -> List[Dict[str, Any]]:
        """Офлайн-рекомендации для пользователей из обучающей выборки."""
        if not self.has_user(user_id):
            return []

        user_idx = self.user_to_idx[user_id]
        item_ids, scores = self.model.recommend(
            user_idx,
            self.sparse_user_item[user_idx],
            N=top_k,
            filter_already_liked_items=filter_already_liked
        )

        return [
            {"movieId": int(self.idx_to_movie[idx]), "score": float(score)}
            for idx, score in zip(item_ids, scores)
        ]

    def recommend_for_ratings(self, user_ratings: Dict[int, float], top_k: int) -> List[Dict[str, Any]]:
        """
        Онлайн-рекомендации на основе оценок пользователя из БД.
        Позволяет генерировать кандидатов для новых пользователей (холодный старт).
        """
        movie_indices = []
        scores = []

        # Переводим внешние ID фильмов из БД во внутренние индексы ALS
        for movie_id, score in user_ratings.items():
            if movie_id in self.movie_to_idx:
                movie_indices.append(self.movie_to_idx[movie_id])
                scores.append(score)

        if not movie_indices:
            return []

        # Формируем разреженную матрицу для одного пользователя (размер 1 x N_items)
        user_items = sparse.csr_matrix(
            (scores, ([0] * len(movie_indices), movie_indices)),
            shape=(1, self.model.item_factors.shape[0])
        )

        # Вызываем ALS. 
        # userid=0 здесь используется как заглушка, так как мы передаем готовую матрицу user_items.
        # Библиотека implicit проигнорирует userid и будет использовать векторы из user_items.
        item_ids, als_scores = self.model.recommend(
            userid=0,
            user_items=user_items,
            N=top_k,
            filter_already_liked_items=True
        )

        return [
            {"movieId": int(self.idx_to_movie[idx]), "score": float(score)}
            for idx, score in zip(item_ids, als_scores)
        ]