"""
TwoTowerRetriever — реализация BaseRetriever на обученной Two-Tower модели.

Инференс:
1. Загружаем user_embeddings через forward_user (с историей из БД).
2. Скалярное произведение с item_embeddings (предвычисленными).
3. Top-K через argpartition (O(N)).
"""
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cinerec.retrieval.base import BaseRetriever
from cinerec.retrieval.experimental.two_tower.model import TwoTowerModel

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/two_tower")
DATA_DIR = Path("data/raw")


class TwoTowerRetriever(BaseRetriever):
    """Retrieval на базе обученной Two-Tower модели."""

    def __init__(self, embedding_dim: int = 64, device: str = "cpu") -> None:
        self.device = torch.device(device)
        
        logger.info("Загрузка Two-Tower mappings...")
        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            self.user_to_idx = data["user_to_idx"]
            self.movie_to_idx = data["movie_to_idx"]
            self.idx_to_movie = data["idx_to_movie"]
            self.content_features = torch.tensor(data["content_features"], dtype=torch.float32)

        # Загружаем item embeddings (предвычисленные)
        logger.info("Загрузка item embeddings...")
        self.item_embeddings = torch.tensor(
            np.load(MODEL_DIR / "item_embeddings.npy"),
            dtype=torch.float32,
        ).to(self.device)

        # Восстанавливаем модель
        logger.info("Загрузка модели Two-Tower...")
        self.model = TwoTowerModel(
            num_users=len(self.user_to_idx),
            num_movies=len(self.movie_to_idx),
            content_dim=self.content_features.shape[1],
            embedding_dim=embedding_dim,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(MODEL_DIR / "model.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        logger.info("TwoTowerRetriever готов. Items: %d", len(self.movie_to_idx))

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_to_idx

    def _get_user_embedding(
        self,
        user_id: int,
        user_ratings: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """
        Возвращает L2-normalized user embedding (1, dim).
        
        Если у пользователя есть ratings из БД — используем mean history embedding.
        Иначе — только user_embedding из модели.
        """
        with torch.no_grad():
            if user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
            else:
                # Новый пользователь — используем user_idx=0 как заглушку,
                # сигнал будет только из истории
                user_idx = 0

            # История: mean of item embeddings по оценкам >= 4.0
            history_emb = torch.zeros(1, self.item_embeddings.shape[1], device=self.device)
            if user_ratings:
                positive_movie_ids = [mid for mid, score in user_ratings.items() if score >= 4.0]
                positive_idxs = [
                    self.movie_to_idx[mid] for mid in positive_movie_ids
                    if mid in self.movie_to_idx
                ]
                if positive_idxs:
                    history_emb[0] = self.item_embeddings[positive_idxs].mean(dim=0)

            user_idx_tensor = torch.tensor([user_idx], device=self.device)
            user_emb = self.model.forward_user(user_idx_tensor, history_emb)
            return user_emb

    def recommend_for_ratings(
        self, user_ratings: dict[int, float], top_k: int
    ) -> list[dict[str, Any]]:
        """Онлайн-рекомендации на основе оценок пользователя."""
        # Используем user_id из словаря (первый ключ)
        user_id = next(iter(user_ratings), 0)
        return self._recommend_internal(user_id, user_ratings, top_k)

    def recommend(
        self, user_id: int, top_k: int, filter_already_liked: bool = True
    ) -> list[dict[str, Any]]:
        """Офлайн-рекомендации для пользователя из датасета (без истории из БД)."""
        return self._recommend_internal(user_id, None, top_k, filter_already_liked)

    def _recommend_internal(
        self,
        user_id: int,
        user_ratings: dict[int, float] | None,
        top_k: int,
        filter_already_liked: bool = True,
    ) -> list[dict[str, Any]]:
        """Общая логика рекомендаций."""
        user_emb = self._get_user_embedding(user_id, user_ratings)  # (1, dim)
        
        # Скалярное произведение со всеми items
        scores = (user_emb @ self.item_embeddings.T).squeeze(0)  # (num_items,)
        
        # Исключаем уже оценённые
        if filter_already_liked and user_ratings:
            for mid in user_ratings:
                if mid in self.movie_to_idx:
                    scores[self.movie_to_idx[mid]] = -float("inf")
        
        # Top-K через argpartition (быстрее полной сортировки)
        k = min(top_k, scores.size(0))
        top_indices = torch.topk(scores, k).indices.cpu().numpy()
        top_scores = scores[top_indices].cpu().numpy()
        
        return [
            {"movieId": int(self.idx_to_movie[int(idx)]), "score": float(score)}
            for idx, score in zip(top_indices, top_scores, strict=False)
        ]