"""
Two-Tower модель: User Tower + Item Tower.

Обучение: in-batch negatives + InfoNCE loss.
Инференс: L2-нормализованные эмбеддинги, скалярное произведение = cosine similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Кодирует пользователя: embedding(user_id) + mean history embedding."""

    def __init__(self, num_users: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_idx: torch.Tensor, history_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx: (batch,) user ids.
            history_embeddings: (batch, embedding_dim) — mean of positive item embeddings.
        Returns:
            (batch, embedding_dim) L2-normalized user embeddings.
        """
        user_emb = self.user_embedding(user_idx)
        combined = torch.cat([user_emb, history_embeddings], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """Кодирует фильм: embedding(movie_id) + content features."""

    def __init__(
        self,
        num_movies: int,
        content_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.content_proj = nn.Linear(content_dim, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, movie_idx: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movie_idx: (batch,) movie ids.
            content_features: (batch, content_dim).
        Returns:
            (batch, embedding_dim) L2-normalized item embeddings.
        """
        movie_emb = self.movie_embedding(movie_idx)
        content_emb = self.content_proj(content_features)
        combined = torch.cat([movie_emb, content_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Полная Two-Tower модель."""

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        content_dim: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_movies, content_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward_user(
        self, user_idx: torch.Tensor, history_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.user_tower(user_idx, history_embeddings)

    def forward_item(
        self, movie_idx: torch.Tensor, content_features: torch.Tensor
    ) -> torch.Tensor:
        return self.item_tower(movie_idx, content_features)