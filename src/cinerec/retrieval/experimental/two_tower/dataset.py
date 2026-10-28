"""
PyTorch Dataset для Two-Tower retrieval.

Формирует батчи (user, positive_item) и использует in-batch negatives:
для каждого user остальные N-1 items в батче считаются негативами.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class TwoTowerDataset(Dataset):
    """
    Dataset с implicit feedback: rating >= 4.0 считается positive.

    Для каждого пользователя хранит список положительных movie_id.
    При вызове __getitem__ возвращает (user_idx, positive_movie_idx).
    In-batch negatives формируются автоматически на уровне loss.
    """

    def __init__(self, ratings_path: Path = DATA_DIR / "ratings.csv", min_rating: float = 4.0):
        logger.info("Загрузка ratings...")
        df = pd.read_csv(ratings_path)[["userId", "movieId", "rating"]]

        # Оставляем только positive interactions
        positives = df[df["rating"] >= min_rating].copy()
        logger.info("Positive interactions: %d", len(positives))

        # Маппинги user/movie -> continuous index
        self.user_ids = sorted(positives["userId"].unique())
        self.movie_ids = sorted(positives["movieId"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        # Для каждого user — список positive movie_idx
        positives["user_idx"] = positives["userId"].map(self.user_to_idx)
        positives["movie_idx"] = positives["movieId"].map(self.movie_to_idx)

        self.user_positives = (
            positives.groupby("user_idx")["movie_idx"]
            .apply(list)
            .to_dict()
        )

        # Плоский список пар (user, positive) для итерации
        self.pairs = list(zip(positives["user_idx"], positives["movie_idx"]))
        logger.info("Пар (user, positive): %d", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        user_idx, movie_idx = self.pairs[idx]
        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "positive_idx": torch.tensor(movie_idx, dtype=torch.long),
        }

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_movies(self) -> int:
        return len(self.movie_ids)


def load_movie_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает content features для Item Tower и popularity distribution для LogQ.
    
    Returns:
        features: (num_movies, content_dim) — one-hot genres + year_norm.
        movie_ids: (num_movies,) — ordered movie_ids.
        item_counts: (num_movies,) — popularity (count в ratings.csv).
    """
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    movies["genres"] = movies["genres"].fillna("")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)

    all_genres = sorted({
        g for genres in movies["genres"] for g in genres.split("|") if g
    })
    logger.info("Уникальных жанров: %d", len(all_genres))

    for genre in all_genres:
        movies[f"is_{genre}"] = movies["genres"].str.contains(genre, regex=False).astype(int)

    year_norm = ((movies["year"] - 1900) / 130).clip(0, 1).fillna(0.5)

    features = pd.concat(
        [movies[[f"is_{g}" for g in all_genres]], year_norm.rename("year_norm")],
        axis=1,
    ).values.astype(np.float32)

    # LogQ correction: популярность каждого movie_id в ratings.csv
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    counts = ratings.groupby("movieId").size().to_dict()
    item_counts = np.array(
        [counts.get(mid, 0) for mid in movies["movieId"].values],
        dtype=np.float32,
    )
    
    return features, movies["movieId"].values, item_counts