import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
from pathlib import Path

import pandas as pd
from implicit.als import AlternatingLeastSquares

MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")

class ALSRecommender:
    """Класс для генерации кандидатов с использованием обученной ALS-модели."""

    def __init__(self) -> None:
        self.model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")
        
        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            self.user_to_idx = data["user_to_idx"]
            self.idx_to_movie = data["idx_to_movie"]
            self.sparse_user_item = data["sparse_user_item"]
            
        self.movies = pd.read_csv(DATA_DIR / "movies.csv").set_index("movieId")

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Генерирует топ-N рекомендаций для заданного пользователя."""
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found.")
            
        user_idx = self.user_to_idx[user_id]
        
        # filter_already_liked_items исключает фильмы из истории пользователя
        item_ids, scores = self.model.recommend(
            user_idx, 
            self.sparse_user_item[user_idx], 
            N=top_n, 
            filter_already_liked_items=True
        )
        
        movie_ids = [self.idx_to_movie[idx] for idx in item_ids]
        
        return self.movies.loc[movie_ids][["title", "genres"]].assign(score=scores)