import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
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
            self.user_to_idx = {int(k): v for k, v in data["user_to_idx"].items()}
            self.idx_to_movie = {int(k): v for k, v in data["idx_to_movie"].items()}
            self.sparse_user_item = data["sparse_user_item"]

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_to_idx

    def recommend(self, user_id: int, top_k: int, filter_already_liked: bool = True) -> List[Dict[str, Any]]:
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