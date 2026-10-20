import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

# Гиперпараметры модели
ALS_FACTORS = 64
ALS_ITERATIONS = 15
ALS_REG = 0.1

def train_model() -> None:
    """Обучает ALS-модель и сохраняет артефакты для инференса."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Загрузка данных...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
    
    # Маппинг ID в непрерывные индексы (требование implicit)
    user_to_idx = {u: i for i, u in enumerate(ratings["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(ratings["movieId"].unique())}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    
    ratings["user_idx"] = ratings["userId"].map(user_to_idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)
    
    # Разреженная матрица User x Item
    sparse_user_item = sparse.csr_matrix(
        (ratings["rating"].astype(np.float32), (ratings["user_idx"], ratings["movie_idx"]))
    )
    
    logger.info(f"Обучение ALS (factors={ALS_FACTORS}, iterations={ALS_ITERATIONS})...")
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS, 
        regularization=ALS_REG, 
        iterations=ALS_ITERATIONS
    )
    model.fit(sparse_user_item)
    
    model.save(MODEL_DIR / "als_model.npz")
    
    with open(MODEL_DIR / "mappings.pkl", "wb") as f:
        pickle.dump({
            "user_to_idx": user_to_idx,
            "movie_to_idx": movie_to_idx,  
            "idx_to_movie": idx_to_movie,
            "sparse_user_item": sparse_user_item
        }, f)
        
    logger.info("Модель обучена и сохранена в models/.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_model()