import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

# Гиперпараметры из env (для экспериментов)
ALS_FACTORS = int(os.getenv("ALS_FACTORS", "64"))
ALS_ITERATIONS = int(os.getenv("ALS_ITERATIONS", "15"))
ALS_REG = float(os.getenv("ALS_REG", "0.1"))
USE_BM25 = os.getenv("USE_BM25", "false").lower() == "true"

BM25_K1 = 100
BM25_B = 0.8


def train_model() -> None:
    """Обучает ALS-модель. Гиперпараметры берутся из env-переменных."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        "Конфигурация: factors=%d, iterations=%d, reg=%.3f, BM25=%s",
        ALS_FACTORS, ALS_ITERATIONS, ALS_REG, USE_BM25,
    )
    
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
    
    user_to_idx = {u: i for i, u in enumerate(ratings["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(ratings["movieId"].unique())}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    
    ratings["user_idx"] = ratings["userId"].map(user_to_idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)
    
    sparse_user_item = sparse.csr_matrix(
        (ratings["rating"].astype(np.float32), (ratings["user_idx"], ratings["movie_idx"]))
    )
    
    # BM25 опционально
    if USE_BM25:
        logger.info("Применение BM25 weighting...")
        sparse_user_item = bm25_weight(sparse_user_item, K1=BM25_K1, B=BM25_B).tocsr()
    
    # IDF-веса для консистентного инференса (нужны для новых пользователей)
    item_freq = np.asarray(sparse_user_item.sum(axis=0)).flatten()
    n_users = sparse_user_item.shape[0]
    idf_weights = np.log((n_users + 1) / (item_freq + 1)).astype(np.float32)
    
    logger.info("Обучение ALS...")
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REG,
        iterations=ALS_ITERATIONS,
    )
    model.fit(sparse_user_item)
    
    # Сохраняем модель с суффиксом конфигурации для сравнения
    suffix = f"f{ALS_FACTORS}_i{ALS_ITERATIONS}_r{ALS_REG}".replace(".", "")
    if USE_BM25:
        suffix += "_bm25"
    
    model_path = MODEL_DIR / f"als_model_{suffix}.npz"
    mappings_path = MODEL_DIR / f"mappings_{suffix}.pkl"
    
    model.save(model_path)
    with open(mappings_path, "wb") as f:
        pickle.dump({
            "user_to_idx": user_to_idx,
            "movie_to_idx": movie_to_idx,
            "idx_to_movie": idx_to_movie,
            "sparse_user_item": sparse_user_item,
            "idf_weights": idf_weights,
        }, f)
    
    # Также сохраняем как основной als_model.npz для совместимости
    model.save(MODEL_DIR / "als_model.npz")
    with open(MODEL_DIR / "mappings.pkl", "wb") as f:
        pickle.dump({
            "user_to_idx": user_to_idx,
            "movie_to_idx": movie_to_idx,
            "idx_to_movie": idx_to_movie,
            "sparse_user_item": sparse_user_item,
            "idf_weights": idf_weights,
        }, f)
    
    logger.info("Модель сохранена: %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_model()