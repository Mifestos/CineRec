import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # Отключаем внутренний пул потоков OpenBLAS
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

def train_model() -> None:
    """Обучает ALS-модель и сохраняет веса с маппингами."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
    
    # Кодируем ID в непрерывные индексы (библиотека implicit требует индексы от 0 до N)
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()
    
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    
    ratings["user_idx"] = ratings["userId"].map(user_to_idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)
    
    # Формируем разреженную матрицу (User x Item) для экономии памяти
    sparse_user_item = sparse.csr_matrix(
        (ratings["rating"].astype(np.float32), (ratings["user_idx"], ratings["movie_idx"]))
    )
    
    # Инициализируем и обучаем модель (64 скрытых фактора, 15 итераций)
    model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=15)
    model.fit(sparse_user_item)
    
    # Сохраняем веса модели
    model.save(MODEL_DIR / "als_model.npz")
    
    # Сохраняем маппинги для обратного преобразования ID
    with open(MODEL_DIR / "mappings.pkl", "wb") as f:
        pickle.dump({
            "user_to_idx": user_to_idx,
            "movie_to_idx": movie_to_idx,
            "idx_to_movie": idx_to_movie,
            "sparse_user_item": sparse_user_item
        }, f)
        
    print("Модель обучена. Артефакты сохранены в директорию models/")

if __name__ == "__main__":
    train_model()