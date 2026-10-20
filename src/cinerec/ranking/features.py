import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
TOP_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]

def build_features() -> None:
    """Создает датасет с признаками, включая ALS-скор."""
    logger.info("Загрузка исходных данных...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    
    logger.info("Расчет ALS-скоров...")
    # Загружаем ALS модель
    als_model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")
    user_factors = als_model.user_factors
    item_factors = als_model.item_factors
    
    # Загружаем маппинги (нужны, чтобы сопоставить ID из CSV с индексами модели)
    import pickle
    with open(MODEL_DIR / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        user_to_idx = mappings["user_to_idx"]
        movie_to_idx = mappings["movie_to_idx"]

    # Добавляем индексы для матричного перемножения
    ratings["user_idx"] = ratings["userId"].map(user_to_idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx)

    # Удаляем строки, где не нашлось соответствия (на всякий случай)
    ratings = ratings.dropna(subset=["user_idx", "movie_idx"])
    
    # Считаем скалярное произведение (dot product) для каждой пары (User, Item)
    # Это и есть предсказанный рейтинг от ALS
    user_vecs = user_factors[ratings["user_idx"].astype(int)]
    item_vecs = item_factors[ratings["movie_idx"].astype(int)]
    ratings["als_score"] = np.sum(user_vecs * item_vecs, axis=1)

    logger.info("Добавление глобальных статистик...")
    movie_stats = ratings.groupby("movieId").agg(
        movie_avg_rating=("rating", "mean"),
        movie_num_ratings=("rating", "count")
    ).reset_index()

    user_stats = ratings.groupby("userId").agg(
        user_avg_rating=("rating", "mean"),
        user_num_ratings=("rating", "count")
    ).reset_index()

    df = ratings.merge(movie_stats, on="movieId").merge(user_stats, on="userId")

    movies["genres"] = movies["genres"].fillna("")
    for genre in TOP_GENRES:
        movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)

    df = df.merge(movies[["movieId"] + [f"is_{g}" for g in TOP_GENRES]], on="movieId")
    df["target"] = (df["rating"] >= 4.0).astype(int)

    features = [
        "userId", "movieId", "rating", "target", "als_score", # <-- Добавили ALS!
        "movie_avg_rating", "movie_num_ratings",
        "user_avg_rating", "user_num_ratings"
    ] + [f"is_{g}" for g in TOP_GENRES]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[features].to_parquet(PROCESSED_DIR / "ranking_features.parquet", index=False)
    
    logger.info(f"Признаки сохранены. Строк: {len(df):,}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_features()