import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TOP_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]


def _load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает сырые данные и базовые маппинги."""
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")[["userId", "movieId", "rating"]]
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    
    movies["genres"] = movies["genres"].fillna("")
    for genre in TOP_GENRES:
        movies[f"is_{genre}"] = movies["genres"].str.contains(genre).astype(int)
        
    return ratings, movies


def _calculate_retrieval_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    """Считает retrieval_score (dot product ALS) для каждой пары user-item."""
    logger.info("Расчет retrieval_score (ALS)...")
    als_model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")
    
    with open(MODEL_DIR / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        
    ratings["user_idx"] = ratings["userId"].map(mappings["user_to_idx"])
    ratings["movie_idx"] = ratings["movieId"].map(mappings["movie_to_idx"])
    ratings = ratings.dropna(subset=["user_idx", "movie_idx"])
    
    user_vecs = als_model.user_factors[ratings["user_idx"].astype(int)]
    item_vecs = als_model.item_factors[ratings["movie_idx"].astype(int)]
    
    ratings["retrieval_score"] = np.sum(user_vecs * item_vecs, axis=1)
    return ratings.drop(columns=["user_idx", "movie_idx"])


# В файле src/cinerec/ranking/features.py

def _add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет кросс-фичи и сохраняет справочники для инференса."""
    logger.info("Генерация кросс-фичей...")
    
    # 1. Глобальные статистики
    movie_stats = df.groupby("movieId").agg(
        movie_avg_rating=("rating", "mean"),
        movie_num_ratings=("rating", "count")
    )
    user_stats = df.groupby("userId").agg(
        user_avg_rating=("rating", "mean"),
        user_num_ratings=("rating", "count")
    )
    
    # СОХРАНЯЕМ СПРАВОЧНИКИ ДЛЯ PIPELINE
    movie_stats.to_parquet(PROCESSED_DIR / "movie_stats.parquet")
    user_stats.to_parquet(PROCESSED_DIR / "user_stats.parquet")
    
    # Сохраняем популярные фильмы для Cold Start
    popular_df = movie_stats[movie_stats["movie_num_ratings"] >= 500].sort_values(
        by=["movie_avg_rating", "movie_num_ratings"], ascending=[False, False]
    )
    popular_df.to_parquet(PROCESSED_DIR / "popular_movies.parquet")
    
    df = df.merge(movie_stats.reset_index(), on="movieId").merge(user_stats.reset_index(), on="userId")
    
    # 2. Кросс-признаки
    df["rating_diff"] = df["user_avg_rating"] - df["movie_avg_rating"]
    df["popularity_score"] = df["movie_num_ratings"] * df["movie_avg_rating"]
    
    # 3. Взаимодействие пользователь-жанр (и сохранение для pipeline)
    user_genre_stats = {}
    for genre in TOP_GENRES:
        is_genre_col = f"is_{genre}"
        genre_df = df[df[is_genre_col] == 1]
        user_genre_avg = genre_df.groupby("userId")["rating"].mean().rename(f"user_avg_{genre}")
        
        df = df.merge(user_genre_avg, on="userId", how="left")
        df[f"user_avg_{genre}"] = df[f"user_avg_{genre}"].fillna(df["user_avg_rating"])
        
        # Сохраняем словарь для pipeline (пока в виде parquet)
        user_genre_avg.to_frame().to_parquet(PROCESSED_DIR / f"{genre}_stats.parquet")
        
    return df

def build_features() -> None:
    """Создает датасет с признаками и все справочники для API."""
    ratings, movies = _load_base_data()
    ratings = _calculate_retrieval_scores(ratings)
    
    df = ratings.merge(movies[["movieId"] + [f"is_{g}" for g in TOP_GENRES]], on="movieId")

    df["target_binary"] = (df["rating"] >= 4.0).astype(int)
    df["target_relevance"] = (df["rating"] - 1).astype(int)
    
    df = _add_cross_features(df)

    features = [
        "userId", "movieId", "rating", "target_binary", "target_relevance", "retrieval_score",
        "movie_avg_rating", "movie_num_ratings", "user_avg_rating", "user_num_ratings",
        "rating_diff", "popularity_score"
    ] + [f"is_{g}" for g in TOP_GENRES] + [f"user_avg_{g}" for g in TOP_GENRES]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[features].to_parquet(PROCESSED_DIR / "ranking_features.parquet", index=False)
    
    logger.info("Признаки и справочники сохранены в %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_features() 