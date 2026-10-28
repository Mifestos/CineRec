"""
Диагностика Retrieval-слоя (ALS).

Проверяет:
1. Что ALS возвращает для пользователей с разной длиной истории.
2. Сравнивает ALS-кандидатов с baseline-popular.
3. Считает hit rate: сколько ALS-кандидатов попадает в test_relevant.
4. Сравнивает с "случайным" retrieval (случайные 50 фильмов).

Запуск: python scripts/diagnose_retrieval.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cinerec.retrieval.als import ALSRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
DATA_DIR = Path("data/raw")


def load_popular_movies():
    """Загружает популярные фильмы из parquet."""
    return pd.read_parquet(PROCESSED_DIR / "popular_movies.parquet").index.tolist()


def evaluate_retrieval(df, als: ALSRetriever, n_users=200, k=50):
    """
    Для каждого пользователя:
    - Делит на train (80%) / test (20%).
    - Получает ALS-кандидатов, popular-кандидатов и случайных кандидатов.
    - Считает hit rate: сколько test-релевантных попало в топ-k.

    Returns:
        df с метриками: user_id, test_relevant_count,
        als_hits, popular_hits, random_hits
    """
    # Фиксируем seed для воспроизводимости
    np.random.seed(42)
    random.seed(42)

    popular_ids = load_popular_movies()
    all_movie_ids = list(als.movie_to_idx.keys())

    user_counts = df.groupby("userId").size()
    active_users = user_counts[user_counts >= 5].index.tolist()
    test_users = np.random.choice(active_users, size=n_users, replace=False)

    results = []
    for user_id in test_users:
        user_df = df[df["userId"] == user_id].sort_values("rating", ascending=False)
        if len(user_df) < 5:
            continue

        split = int(len(user_df) * 0.8)
        train = user_df.iloc[:split]
        test = user_df.iloc[split:]

        if len(test) == 0:
            continue

        test_relevant = set(test[test["rating"] >= 4.0]["movieId"].tolist())
        if not test_relevant:
            continue

        # ALS retrieval
        user_ratings = dict(zip(train["movieId"], train["rating"]))
        als_candidates = als.recommend_for_ratings(user_ratings, top_k=k)
        als_hits = len({c["movieId"] for c in als_candidates} & test_relevant) if als_candidates else 0

        # Popular baseline
        popular_hits = len(set(popular_ids[:k]) & test_relevant)

        # Random baseline
        random_ids = random.sample(all_movie_ids, k)
        random_hits = len(set(random_ids) & test_relevant)

        results.append({
            "user_id": user_id,
            "train_size": len(train),
            "test_relevant_count": len(test_relevant),
            "als_hits": als_hits,
            "popular_hits": popular_hits,
            "random_hits": random_hits,
        })

    return pd.DataFrame(results)


def main():
    logger.info("Загрузка данных...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")
    als = ALSRetriever()

    logger.info("Оценка retrieval (200 пользователей, k=50)...")
    results = evaluate_retrieval(df, als, n_users=200, k=50)

    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ: HIT RATE @ 50")
    logger.info("=" * 60)
    logger.info("\nСредние значения:")
    logger.info("  Test-relevant на пользователя: %.2f", results["test_relevant_count"].mean())
    logger.info("  ALS hits:                      %.3f (%.1f%%)",
                results["als_hits"].mean(),
                100 * results["als_hits"].sum() / results["test_relevant_count"].sum())
    logger.info("  Popular hits:                  %.3f (%.1f%%)",
                results["popular_hits"].mean(),
                100 * results["popular_hits"].sum() / results["test_relevant_count"].sum())
    logger.info("  Random hits:                   %.3f (%.1f%%)",
                results["random_hits"].mean(),
                100 * results["random_hits"].sum() / results["test_relevant_count"].sum())

    logger.info("\nСколько пользователей с ALS hits > 0:")
    logger.info("  %d из %d (%.1f%%)",
                (results["als_hits"] > 0).sum(),
                len(results),
                100 * (results["als_hits"] > 0).mean())

    logger.info("\nСколько пользователей с Popular hits > 0:")
    logger.info("  %d из %d (%.1f%%)",
                (results["popular_hits"] > 0).sum(),
                len(results),
                100 * (results["popular_hits"] > 0).mean())

    logger.info("\nТоп-10 пользователей, где ALS hits = 0:")
    logger.info("\n%s",
                results[results["als_hits"] == 0][["user_id", "train_size", "test_relevant_count"]].head(10).to_string())

    logger.info("\nВЫВОДЫ:")
    if results["als_hits"].sum() < results["popular_hits"].sum():
        logger.warning(
            "ALS ХУЖЕ BASELINE-POPULAR. Проблема в Retrieval, а не в Ranking. "
            "Нужно либо переобучить ALS с большим factors, либо перейти на Two-Tower."
        )
    elif results["als_hits"].sum() == 0:
        logger.error(
            "ALS ВОЗВРАЩАЕТ 0 РЕЛЕВАНТНЫХ КАНДИДАТОВ. "
            "Проблема критическая: либо маппинг movie_id сломан, либо ALS-модель не работает."
        )
    else:
        logger.info(
            "ALS выдаёт какие-то релевантные кандидаты, но мало. "
            "Можно попробовать улучшить Ranking, а не Retrieval."
        )


if __name__ == "__main__":
    main()