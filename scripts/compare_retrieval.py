"""
Сравнение Retrieval-моделей: Two-Tower vs ALS vs Popular.
Правильный протокол: random split ТОЛЬКО среди positive interactions (rating >= 4.0).
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cinerec.retrieval.als_retriever import ALSRetriever
from cinerec.retrieval.two_tower.retriever import TwoTowerRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def evaluate_model(model, test_users, positives_by_user, k=50, model_name="model"):
    """
    Правильный протокол:
    - Для каждого пользователя positives (rating >= 4.0) делятся 80/20.
    - Train = 80% positives → истории модели.
    - Test = 20% positives → что модель должна найти.
    - Модель получает только train, исключает то, что уже видела.
    """
    hits_total = 0
    relevant_total = 0
    hits_users = 0

    for user_id in test_users:
        positives = positives_by_user.get(user_id, [])
        if len(positives) < 5:
            continue

        # Random split ТОЛЬКО positives
        random.shuffle(positives)
        split = int(len(positives) * 0.8)
        train_positives = positives[:split]
        test_positives = positives[split:]

        if not test_positives:
            continue

        # Модель знает только train_positives
        user_ratings = {mid: 5.0 for mid in train_positives}
        recs = model.recommend_for_ratings(user_ratings, top_k=k)

        if recs:
            rec_ids = {c["movieId"] for c in recs}
            hits = len(rec_ids & set(test_positives))
            hits_total += hits
            if hits > 0:
                hits_users += 1
        relevant_total += len(test_positives)

    hit_rate = 100 * hits_total / relevant_total if relevant_total else 0
    user_coverage = 100 * hits_users / len(test_users) if len(test_users) > 0 else 0

    logger.info(
        "%s — hit_rate=%.2f%% | users_with_hits=%d/%d (%.1f%%)",
        model_name, hit_rate, hits_users, len(test_users), user_coverage,
    )
    return hit_rate


def main():
    np.random.seed(42)
    random.seed(42)

    logger.info("Загрузка данных...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")
    popular_ids = pd.read_parquet(PROCESSED_DIR / "popular_movies.parquet").index.tolist()

    # Собираем positives (rating >= 4.0) по каждому пользователю
    positives_by_user = (
        df[df["rating"] >= 4.0]
        .groupby("userId")["movieId"]
        .apply(list)
        .to_dict()
    )

    # 200 пользователей с >= 5 positives
    eligible_users = [uid for uid, p in positives_by_user.items() if len(p) >= 5]
    test_users = np.random.choice(eligible_users, size=200, replace=False).tolist()
    logger.info("Оценка на %d пользователях, k=50", len(test_users))

    # --- Popular baseline (тоже на positives split) ---
    hits = 0
    rel = 0
    for user_id in test_users:
        positives = positives_by_user[user_id]
        random.shuffle(positives)
        split = int(len(positives) * 0.8)
        test_positives = positives[split:]
        if test_positives:
            hits += len(set(popular_ids[:50]) & set(test_positives))
            rel += len(test_positives)
    logger.info("Popular baseline — hit_rate=%.2f%%", 100 * hits / rel if rel else 0)

    # --- ALS ---
    als = ALSRetriever()
    evaluate_model(als, test_users, positives_by_user, k=50, model_name="ALS")

    # --- Two-Tower ---
    tt = TwoTowerRetriever(device="cpu")
    evaluate_model(tt, test_users, positives_by_user, k=50, model_name="Two-Tower")


if __name__ == "__main__":
    main()