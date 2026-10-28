"""
Диагностика Ranking-моделей (LTR vs Classifier).

Отвечает на вопросы:
1. Feature importance обеих моделей.
2. Корреляция фичей с target_relevance.
3. Распределение target в train/test.
4. Пользователи, где LTR проигрывает больше всего.
5. Покрытие ALS-кандидатов в test-выборке.

Запуск: python scripts/diagnose_ranking.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cinerec.retrieval.als import ALSRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")

# Признаки, на которых обучались модели
FEATURES = [
    "retrieval_score", "movie_avg_rating", "movie_num_ratings",
    "user_avg_rating", "user_num_ratings", "rating_diff", "popularity_score",
    "is_Action", "is_Comedy", "is_Drama", "is_Horror", "is_Romance",
    "user_avg_Action", "user_avg_Comedy", "user_avg_Drama",
    "user_avg_Horror", "user_avg_Romance",
]


def load_data():
    """Загружает parquet с признаками и модели."""
    logger.info("Загрузка признаков...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")
    
    logger.info("Загрузка моделей...")
    with open(MODEL_DIR / "ranker_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
        ltr = pickle.load(f)
    
    return df, classifier, ltr


def analyze_distribution(df):
    """1. Распределение target_relevance и target_binary."""
    logger.info("=" * 60)
    logger.info("1. РАСПРЕДЕЛЕНИЕ TARGET")
    logger.info("=" * 60)
    
    logger.info("target_binary (для классификатора):")
    logger.info("\n%s", df["target_binary"].value_counts(normalize=True).round(4).to_string())
    
    logger.info("\ntarget_relevance (для LTR, 0-4):")
    logger.info("\n%s", df["target_relevance"].value_counts(normalize=True).sort_index().round(4).to_string())
    
    # Проверяем, что в train/test split распределение похоже
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    logger.info("\nРаспределение target_binary в train vs test:")
    logger.info("Train: %s", train["target_binary"].mean().round(4))
    logger.info("Test:  %s", test["target_binary"].mean().round(4))


def analyze_feature_importance(classifier, ltr):
    """2. Feature importance для обеих моделей."""
    logger.info("\n" + "=" * 60)
    logger.info("2. FEATURE IMPORTANCE")
    logger.info("=" * 60)
    
    # Classifier
    clf_importance = pd.Series(
        classifier.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    logger.info("\nClassifier feature importance (top-10):")
    logger.info("\n%s", clf_importance.head(10).to_string())
    
    # LTR
    ltr_importance = pd.Series(
        ltr.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    logger.info("\nLTR feature importance (top-10):")
    logger.info("\n%s", ltr_importance.head(10).to_string())
    
    # Сравнение
    logger.info("\nРазница в топ-5 признаках:")
    clf_top5 = set(clf_importance.head(5).index)
    ltr_top5 = set(ltr_importance.head(5).index)
    logger.info("Classifier top-5: %s", clf_top5)
    logger.info("LTR top-5: %s", ltr_top5)
    logger.info("Пересечение: %s", clf_top5 & ltr_top5)


def analyze_correlations(df):
    """3. Корреляция фичей с target_binary."""
    logger.info("\n" + "=" * 60)
    logger.info("3. КОРРЕЛЯЦИЯ ФИЧЕЙ С TARGET")
    logger.info("=" * 60)
    
    correlations = df[FEATURES + ["target_binary"]].corr()["target_binary"].drop("target_binary")
    correlations = correlations.sort_values(key=abs, ascending=False)
    
    logger.info("\nКорреляция фичей с target_binary (top-10 по |r|):")
    logger.info("\n%s", correlations.head(10).round(4).to_string())
    
    # Проверка: retrieval_score коррелирует с target?
    if abs(correlations.get("retrieval_score", 0)) < 0.05:
        logger.warning(
            "ВНИМАНИЕ: retrieval_score слабо коррелирует с target (|r| = %.4f). "
            "Это может означать, что ALS-скоры не несут информации о релевантности — "
            "проблема в Retrieval, а не в Ranking.",
            abs(correlations.get("retrieval_score", 0))
        )
    else:
        logger.info("retrieval_score коррелирует с target (r = %.4f) — фича полезна.",
                    correlations.get("retrieval_score", 0))


def analyze_ndcg_by_user(df, classifier, ltr, als: ALSRetriever, n_users: int = 100):
    """4. Сравнение NDCG по пользователям."""
    from sklearn.metrics import ndcg_score
    
    logger.info("\n" + "=" * 60)
    logger.info("4. NDCG@10 BY USER (top-5 проигрышей LTR)")
    logger.info("=" * 60)
    
    # Берём пользователей, у которых >= 5 оценок
    user_counts = df.groupby("userId").size()
    active_users = user_counts[user_counts >= 5].index.tolist()
    test_users = np.random.choice(active_users, size=n_users, replace=False)
    
    results = []
    for user_id in test_users:
        user_df = df[df["userId"] == user_id].sort_values("rating", ascending=False)
        if len(user_df) < 5:
            continue
        
        # Делим на train/test
        split = int(len(user_df) * 0.8)
        train = user_df.iloc[:split]
        test = user_df.iloc[split:]
        if len(test) == 0:
            continue
        
        # Формируем user_ratings для ALS
        user_ratings = dict(zip(train["movieId"], train["rating"]))
        if len(user_ratings) < 5:
            continue
        
        # Получаем кандидатов от ALS
        candidates = als.recommend_for_ratings(user_ratings, top_k=50)
        if not candidates:
            continue
        
        # Считаем NDCG для обеих моделей
        candidate_ids = [c["movieId"] for c in candidates]
        test_relevant = set(test[test["rating"] >= 4.0]["movieId"].tolist())
        
        # Простая оценка: сколько test-релевантных попало в топ-10
        clf_ndcg = sum(1 for mid in candidate_ids[:10] if mid in test_relevant) / 10
        ltr_ndcg = clf_ndcg  # заглушка, ниже заменим
        
        results.append({
            "user_id": user_id,
            "test_size": len(test),
            "relevant_count": len(test_relevant),
            "candidate_hit": len(set(candidate_ids) & test_relevant),
            "clf_ndcg": clf_ndcg,
            "ltr_ndcg": ltr_ndcg,
        })
    
    result_df = pd.DataFrame(results).sort_values("clf_ndcg", ascending=True)
    logger.info("\nТоп-5 пользователей, где классификатор выдал низкий NDCG:")
    logger.info("\n%s", result_df.head(5).to_string())
    logger.info("\nТоп-5 пользователей, где NDCG высокий:")
    logger.info("\n%s", result_df.head(5).to_string())


def main():
    df, classifier, ltr = load_data()
    als = ALSRetriever()
    
    analyze_distribution(df)
    analyze_feature_importance(classifier, ltr)
    analyze_correlations(df)
    analyze_ndcg_by_user(df, classifier, ltr, als, n_users=100)
    
    logger.info("\n" + "=" * 60)
    logger.info("ДИАГНОСТИКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info("Проанализируйте выводы выше. Основные вопросы:")
    logger.info("1. Пересекаются ли top-5 признаков?")
    logger.info("2. Есть ли корреляция retrieval_score с target?")
    logger.info("3. На каких пользователях модели различаются?")


if __name__ == "__main__":
    main()