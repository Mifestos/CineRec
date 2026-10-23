"""
Offline A/B тест для сравнения классификатора и LTR-ранжировщика.
Запуск: pytest tests/evaluation/ -m evaluation -v
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from cinerec.pipeline import TwoStageRecommender
from cinerec.retrieval.als_retriever import ALSRetriever

MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")
MIN_NDCG_THRESHOLD = 0.001  # Минимальный порог "модель вообще работает"


def evaluate_model(
    model: TwoStageRecommender,
    test_users: list,
    ratings: pd.DataFrame,
    k: int = 10,
    train_ratio: float = 0.8
) -> float:
    """Считает средний NDCG@k на train/test split по каждому пользователю."""
    ndcg_scores = []
    for user_id in test_users:
        user_ratings = ratings[ratings["userId"] == user_id]
        if len(user_ratings) < 5:
            continue

        split_idx = int(len(user_ratings) * train_ratio)
        train_df = user_ratings.iloc[:split_idx]
        test_df = user_ratings.iloc[split_idx:]

        if len(test_df) == 0:
            continue

        user_ratings_dict = dict(zip(train_df["movieId"], train_df["rating"], strict=False))
        recs = model.recommend(user_ratings_dict, top_n=k)
        if not recs:
            ndcg_scores.append(0.0)
            continue

        relevant = test_df[test_df["rating"] >= 4.0]["movieId"].tolist()
        if not relevant:
            continue

        rec_ids = [r["movieId"] for r in recs]
        true_rel = np.array([1 if m in relevant else 0 for m in rec_ids])
        scores = np.array([r["score"] for r in recs])

        if true_rel.sum() > 0:
            ndcg_scores.append(ndcg_score([true_rel], [scores], k=k))
        else:
            ndcg_scores.append(0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


@pytest.mark.evaluation
def test_pipeline_smoke():
    """
    Smoke test: пайплайн работает — обе модели возвращают NDCG > 0.
    Это проверяет, что код не сломан (а не что LTR лучше).
    """
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= 5].index.tolist()
    test_users = np.random.choice(active_users, size=100, replace=False).tolist()

    retriever = ALSRetriever()
    with open(MODEL_DIR / "ranker_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
        ltr = pickle.load(f)

    baseline = evaluate_model(
        TwoStageRecommender(retriever=retriever, ranker=classifier),
        test_users, ratings, k=10
    )
    challenger = evaluate_model(
        TwoStageRecommender(retriever=retriever, ranker=ltr),
        test_users, ratings, k=10
    )

    # Проверяем только, что обе модели вообще работают (а не качество)
    assert baseline > MIN_NDCG_THRESHOLD, f"Baseline NDCG={baseline} too low"
    assert challenger > MIN_NDCG_THRESHOLD, f"Challenger NDCG={challenger} too low"


@pytest.mark.evaluation
def test_ab_ranking_report():
    """
    Информационный A/B-тест: выводит NDCG обеих моделей.
    Не падает при деградации LTR — только предупреждает (warnings.warn).
    """
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= 5].index.tolist()
    test_users = np.random.choice(active_users, size=500, replace=False).tolist()

    retriever = ALSRetriever()
    with open(MODEL_DIR / "ranker_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
        ltr = pickle.load(f)

    baseline = evaluate_model(
        TwoStageRecommender(retriever=retriever, ranker=classifier),
        test_users, ratings, k=10
    )
    challenger = evaluate_model(
        TwoStageRecommender(retriever=retriever, ranker=ltr),
        test_users, ratings, k=10
    )
    lift = ((challenger - baseline) / baseline) * 100 if baseline > 0 else 0.0

    # Отчет (печатается в консоль)
    print(f"\n{'='*50}")
    print(f"A/B Ranking Report (NDCG@10, {len(test_users)} users)")
    print(f"{'='*50}")
    print(f"Baseline (Classifier): {baseline:.4f}")
    print(f"Challenger (LTR):      {challenger:.4f}")
    print(f"Lift:                  {lift:+.2f}%")
    print(f"{'='*50}\n")

    # Мягкая проверка: предупреждаем, но не падаем
    if challenger < baseline * 0.95:
        warnings.warn(
            f"LTR деградировал на {abs(lift):.1f}% относительно классификатора. "
            f"Это не баг, а результат эксперимента. "
            f"Требуется настройка гиперпараметров или фичей.",
            UserWarning,
            stacklevel=2
        )