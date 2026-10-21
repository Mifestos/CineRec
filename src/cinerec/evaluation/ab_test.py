import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from cinerec.pipeline import TwoStageRecommender
from cinerec.retrieval.als_retriever import ALSRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")


def evaluate_model(
    model: TwoStageRecommender, 
    test_users: List[int], 
    ratings: pd.DataFrame, 
    k: int = 10
) -> float:
    """Считает средний NDCG@k для переданной модели на заданных пользователях."""
    ndcg_scores = []
    
    for user_id in test_users:
        # Получаем реальные фильмы, которые юзер лайкнул (оценка >= 4)
        user_history = ratings[ratings["userId"] == user_id]
        relevant_movies = user_history[user_history["rating"] >= 4.0]["movieId"].tolist()
        
        if not relevant_movies:
            continue
            
        # Получаем рекомендации (отключаем фильтрацию, чтобы проверить, попали ли лайки в топ)
        recs = model.recommend(user_id, top_n=k, filter_already_liked=False)
        if not recs:
            ndcg_scores.append(0.0)
            continue
            
        # Формируем бинарный вектор релевантности (1, если рекомендованный фильм юзер лайкнул)
        rec_movie_ids = [r["movieId"] for r in recs]
        true_relevance = np.array([1 if m in relevant_movies else 0 for m in rec_movie_ids])
        
        # Скоры от нашей модели (для сортировки)
        pred_scores = np.array([r["score"] for r in recs])
        
        # Если в топ-k попал хотя бы один релевантный фильм, считаем NDCG
        if true_relevance.sum() > 0:
            score = ndcg_score([true_relevance], [pred_scores], k=k)
            ndcg_scores.append(score)
        else:
            ndcg_scores.append(0.0)
            
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def run_ab_test(num_users: int = 500, k: int = 10) -> None:
    """Запускает Offline A/B тест между классификатором и LTR-ранжировщиком."""
    logger.info("Загрузка исторических данных...")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    
    # Берем случайных активных пользователей (у которых >= 5 оценок)
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts[user_counts >= 5].index.tolist()
    test_users = np.random.choice(active_users, size=num_users, replace=False).tolist()
    
    logger.info("Загрузка моделей...")
    # Ретривер (ALS) общий для обеих групп
    retriever = ALSRetriever() 
    
    with open(MODEL_DIR / "ranker_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(MODEL_DIR / "ranker_ltr.pkl", "rb") as f:
        ltr_ranker = pickle.load(f)
        
    logger.info("Инициализация пайплайнов (Group A и Group B)...")
    model_baseline = TwoStageRecommender(retriever=retriever, ranker=classifier)
    model_challenger = TwoStageRecommender(retriever=retriever, ranker=ltr_ranker)
    
    logger.info("Оценка Group A (Baseline - LGBMClassifier) на %s пользователях...", len(test_users))
    ndcg_baseline = evaluate_model(model_baseline, test_users, ratings, k)
    
    logger.info("Оценка Group B (Challenger - LGBMRanker) на %s пользователях...", len(test_users))
    ndcg_challenger = evaluate_model(model_challenger, test_users, ratings, k)
    
    # Защита от деления на ноль при расчете Lift
    lift = ((ndcg_challenger - ndcg_baseline) / ndcg_baseline) * 100 if ndcg_baseline > 0 else 0.0
    
    # Вывод результатов
    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ A/B ТЕСТА (NDCG@%s)", k)
    logger.info("Group A (Baseline - LGBMClassifier): %.4f", ndcg_baseline)
    logger.info("Group B (Challenger - LGBMRanker):  %.4f", ndcg_challenger)
    logger.info("Прирост качества (Lift): %+.2f%%", lift)
    logger.info("=" * 50)


if __name__ == "__main__":
    run_ab_test(num_users=500, k=10)