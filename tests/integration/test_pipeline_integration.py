"""
Integration-тесты для проверки реальной связки:
- PostgreSQL (загруженные фильмы и оценки)
- ALS-модель (онлайн-рекомендации)
- LightGBM-ранкер (реальные веса)

Запуск: pytest tests/integration/ -m integration -v
Требования: 
  - Запущенный Docker Compose (docker compose up -d)
  - Загруженные фильмы (python -m cinerec.cli db-seed-movies)
  - Обученные модели (python src/cinerec/ranking/train_ranker.py)
"""
import pytest

from cinerec.pipeline import TwoStageRecommender


@pytest.mark.integration
def test_pipeline_returns_recommendations():
    """Реальный пайплайн (ALS + LTR) возвращает непустой список рекомендаций."""
    recommender = TwoStageRecommender()
    
    # Пользователь с историей (например, userId=1 из MovieLens)
    user_ratings = {1: 5.0, 2: 4.5, 3: 4.0, 4: 5.0, 5: 4.5}
    results = recommender.recommend(user_ratings, top_n=5)
    
    assert isinstance(results, list)
    assert len(results) == 5
    assert all("movieId" in r and "title" in r and "score" in r for r in results)
    # Проверяем, что скоры разные (LTR работает, а не возвращает константу)
    scores = [r["score"] for r in results]
    assert len(set(scores)) > 1, "All scores are identical — LTR may be broken"


@pytest.mark.integration
def test_pipeline_cold_start():
    """Cold Start: если оценок < 5, возвращается витрина популярных фильмов."""
    recommender = TwoStageRecommender()
    results = recommender.recommend({1: 5.0}, top_n=5)
    
    assert len(results) == 5
    # В Cold Start скоры — это средний рейтинг / 5.0 (от 0 до 1)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)


@pytest.mark.integration
def test_pipeline_unknown_movies():
    """Если пользователь оценил несуществующие фильмы — не падаем, возвращаем fallback."""
    recommender = TwoStageRecommender()
    fake_ratings = {999_999: 5.0, 888_888: 4.0, 777_777: 5.0, 666_666: 4.0, 555_555: 5.0}
    results = recommender.recommend(fake_ratings, top_n=5)
    
    # ALS не найдёт кандидатов, сработает fallback
    assert len(results) == 5