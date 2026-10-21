import pytest
from cinerec.pipeline import TwoStageRecommender

# Помечаем тест как медленный (Integration)
@pytest.fixture(scope="module")
def real_pipeline():
    # Загружает реальные модели из папки models/
    return TwoStageRecommender()

@pytest.mark.integration
def test_real_pipeline_output(real_pipeline):
    """Интеграционный тест: проверяем реальный вывод ML-модели."""
    results = real_pipeline.recommend(user_id=1, top_n=5)
    
    assert isinstance(results, list)
    assert len(results) == 5
    
    first_movie = results[0]
    assert "movieId" in first_movie
    assert "title" in first_movie
    assert "genres" in first_movie
    assert isinstance(first_movie["score"], float)
    # Убеждаемся, что оценки не выходят за пределы [0, 1]
    assert 0.0 <= first_movie["score"] <= 1.0

@pytest.mark.integration
def test_real_pipeline_invalid_user(real_pipeline):
    """Интеграционный тест: проверка несуществующего пользователя."""
    with pytest.raises(ValueError, match="not found"):
        real_pipeline.recommend(user_id=-1)