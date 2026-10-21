import pytest
from fastapi import status

def test_recommend_success(client, mock_recommender):
    """Проверка успешного ответа API."""
    response = client.get("/api/v1/recommend/1?top_n=5")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "The Matrix"
    # Проверяем, что мок был вызван с правильными аргументами
    mock_recommender.recommend.assert_called_once_with(1, 5)

@pytest.mark.parametrize("top_n", [1, 10, 50])
def test_recommend_top_n_parameter(client, top_n):
    """Параметризованный тест: проверяем валидацию разных top_n."""
    response = client.get(f"/api/v1/recommend/1?top_n={top_n}")
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.parametrize("invalid_top_n", [0, 51, -1])
def test_recommend_invalid_top_n(client, invalid_top_n):
    """Параметризованный тест: проверяем, что API отклоняет неверные top_n (422)."""
    response = client.get(f"/api/v1/recommend/1?top_n={invalid_top_n}")
    # FastAPI автоматически возвращает 422 Unprocessable Entity при ошибке валидации
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_recommend_user_not_found(client, mock_recommender):
    """Проверка обработки ошибки, если пользователь не найден."""
    # Настраиваем мок так, чтобы он выбрасывал ValueError
    mock_recommender.recommend.side_effect = ValueError("User not found")
    
    response = client.get("/api/v1/recommend/999")
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Пользователь не найден"

def test_recommend_cache_hit(client, mock_redis):
    """Проверка логики: если данные есть в Redis, модель не вызывается."""
    # Настраиваем мок Redis так, чтобы он возвращал закэшированные данные
    mock_redis.get.return_value = '[{"movieId": 99, "title": "Cached Movie", "genres": "Action", "score": 0.8}]'
    
    response = client.get("/api/v1/recommend/1")
    
    assert response.status_code == status.HTTP_200_OK
    assert response.json()[0]["title"] == "Cached Movie"
    # Убеждаемся, что пайплайн НЕ вызывался
    client.app.state.recommender.recommend.assert_not_called()