import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

@pytest.fixture
def mock_recommender():
    """Мок для ML-пайплайна (чтобы не грузить реальные модели)."""
    mock = MagicMock()
    # Имитируем успешный ответ
    mock.recommend.return_value = [
        {"movieId": 1, "title": "The Matrix", "genres": "Sci-Fi", "score": 0.99}
    ]
    return mock

@pytest.fixture
def mock_redis():
    """Мок для асинхронного клиента Redis."""
    mock = AsyncMock()
    mock.get.return_value = None  # По умолчанию кэш пуст (Cache Miss)
    return mock

@pytest.fixture
def client(mock_recommender, mock_redis):
    """Тестовый клиент FastAPI с подмененными зависимостями."""
    # Патчим инициализацию внутри lifespan, чтобы не вызывать реальные классы
    with patch("cinerec.api.main.TwoStageRecommender", return_value=mock_recommender), \
         patch("cinerec.api.main.get_redis_client", return_value=mock_redis):
        
        from cinerec.api.main import app
        with TestClient(app) as test_client:
            # Привязываем моки к состоянию приложения
            app.state.recommender = mock_recommender
            app.state.redis = mock_redis
            yield test_client