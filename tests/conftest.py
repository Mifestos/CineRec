import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cinerec.api.main import app
from cinerec.core.security import create_access_token, get_password_hash
from cinerec.db.models import Base, User
from cinerec.db.session import get_db

# Тестовая БД (PostgreSQL)
SQLALCHEMY_DATABASE_URL = "postgresql://cinerec_user:cinerec_password@localhost:5432/cinerec_test_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Создает схему тестовой БД один раз перед всеми тестами."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session() -> Generator:
    """Изолированная сессия с rollback после каждого теста."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def mock_redis() -> AsyncMock:
    """Мок асинхронного Redis-клиента (имитирует кэш без реального подключения)."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)  # Cache miss по умолчанию
    mock.setex = AsyncMock(return_value=True)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture(scope="function")
def mock_recommender() -> MagicMock:
    """Мок ML-пайплайна (не загружает ALS/LightGBM веса)."""
    mock = MagicMock()
    mock.recommend.return_value = [
        {"movieId": 1, "title": "Test Movie", "genres": "Action", "score": 0.99},
    ]
    return mock


@pytest.fixture(scope="function")
def client(
    db_session, 
    mock_redis: AsyncMock, 
    mock_recommender: MagicMock
) -> Generator[TestClient, None, None]:
    """
    Тестовый HTTP-клиент с подмененными зависимостями:
      - get_db → тестовая БД
      - get_redis_client → mock_redis
      - TwoStageRecommender → mock_recommender
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Патчим lifespan-зависимости ДО запуска TestClient
    with patch("cinerec.api.main.get_redis_client", return_value=mock_redis), \
         patch("cinerec.api.main.TwoStageRecommender", return_value=mock_recommender):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session) -> User:
    """Создает тестового пользователя в БД."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("password123"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_movie(db_session):
    """Создает тестовый фильм в БД."""
    from cinerec.db.models import Movie
    movie = Movie(
        id=1,
        title="Test Movie",
        genres="Action|Drama",
        release_year=2020,
    )
    db_session.add(movie)
    db_session.commit()
    db_session.refresh(movie)
    return movie


@pytest.fixture
def auth_headers(test_user) -> dict:
    """JWT-токен для тестового пользователя."""
    token = create_access_token(subject=test_user.id)
    return {"Authorization": f"Bearer {token}"}