import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from cinerec.api.cache import get_redis_client
from cinerec.api.auth import router as auth_router
from cinerec.api.ratings import router as ratings_router  # <-- Добавлен импорт
from cinerec.api.routes import router as recommend_router
from cinerec.pipeline import TwoStageRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения (загрузка ML-моделей и Redis)."""
    logger.info("Starting up: loading ML models and Redis connection...")
    
    # Загружаем ML-пайплайн в память
    app.state.recommender = TwoStageRecommender()
    
    # Подключаемся к Redis (Fail-fast, если недоступен)
    redis_client = get_redis_client()
    await redis_client.ping()
    app.state.redis = redis_client
    
    logger.info("Application successfully started.")
    yield
    
    logger.info("Shutting down: closing Redis connection...")
    await app.state.redis.aclose()
    logger.info("Application stopped.")


app = FastAPI(
    title="CineRec API",
    description="Full-stack Recommender System with UGC (v2.0)",
    version="2.0.0",
    lifespan=lifespan
)

# Подключаем роутеры
app.include_router(auth_router)
app.include_router(ratings_router)  # <-- Подключен роутер оценок
app.include_router(recommend_router, prefix="/api/v1", tags=["Recommendations"])

# Инструментатор для сбора метрик Prometheus
Instrumentator().instrument(app).expose(app)