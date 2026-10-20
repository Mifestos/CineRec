import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, ConfigDict

from cinerec.pipeline import TwoStageRecommender

# Настройка лаконичного логирования (в проде настраивается через dictConfig/YAML)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MovieRecommendation(BaseModel):
    """Схема ответа API для рекомендованного фильма."""
    model_config = ConfigDict(populate_by_name=True)
    
    movie_id: int = Field(..., alias="movieId")
    title: str
    genres: str
    score: float = Field(..., description="Оценка релевантности")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация двухэтапной рекомендательной системы...")
    # Заменяем ALSRecommender на TwoStageRecommender
    app.state.recommender = TwoStageRecommender()
    logger.info("Модели успешно загружены в память.")
    yield
    logger.info("Завершение работы приложения.")


# Инициализация приложения
app = FastAPI(
    title="CineRec API",
    description="Двухэтапная рекомендательная система",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/api/v1/recommend/{user_id}", response_model=List[MovieRecommendation])
async def get_recommendations(
    request: Request,
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Количество рекомендаций")
):
    """
    Возвращает топ-N рекомендаций для заданного пользователя.
    """
    # Достаем модель из состояния приложения
    recommender = request.app.state.recommender
    
    try:
        # recommend() уже возвращает готовый список словарей!
        recommendations = recommender.recommend(user_id, top_n=top_n)
        return recommendations
        
    except ValueError as e:
        logger.warning(f"Пользователь {user_id} не найден: {e}")
        raise HTTPException(status_code=404, detail="Пользователь не найден")