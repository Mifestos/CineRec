import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, ConfigDict

from cinerec.retrieval.recommender import ALSRecommender

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
    """
    Управление жизненным циклом приложения (Lifespan).
    Загружаем модель один раз при старте и храним в app.state.
    """
    logger.info("Инициализация ALS-модели...")
    # Привязываем модель к состоянию приложения (Dependency Injection)
    app.state.recommender = ALSRecommender()
    logger.info("Модель успешно загружена.")
    
    yield # Сервер работает
    
    logger.info("Завершение работы сервера, очистка ресурсов...")


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
        recs_df = recommender.recommend(user_id, top_n=top_n)
        return recs_df.reset_index().to_dict(orient="records")
        
    except ValueError as e:
        logger.warning(f"Пользователь {user_id} не найден: {e}")
        raise HTTPException(status_code=404, detail="Пользователь не найден")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации рекомендаций для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")