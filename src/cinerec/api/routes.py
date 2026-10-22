import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from sqlalchemy.orm import Session

from cinerec.api.schemas import MovieRecommendation
from cinerec.api.dependencies import get_current_user
from cinerec.db.session import get_db
from cinerec.db import crud
from cinerec.db.models import User

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/recommend", response_model=List[MovieRecommendation])
async def get_personal_recommendations(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    top_n: int = Query(10, ge=1, le=50)
):
    """
    Возвращает топ-N персональных рекомендаций для авторизованного пользователя.
    """
    user_id = current_user.id
    # Ключ кэша зависит от ID пользователя и количества запрошенных фильмов
    cache_key = f"recs:user_{user_id}:top_{top_n}"

    # 1. Проверка кэша Redis
    if cached := await request.app.state.redis.get(cache_key):
        logger.debug("Cache hit: user_id=%s", user_id)
        return json.loads(cached)

    logger.debug("Cache miss: user_id=%s", user_id)

    try:
        # 2. Достаем актуальную историю оценок пользователя из PostgreSQL
        user_ratings = crud.get_user_ratings(db, user_id=user_id)

        # 3. Передаем оценки в ML-пайплайн
        # Внутри pipeline.py сработает логика: если оценок < 5, вернется Fallback (популярное)
        recommendations = request.app.state.recommender.recommend(
            user_ratings=user_ratings, 
            top_n=top_n
        )

        # 4. Сохраняем в кэш на 5 минут (300 секунд). 
        # Для активных пользователей кэш будет обновляться при добавлении новых оценок.
        await request.app.state.redis.setex(cache_key, 300, json.dumps(recommendations))

        return recommendations

    except Exception as e:
        logger.error("Recommendation failed for user_id=%s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")