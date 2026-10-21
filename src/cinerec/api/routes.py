import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query, Request

from cinerec.api.schemas import MovieRecommendation

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/recommend/{user_id}", response_model=List[MovieRecommendation])
async def get_recommendations(
    request: Request,
    user_id: int,
    top_n: int = Query(10, ge=1, le=50)
):
    """Возвращает топ-N рекомендаций с кэшированием."""
    cache_key = f"recs:{user_id}:{top_n}"
    
    # Проверка кэша (моржовый оператор для краткости)
    if cached := await request.app.state.redis.get(cache_key):
        logger.debug("Cache hit: user=%s, top_n=%s", user_id, top_n)
        return json.loads(cached)

    logger.debug("Cache miss: user=%s, top_n=%s", user_id, top_n)
    
    try:
        recommendations = request.app.state.recommender.recommend(user_id, top_n)
        # Кэшируем на час (3600 секунд)
        await request.app.state.redis.setex(cache_key, 3600, json.dumps(recommendations))
        return recommendations
        
    except ValueError as e:
        logger.warning("User not found: %s", e)
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    except Exception as e:
        logger.error("Recommendation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")