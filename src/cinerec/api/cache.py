import redis.asyncio as redis
from cinerec.core.config import settings


def get_redis_client() -> redis.Redis:
    """Возвращает асинхронный клиент Redis на основе глобальной конфигурации."""
    return redis.from_url(settings.REDIS_URL, decode_responses=True)