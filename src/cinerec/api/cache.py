import os
import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_redis_client() -> redis.Redis:
    """Возвращает асинхронный клиент Redis."""
    return redis.from_url(REDIS_URL, decode_responses=True)