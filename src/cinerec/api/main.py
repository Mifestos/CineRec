import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cinerec.api.cache import get_redis_client
from cinerec.api.routes import router
from cinerec.pipeline import TwoStageRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading ML models and Redis connection...")
    
    app.state.recommender = TwoStageRecommender()
    
    redis_client = get_redis_client()
    await redis_client.ping()  # Fail-fast, если Redis недоступен
    app.state.redis = redis_client
    
    logger.info("Application successfully started.")
    yield
    
    logger.info("Shutting down: closing Redis connection...")
    await app.state.redis.aclose()
    logger.info("Application stopped.")

app = FastAPI(
    title="CineRec API",
    description="Two-stage Recommender System",
    version="1.1.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1", tags=["recommendations"])