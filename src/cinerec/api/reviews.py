import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from cinerec.api import schemas
from cinerec.api.dependencies import get_current_user
from cinerec.db.session import get_db
from cinerec.db import crud
from cinerec.db.models import User

router = APIRouter(prefix="/reviews", tags=["Reviews"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=schemas.ReviewResponse, status_code=status.HTTP_201_CREATED)
def create_review(
    review_in: schemas.ReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Создает или обновляет отзыв от имени текущего пользователя.
    Автоматически запускает NLP-анализ тональности и извлечение аспектов.
    """
    logger.info("User %s is submitting a review for movie %s", current_user.username, review_in.movie_id)
    
    # Проверяем, существует ли фильм в базе
    movie = crud.get_movie_by_id(db, movie_id=review_in.movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Movie with id {review_in.movie_id} not found"
        )

    # Создаем или обновляем отзыв (NLP-анализ происходит внутри crud.upsert_review)
    review = crud.upsert_review(
        db, 
        user_id=current_user.id, 
        movie_id=review_in.movie_id, 
        text=review_in.text
    )
    
    logger.info(
        "Review processed successfully. Sentiment: %s, Aspects found: %s", 
        review.sentiment, 
        list(review.aspects.keys()) if review.aspects else []
    )
    
    return review


@router.get("/{movie_id}", response_model=List[schemas.ReviewResponse])
def get_movie_reviews(
    movie_id: int,
    skip: int = Query(0, ge=0, description="Количество отзывов для пропуска (пагинация)"),
    limit: int = Query(50, ge=1, le=100, description="Максимальное количество отзывов"),
    db: Session = Depends(get_db)
):
    """
    Возвращает список отзывов для конкретного фильма.
    Доступно без авторизации.
    """
    movie = crud.get_movie_by_id(db, movie_id=movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Movie with id {movie_id} not found"
        )
        
    reviews = crud.get_reviews_by_movie(db, movie_id=movie_id, skip=skip, limit=limit)
    return reviews