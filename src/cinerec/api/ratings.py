import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from cinerec.api import schemas
from cinerec.api.dependencies import get_current_user
from cinerec.db.session import get_db
from cinerec.db import crud
from cinerec.db.models import User

router = APIRouter(prefix="/ratings", tags=["Ratings"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=schemas.RatingResponse, status_code=status.HTTP_201_CREATED)
def rate_movie(
    rating_in: schemas.RatingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Выставляет или обновляет оценку фильма от имени текущего пользователя.
    """
    logger.info(f"User {current_user.username} rating movie {rating_in.movie_id} with score {rating_in.score}")
    
    # Проверяем, существует ли фильм (защита от Foreign Key Error)
    movie = crud.get_movie_by_id(db, movie_id=rating_in.movie_id)
    if not movie:
        logger.warning(f"Movie {rating_in.movie_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Movie with id {rating_in.movie_id} not found"
        )

    # Сохраняем оценку в БД
    rating = crud.create_or_update_rating(
        db, 
        user_id=current_user.id, 
        movie_id=rating_in.movie_id, 
        score=rating_in.score
    )
    
    return rating