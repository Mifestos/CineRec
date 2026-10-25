from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cinerec.api import schemas
from cinerec.db import crud
from cinerec.db.session import get_db

router = APIRouter(prefix="/movies", tags=["Movies"])

@router.get("/{movie_id}", response_model=schemas.MovieResponse)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = crud.get_movie_by_id(db, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie