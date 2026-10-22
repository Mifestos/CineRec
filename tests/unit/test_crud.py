from unittest.mock import patch
import pytest

from cinerec.db import crud
from cinerec.db.models import User, Rating, Review


def test_create_user(db_session):
    """Тест: создание нового пользователя."""
    user = crud.create_user(
        db=db_session, 
        username="newuser", 
        email="new@example.com", 
        password="securepassword123"
    )
    
    assert user.id is not None
    assert user.username == "newuser"
    assert user.email == "new@example.com"
    assert user.hashed_password != "securepassword123"


def test_upsert_rating_create_and_update(db_session, test_user, test_movie):
    """Тест: создание оценки, а затем её обновление (Upsert-логика)."""
    # 1. Создаем оценку (используем ID фильма из фикстуры)
    rating = crud.upsert_rating(
        db=db_session, 
        user_id=test_user.id, 
        movie_id=test_movie.id, 
        score=4.0
    )
    
    assert rating.id is not None
    assert rating.score == 4.0
    
    # 2. Обновляем ту же оценку
    updated_rating = crud.upsert_rating(
        db=db_session, 
        user_id=test_user.id, 
        movie_id=test_movie.id, 
        score=5.0
    )
    
    # Проверяем, что запись обновилась, а не создалась новая
    assert updated_rating.id == rating.id
    assert updated_rating.score == 5.0
    
    count = db_session.query(Rating).filter(Rating.user_id == test_user.id).count()
    assert count == 1


@patch("cinerec.db.crud.analyze_sentiment")
@patch("cinerec.db.crud.analyze_aspects")
def test_upsert_review(
    mock_analyze_aspects, 
    mock_analyze_sentiment, 
    db_session, 
    test_user, 
    test_movie
):
    """
    Тест: создание отзыва с NLP-анализом.
    Используем моки, чтобы не загружать RuBERT в память.
    """
    # Настраиваем моки
    mock_analyze_sentiment.return_value = ("positive", 0.95)
    mock_analyze_aspects.return_value = {"plot": "positive", "actors": "negative"}
    
    # 1. Создаем отзыв
    review = crud.upsert_review(
        db=db_session,
        user_id=test_user.id,
        movie_id=test_movie.id,
        text="Отличный сюжет, но актеры слабые."
    )
    
    assert review.id is not None
    assert review.sentiment == "positive"
    assert review.sentiment_score == 0.95
    assert review.aspects["plot"] == "positive"
    assert review.aspects["actors"] == "negative"
    
    # 2. Обновляем отзыв (пользователь передумал)
    mock_analyze_sentiment.return_value = ("negative", 0.80)
    mock_analyze_aspects.return_value = {"plot": "negative"}
    
    updated_review = crud.upsert_review(
        db=db_session,
        user_id=test_user.id,
        movie_id=test_movie.id,
        text="На самом деле сюжет ужасен."
    )
    
    assert updated_review.id == review.id  
    assert updated_review.sentiment == "negative"
    assert updated_review.sentiment_score == 0.80
    assert updated_review.aspects == {"plot": "negative"}