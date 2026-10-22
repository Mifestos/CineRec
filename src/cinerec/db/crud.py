from typing import List
from sqlalchemy.orm import Session

from cinerec.db.models import User, Rating, Movie, Review
from cinerec.core.security import get_password_hash
from cinerec.nlp.sentiment import analyze_sentiment, analyze_aspects


# ==========================================
# Операции с пользователями (Users)
# ==========================================
def get_user_by_username(db: Session, username: str) -> User | None:
    """Возвращает пользователя по имени."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    """Возвращает пользователя по email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Возвращает пользователя по ID."""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Создает нового пользователя с хешированным паролем."""
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# ==========================================
# Операции с фильмами (Movies)
# ==========================================
def get_movie_by_id(db: Session, movie_id: int) -> Movie | None:
    """Возвращает фильм по ID (для валидации при выставлении оценки)."""
    return db.query(Movie).filter(Movie.id == movie_id).first()


# ==========================================
# Операции с оценками (Ratings)
# ==========================================
def get_user_ratings(db: Session, user_id: int) -> dict[int, float]:
    """Возвращает словарь с оценками пользователя в формате {movie_id: score}."""
    ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    return {rating.movie_id: rating.score for rating in ratings}


def create_or_update_rating(db: Session, user_id: int, movie_id: int, score: float) -> Rating:
    """Создает новую оценку или обновляет существующую (Upsert-логика)."""
    existing_rating = db.query(Rating).filter(
        Rating.user_id == user_id,
        Rating.movie_id == movie_id
    ).first()

    if existing_rating:
        existing_rating.score = score
        db.commit()
        db.refresh(existing_rating)
        return existing_rating

    new_rating = Rating(user_id=user_id, movie_id=movie_id, score=score)
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    return new_rating


# ==========================================
# Операции с отзывами (Reviews)
# ==========================================
def create_review(db: Session, user_id: int, movie_id: int, text: str) -> Review:
    """
    Создает отзыв, прогоняет текст через NLP-модели 
    и сохраняет результаты анализа в БД.
    """
    # 1. Вызываем NLP-модуль
    sentiment, score = analyze_sentiment(text)
    aspects = analyze_aspects(text)
    
    # 2. Создаем запись в БД
    db_review = Review(
        user_id=user_id,
        movie_id=movie_id,
        text=text,
        sentiment=sentiment,
        sentiment_score=score,
        aspects=aspects
    )
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review


def get_reviews_by_movie(db: Session, movie_id: int, skip: int = 0, limit: int = 50) -> List[Review]:
    """Возвращает список отзывов для фильма с пагинацией."""
    return (
        db.query(Review)
        .filter(Review.movie_id == movie_id)
        .order_by(Review.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )