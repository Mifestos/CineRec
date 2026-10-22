from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from cinerec.db.models import User, Rating, Movie, Review
from cinerec.core.security import get_password_hash
from cinerec.nlp.sentiment import analyze_sentiment, analyze_aspects


# ==========================================
# Users
# ==========================================
def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Создает нового пользователя с хешированным паролем."""
    db_user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# ==========================================
# Movies
# ==========================================
def get_movie_by_id(db: Session, movie_id: int) -> Movie | None:
    return db.query(Movie).filter(Movie.id == movie_id).first()


# ==========================================
# Ratings
# ==========================================
def get_user_ratings(db: Session, user_id: int) -> dict[int, float]:
    """Возвращает словарь с оценками пользователя: {movie_id: score}."""
    rows = db.query(Rating.movie_id, Rating.score).filter(Rating.user_id == user_id).all()
    return {movie_id: score for movie_id, score in rows}


def upsert_rating(db: Session, user_id: int, movie_id: int, score: float) -> Rating:
    """
    Атомарный upsert через PostgreSQL ON CONFLICT.
    Гарантирует, что даже при параллельных запросах не будет дубликата.
    """
    stmt = (
        pg_insert(Rating)
        .values(user_id=user_id, movie_id=movie_id, score=score)
        .on_conflict_do_update(
            constraint="uq_user_movie_rating",
            set_={"score": score},
        )
        .returning(Rating)
    )
    result = db.execute(stmt)
    db.commit()
    return result.scalar_one()


# ==========================================
# Reviews
# ==========================================
def upsert_review(db: Session, user_id: int, movie_id: int, text: str) -> Review:
    """
    Создает или обновляет отзыв. При обновлении повторно запускает NLP-анализ
    (тональность и аспекты могут измениться, если пользователь отредактировал текст).
    """
    sentiment, score = analyze_sentiment(text)
    aspects = analyze_aspects(text)

    stmt = (
        pg_insert(Review)
        .values(
            user_id=user_id,
            movie_id=movie_id,
            text=text,
            sentiment=sentiment,
            sentiment_score=score,
            aspects=aspects,
        )
        .on_conflict_do_update(
            constraint="uq_user_movie_review",
            set_={
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": score,
                "aspects": aspects,
            },
        )
        .returning(Review)
    )
    result = db.execute(stmt)
    db.commit()
    return result.scalar_one()


def get_reviews_by_movie(db: Session, movie_id: int, skip: int = 0, limit: int = 50) -> List[Review]:
    """Возвращает список отзывов для фильма (от новых к старым) с пагинацией."""
    return (
        db.query(Review)
        .filter(Review.movie_id == movie_id)
        .order_by(Review.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )