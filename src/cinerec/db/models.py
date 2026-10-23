from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    # Используем timezone-aware datetime
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tmdb_id: Mapped[int | None] = mapped_column(Integer, unique=True, index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    genres: Mapped[str] = mapped_column(String(255))
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    poster_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan"
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan"
    )


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"))
    score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    user: Mapped["User"] = relationship(back_populates="ratings")
    movie: Mapped["Movie"] = relationship(back_populates="ratings")

    # Защита на уровне БД: один пользователь может поставить только одну оценку фильму
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating"),
    )


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)

    # Результаты NLP-анализа
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # JSONB быстрее и поддерживает GIN-индексы
    aspects: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    user: Mapped["User"] = relationship(back_populates="reviews")
    movie: Mapped["Movie"] = relationship(back_populates="reviews")

    # GIN-индекс для быстрого поиска по содержимому JSONB (например, WHERE aspects->>'plot' = 'positive')
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_review"),
        Index("ix_reviews_aspects_gin", "aspects", postgresql_using="gin"),
    )