from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ==========================================
# Схемы для Аутентификации и Пользователей
# ==========================================
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Уникальное имя пользователя")
    email: EmailStr = Field(..., description="Валидный email")
    password: str = Field(..., min_length=8, description="Пароль (минимум 8 символов)")

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: int | None = None


# ==========================================
# Схемы для Оценок (Ratings)
# ==========================================
class RatingCreate(BaseModel):
    movie_id: int = Field(..., description="ID фильма")
    score: float = Field(..., ge=1.0, le=5.0, description="Оценка от 1.0 до 5.0")

class RatingResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    score: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Схемы для Отзывов (Reviews) — v2.0
# ==========================================
class ReviewCreate(BaseModel):
    movie_id: int = Field(..., description="ID фильма")
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=2000, 
        description="Текст отзыва (от 10 до 2000 символов)"
    )

class ReviewResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    text: str
    
    # Результаты NLP-анализа
    sentiment: str | None = Field(None, description="Общая тональность (positive/negative/neutral)")
    sentiment_score: float | None = Field(None, description="Уверенность модели (0.0 - 1.0)")
    aspects: dict[str, Any] | None = Field(None, description="Тональность по аспектам (plot, actors, music и т.д.)")
    
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Схемы для Рекомендаций (ML)
# ==========================================
class MovieRecommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True) 
    
    movie_id: int = Field(..., alias="movieId", description="ID фильма")
    title: str = Field(..., description="Название фильма")
    genres: str = Field(..., description="Жанры фильма")
    score: float = Field(..., description="Оценка релевантности от ML-модели")

# ==========================================
# Карточка фильма
# ==========================================
class MovieResponse(BaseModel):
    id: int
    title: str
    genres: str
    release_year: int | None = None

    model_config = ConfigDict(from_attributes=True)