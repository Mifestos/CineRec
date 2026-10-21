from pydantic import BaseModel, Field

class MovieRecommendation(BaseModel):
    movie_id: int = Field(..., alias="movieId")
    title: str
    genres: str
    score: float = Field(..., description="Оценка релевантности от ALS-модели")

    class Config:
        populate_by_name = True