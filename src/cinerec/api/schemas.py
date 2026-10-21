from pydantic import BaseModel, Field, ConfigDict

class MovieRecommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True) 
    
    movie_id: int = Field(..., alias="movieId")
    title: str
    genres: str
    score: float = Field(..., description="Оценка релевантности от ALS-модели")