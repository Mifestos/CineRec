import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pickle
from pathlib import Path
import pandas as pd
from implicit.als import AlternatingLeastSquares  
MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")

class ALSRecommender:
    def __init__(self) -> None:
        # Создаём экземпляр модели (фабрика выберет CPU) и загружаем веса
        self.model = AlternatingLeastSquares().load(MODEL_DIR / "als_model.npz")
        
        # Восстанавливаем маппинги ID
        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            self.user_to_idx = data["user_to_idx"]
            self.idx_to_movie = data["idx_to_movie"]
            self.sparse_user_item = data["sparse_user_item"]
            
        # Загружаем метаданные фильмов
        self.movies = pd.read_csv(DATA_DIR / "movies.csv").set_index("movieId")

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Генерирует топ-N рекомендаций для заданного пользователя."""
        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found in training data.")
            
        user_idx = self.user_to_idx[user_id]
        
        # Получаем рекомендации (filter_already_liked_items исключает уже просмотренное)
        item_ids, scores = self.model.recommend(
            user_idx, 
            self.sparse_user_item[user_idx], 
            N=top_n, 
            filter_already_liked_items=True
        )
        
        # Переводим внутренние индексы обратно в movieId
        movie_ids = [self.idx_to_movie[idx] for idx in item_ids]
        
        # Возвращаем таблицу с названиями, жанрами и скорами релевантности
        return self.movies.loc[movie_ids][["title", "genres"]].assign(score=scores)

if __name__ == "__main__":
    rec = ALSRecommender()
    print(rec.recommend(user_id=1, top_n=5))