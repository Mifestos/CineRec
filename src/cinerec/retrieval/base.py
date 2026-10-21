from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Базовый интерфейс для алгоритмов генерации кандидатов (Stage 1)."""

    @abstractmethod
    def recommend(self, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """
        Возвращает топ-K рекомендованных фильмов.
        Формат: [{'movieId': int, 'score': float}, ...]
        """
        ...

    @abstractmethod
    def has_user(self, user_id: int) -> bool:
        """Проверяет, существует ли пользователь в базе (для обработки Cold Start)."""
        ...