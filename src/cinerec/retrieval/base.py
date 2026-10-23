from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    @abstractmethod
    def recommend(self, user_id: int, top_k: int, filter_already_liked: bool = True) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def has_user(self, user_id: int) -> bool:
        ...