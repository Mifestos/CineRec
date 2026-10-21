from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def recommend(self, user_id: int, top_k: int, filter_already_liked: bool = True) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def has_user(self, user_id: int) -> bool:
        ...