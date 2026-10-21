from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

class BaseRanker(ABC):
    """Базовый интерфейс для алгоритмов ранжирования (Stage 2)."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, group: Optional[np.ndarray] = None) -> None:
        """
        Обучение модели.
        Параметр `group` используется только в алгоритмах Learning-to-Rank (LTR).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает массив скоров для ранжирования кандидатов."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Сериализация весов модели на диск."""
        ...