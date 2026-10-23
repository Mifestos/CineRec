from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """Базовый интерфейс для алгоритмов ранжирования (Stage 2)."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, group: np.ndarray | None = None) -> None:
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