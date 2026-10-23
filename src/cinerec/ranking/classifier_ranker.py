import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from cinerec.ranking.base import BaseRanker


class LGBMClassifierRanker(BaseRanker):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray, group: np.ndarray = None) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)