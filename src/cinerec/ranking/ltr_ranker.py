import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from cinerec.ranking.base import BaseRanker


class LGBMRankerRanker(BaseRanker):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", **kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray, group: np.ndarray = None) -> None:
        if group is None:
            raise ValueError("LGBMRanker requires 'group' sizes for ranking.")
        self.model.fit(X, y, group=group)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)