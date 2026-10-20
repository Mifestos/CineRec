import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def train_ranker() -> None:
    """Обучает LGBMClassifier для ранжирования кандидатов."""
    logger.info("Загрузка признаков...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")

    # Исключаем ID и исходный рейтинг, оставляем только фичи и таргет
    features = [c for c in df.columns if c not in ["userId", "movieId", "rating", "target"]]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Обучение LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "ranker.pkl", "wb") as f:
        pickle.dump(model, f)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info(f"Модель обучена. ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_ranker()