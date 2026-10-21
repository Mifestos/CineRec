import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from cinerec.ranking.classifier_ranker import LGBMClassifierRanker
from cinerec.ranking.ltr_ranker import LGBMRankerRanker

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def train_models() -> None:
    logger.info("Загрузка признаков...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")
    
    features = [c for c in df.columns if c not in ["userId", "movieId", "rating", "target_binary", "target_relevance"]]
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. ОБУЧЕНИЕ LTR (Исследуем проблему)
    # ==========================================
    logger.info("Подготовка данных для LGBMRanker (LTR)...")
    
    unique_users = df["userId"].unique()
    sampled_users = pd.Series(unique_users).sample(n=5000, random_state=42).tolist()
    
    df_ltr = df[df["userId"].isin(sampled_users)].copy()
    df_ltr = df_ltr.sort_values(by="userId").reset_index(drop=True)
    
    # Делим пользователей на Train/Val (80/20), чтобы не рвать группы
    train_users = sampled_users[:4000]
    val_users = sampled_users[4000:]
    
    train_df = df_ltr[df_ltr["userId"].isin(train_users)]
    val_df = df_ltr[df_ltr["userId"].isin(val_users)]
    
    X_train = train_df[features]
    y_train = train_df["target_relevance"]
    train_groups = train_df.groupby("userId").size().to_numpy()
    
    X_val = val_df[features]
    y_val = val_df["target_relevance"]
    val_groups = val_df.groupby("userId").size().to_numpy()
    
    logger.info("Обучение LGBMRanker с валидацией...")
    ltr = LGBMRankerRanker(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        min_child_samples=10, 
        verbose=-1, 
        random_state=42
    )
    
    # Обучаем с eval_set, чтобы видеть метрику
    ltr.model.fit(
        X_train, y_train, 
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        eval_at=[10],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=10)]
    )
    
    # ВЫВОДИМ ВАЖНОСТЬ ПРИЗНАКОВ (чтобы понять, кто доминирует)
    importances = pd.Series(ltr.model.feature_importances_, index=features).sort_values(ascending=False)
    logger.info("\n--- ТОП-5 ВАЖНЫХ ПРИЗНАКОВ ДЛЯ LTR ---\n%s", importances.head(5).to_string())

    # ==========================================
    # 2. ОБУЧЕНИЕ КЛАССИФИКАТОРА
    # ==========================================
    logger.info("\nОбучение LGBMClassifier (Baseline)...")
    df_clf = df.sample(n=3_000_000, random_state=42)
    X_clf = df_clf[features]
    y_clf = df_clf["target_binary"]
    
    clf = LGBMClassifierRanker(
        n_estimators=100, learning_rate=0.1, max_depth=5, 
        min_child_samples=10, verbose=-1, random_state=42
    )
    clf.fit(X_clf, y_clf)
    
    # ВЫВОДИМ ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ КЛАССИФИКАТОРА
    clf_importances = pd.Series(clf.model.feature_importances_, index=features).sort_values(ascending=False)
    logger.info("\n--- ТОП-5 ВАЖНЫХ ПРИЗНАКОВ ДЛЯ CLASSIFIER ---\n%s", clf_importances.head(5).to_string())

    clf.save(MODEL_DIR / "ranker_classifier.pkl")
    ltr.save(MODEL_DIR / "ranker_ltr.pkl")
    logger.info("\nОбе модели успешно сохранены.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_models()