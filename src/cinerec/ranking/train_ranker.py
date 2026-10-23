import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from cinerec.ranking.classifier_ranker import LGBMClassifierRanker
from cinerec.ranking.ltr_ranker import LGBMRankerRanker

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Настраиваем MLflow на локальную папку (в проде тут был бы сервер)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("CineRec-Ranking")

def train_models() -> None:
    logger.info("Загрузка признаков...")
    df = pd.read_parquet(PROCESSED_DIR / "ranking_features.parquet")
    
    features = [c for c in df.columns if c not in ["userId", "movieId", "rating", "target_binary", "target_relevance"]]
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. ОБУЧЕНИЕ КЛАССИФИКАТОРА (Baseline)
    # ==========================================
    logger.info("Обучение LGBMClassifier (Baseline)...")
    df_clf = df.sample(n=3_000_000, random_state=42)
    X_clf = df_clf[features]
    y_clf = df_clf["target_binary"]
    
    clf_params = {
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, 
        "min_child_samples": 10, "random_state": 42
    }
    
    with mlflow.start_run(run_name="LGBMClassifier-Baseline"):
        mlflow.log_params(clf_params)
        
        clf = LGBMClassifierRanker(**clf_params, verbose=-1)
        clf.fit(X_clf, y_clf)
        
        # Логируем артефакт (веса модели)
        clf.save(MODEL_DIR / "ranker_classifier.pkl")
        mlflow.log_artifact(MODEL_DIR / "ranker_classifier.pkl")
        
        # В реальном коде здесь мы бы считали AUC на валидации и логировали его:
        # mlflow.log_metric("roc_auc", auc_score)
        logger.info("Классификатор сохранен и залогирован в MLflow.")

    # ==========================================
    # 2. ОБУЧЕНИЕ LTR (Challenger)
    # ==========================================
    logger.info("Подготовка данных для LGBMRanker (LTR)...")
    unique_users = df["userId"].unique()
    sampled_users = pd.Series(unique_users).sample(n=5000, random_state=42).tolist()
    
    df_ltr = df[df["userId"].isin(sampled_users)].copy()
    df_ltr = df_ltr.sort_values(by="userId").reset_index(drop=True)
    
    train_users = sampled_users[:4000]
    val_users = sampled_users[4000:]
    
    train_df = df_ltr[df_ltr["userId"].isin(train_users)]
    val_df = df_ltr[df_ltr["userId"].isin(val_users)]
    
    X_train, y_train = train_df[features], train_df["target_relevance"]
    X_val, y_val = val_df[features], val_df["target_relevance"]
    train_groups = train_df.groupby("userId").size().to_numpy()
    val_groups = val_df.groupby("userId").size().to_numpy()
    
    ltr_params = {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, 
        "min_child_samples": 10, "random_state": 42
    }
    
    logger.info("Обучение LGBMRanker (LTR) с логированием в MLflow...")
    with mlflow.start_run(run_name="LGBMRanker-LTR"):
        mlflow.log_params(ltr_params)
        
        ltr = LGBMRankerRanker(**ltr_params, verbose=-1)
        
        # Обучаем с eval_set, чтобы достать финальную метрику NDCG
        ltr.model.fit(
            X_train, y_train, 
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_at=[10],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Достаем лучший NDCG из истории обучения и логируем его
        best_ndcg = ltr.model.best_score_["valid_0"]["ndcg@10"]
        mlflow.log_metric("ndcg_at_10", best_ndcg)
        
        ltr.save(MODEL_DIR / "ranker_ltr.pkl")
        mlflow.log_artifact(MODEL_DIR / "ranker_ltr.pkl")
        
        logger.info("LTR-ранжировщик сохранен. NDCG@10: %.4f. Залогирован в MLflow.", best_ndcg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_models()