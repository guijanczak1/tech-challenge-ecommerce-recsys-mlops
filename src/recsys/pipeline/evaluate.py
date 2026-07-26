"""Estágio ``evaluate``: mede o modelo salvo e registra métricas no MLflow.

Gera ``metrics.json`` com precision@k e recall@k médios sobre os usuários do
conjunto de teste e loga as métricas num run do MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import joblib
import mlflow
import pandas as pd

from recsys.config import AppConfig, EnvSettings, load_config
from recsys.evaluation.metrics import precision_at_k, recall_at_k
from recsys.pipeline.feature_eng import TEST_FILE
from recsys.pipeline.train import EXPERIMENT, MODEL_FILE

METRICS_FILE = "metrics.json"


def per_user_relevant(test: pd.DataFrame, user_col: str, item_col: str) -> dict[int, set[int]]:
    """Mapa usuário -> conjunto de itens relevantes (interações de teste)."""
    return {int(user): set(group[item_col]) for user, group in test.groupby(user_col)}


def evaluate_model(model: object, test: pd.DataFrame, cfg: AppConfig) -> dict[str, float]:
    """Calcula precision@k e recall@k médios sobre os usuários de teste."""
    k = cfg.evaluation.k
    recommended = model.recommend(k)  # type: ignore[attr-defined]
    relevant = per_user_relevant(test, cfg.data.user_col, cfg.data.item_col)
    precisions = [precision_at_k(recommended, rel, k) for rel in relevant.values()]
    recalls = [recall_at_k(recommended, rel, k) for rel in relevant.values()]
    return {
        "precision_at_k": float(mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(mean(recalls)) if recalls else 0.0,
        "n_users": float(len(relevant)),
    }


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC): avalia e loga no MLflow."""
    cfg = load_config()
    model = joblib.load(Path("models") / MODEL_FILE)
    test = pd.read_parquet(Path(cfg.data.processed_dir) / TEST_FILE)
    metrics = evaluate_model(model, test, cfg)
    Path(METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mlflow.set_tracking_uri(EnvSettings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluate-popularity"):
        mlflow.log_metrics(metrics)
    print(f"[evaluate] {metrics}")


if __name__ == "__main__":
    main()
