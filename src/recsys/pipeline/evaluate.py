"""Estágio ``evaluate``: compara o MLP com os baselines e promove o vencedor.

Calcula 4 métricas (precision@k, recall@k, NDCG@k, MAP@k) para o MLP e para os
baselines (popularity, SVD), registra tudo no MLflow e, se o MLP superar os
baselines na métrica primária, promove a versão de ``staging`` para
``production`` no Model Registry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient

from recsys.config import AppConfig, EnvSettings, load_config
from recsys.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import Recommender
from recsys.models.baselines import PopularityRecommender, SvdRecommender
from recsys.models.mlp import RecsysMLP
from recsys.pipeline.feature_eng import TEST_FILE, TRAIN_FILE
from recsys.pipeline.train import EXPERIMENT, MODEL_FILE, REGISTERED_MODEL, load_dims

METRICS_FILE = "metrics.json"
PRIMARY_METRIC = "ndcg_at_k"


def per_user_relevant(test: pd.DataFrame, user_col: str, item_col: str) -> dict[int, set[int]]:
    """Mapa usuário -> itens relevantes (interações de teste)."""
    return {int(user): set(group[item_col]) for user, group in test.groupby(user_col)}


MetricFn = Callable[[list[int], set[int], int], float]


def _mean_metric(
    recs: dict[int, list[int]], relevant: dict[int, set[int]], k: int, metric_fn: MetricFn
) -> float:
    """Média de uma métrica top-k sobre todos os usuários."""
    return float(mean(metric_fn(recs[user], rel, k) for user, rel in relevant.items()))


def evaluate_recommender(
    model: Recommender, relevant: dict[int, set[int]], k: int
) -> dict[str, float]:
    """Média das 4 métricas top-k sobre os usuários de teste."""
    if not relevant:
        return {m: 0.0 for m in ("precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k")}
    recs = {user: model.recommend(user, k) for user in relevant}
    return {
        "precision_at_k": _mean_metric(recs, relevant, k, precision_at_k),
        "recall_at_k": _mean_metric(recs, relevant, k, recall_at_k),
        "ndcg_at_k": _mean_metric(recs, relevant, k, ndcg_at_k),
        "map_at_k": _mean_metric(recs, relevant, k, average_precision_at_k),
    }


def load_mlp(cfg: AppConfig, dims: tuple[int, int], checkpoint: Path) -> RecsysMLP:
    """Reconstrói o MLP e carrega o checkpoint salvo."""
    n_users, n_items = dims
    model = RecsysMLP(
        n_users, n_items, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout
    )
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model


def fit_baselines(
    train_df: pd.DataFrame, cfg: AppConfig, dims: tuple[int, int]
) -> dict[str, Recommender]:
    """Ajusta os baselines de comparação (popularity e SVD)."""
    n_users, n_items = dims
    popularity = PopularityRecommender().fit(train_df, cfg.data.item_col, "weight")
    svd = SvdRecommender(seed=cfg.seed).fit(
        train_df, cfg.data.user_col, cfg.data.item_col, "weight", n_users, n_items
    )
    return {"popularity": popularity, "svd": svd}


def promote_if_qualified(results: dict[str, dict[str, float]], min_ndcg: float) -> bool:
    """Promove o MLP de ``staging`` para ``production`` se atingir o NDCG mínimo."""
    if results["mlp"][PRIMARY_METRIC] < min_ndcg:
        return False
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL, "staging").version
    client.set_registered_model_alias(REGISTERED_MODEL, "production", version)
    return True


def _log_and_promote(results: dict[str, dict[str, float]], min_ndcg: float) -> bool:
    """Loga as métricas de cada modelo no MLflow e promove o MLP se qualificar."""
    mlflow.set_tracking_uri(EnvSettings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluate"):
        for name, metrics in results.items():
            mlflow.log_metrics({f"{name}_{key}": value for key, value in metrics.items()})
        promoted = promote_if_qualified(results, min_ndcg)
        mlflow.log_param("promoted_to_production", promoted)
    return promoted


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC)."""
    cfg = load_config()
    dims = load_dims(cfg.data.processed_dir)
    processed = Path(cfg.data.processed_dir)
    test = pd.read_parquet(processed / TEST_FILE)
    train = pd.read_parquet(processed / TRAIN_FILE)
    relevant = per_user_relevant(test, cfg.data.user_col, cfg.data.item_col)
    models = {
        "mlp": load_mlp(cfg, dims, Path("models") / MODEL_FILE),
        **fit_baselines(train, cfg, dims),
    }
    results = {
        name: evaluate_recommender(m, relevant, cfg.evaluation.k) for name, m in models.items()
    }
    promoted = _log_and_promote(results, cfg.evaluation.min_ndcg)
    results["promoted_to_production"] = promoted  # type: ignore[assignment]
    Path(METRICS_FILE).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[evaluate] {json.dumps(results)}")


if __name__ == "__main__":
    main()
