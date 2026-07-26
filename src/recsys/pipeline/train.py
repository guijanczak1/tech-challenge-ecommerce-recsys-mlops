"""Estágio ``train``: treina o baseline e registra o run no MLflow.

O modelo neural (PyTorch) entra na Etapa 4; aqui treina-se o baseline de
popularidade para servir de referência e exercitar o tracking.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pandas as pd

from recsys.config import AppConfig, EnvSettings, load_config
from recsys.models import baselines  # noqa: F401  (registra "popularity" na Factory)
from recsys.models.factory import create_model
from recsys.pipeline.feature_eng import TRAIN_FILE

MODEL_FILE = "baseline.joblib"
EXPERIMENT = "recsys-baseline"


def fit_baseline(train_df: pd.DataFrame, cfg: AppConfig) -> object:
    """Cria o recomendador via Factory e o ajusta ao conjunto de treino."""
    return create_model("popularity").fit(train_df, cfg.data.item_col, "weight")


def train_model(cfg: AppConfig, models_dir: Path = Path("models")) -> tuple[object, Path]:
    """Treina o baseline a partir de ``train.parquet`` e salva o artefato."""
    train_df = pd.read_parquet(Path(cfg.data.processed_dir) / TRAIN_FILE)
    model = fit_baseline(train_df, cfg)
    out = models_dir / MODEL_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return model, out


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC): treina e loga no MLflow."""
    cfg = load_config()
    mlflow.set_tracking_uri(EnvSettings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train-popularity"):
        mlflow.log_params(
            {
                "model": "popularity",
                "min_interactions": cfg.data.min_interactions,
                "k": cfg.evaluation.k,
            }
        )
        model, out = train_model(cfg)
        mlflow.log_metric("catalog_size", len(model.ranked_items))
        mlflow.log_artifact(str(out))
    print(f"[train] modelo salvo em {out}")


if __name__ == "__main__":
    main()
