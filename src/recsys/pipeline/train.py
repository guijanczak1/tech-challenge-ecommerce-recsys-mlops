"""Estágio ``train``: treina o MLP, salva o checkpoint e registra no MLflow.

Loga params/métricas, salva o artefato do modelo e registra uma versão no
Model Registry com o alias ``staging`` (a promoção a ``production`` acontece no
estágio de avaliação, se o MLP superar os baselines).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader, random_split

from recsys.config import AppConfig, EnvSettings, load_config
from recsys.data.dataset import InteractionsDataset
from recsys.models import mlp  # noqa: F401  (registra "mlp" na Factory)
from recsys.models.factory import create_model
from recsys.pipeline.feature_eng import DIMS_FILE, TRAIN_FILE
from recsys.training.recsys_trainer import RecsysTrainer
from recsys.training.trainer import TrainingHistory
from recsys.utils.seed import set_seed

MODEL_FILE = "mlp.pt"
EXPERIMENT = "recsys"
REGISTERED_MODEL = "recsys-mlp"


def load_dims(processed_dir: str) -> tuple[int, int]:
    """Lê ``dims.json`` e devolve ``(n_users, n_items)``."""
    dims = json.loads((Path(processed_dir) / DIMS_FILE).read_text(encoding="utf-8"))
    return dims["n_users"], dims["n_items"]


def build_loaders(
    train_df: pd.DataFrame, cfg: AppConfig, n_items: int
) -> tuple[DataLoader, DataLoader]:
    """Cria DataLoaders de treino e validação (split 90/10 para early stopping)."""
    dataset = InteractionsDataset(
        train_df[cfg.data.user_col].to_numpy(),
        train_df[cfg.data.item_col].to_numpy(),
        n_items,
        cfg.training.negatives_per_positive,
        cfg.seed,
    )
    n_val = max(1, int(len(dataset) * 0.1))
    lengths = [len(dataset) - n_val, n_val]
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, lengths, generator=generator)
    batch = cfg.training.batch_size
    return DataLoader(train_ds, batch, shuffle=True), DataLoader(val_ds, batch)


def train_model(cfg: AppConfig, dims: tuple[int, int], checkpoint: Path) -> TrainingHistory:
    """Treina o MLP e recarrega o melhor checkpoint no modelo."""
    set_seed(cfg.seed)
    n_users, n_items = dims
    train_df = pd.read_parquet(Path(cfg.data.processed_dir) / TRAIN_FILE)
    model = create_model(
        "mlp",
        n_users=n_users,
        n_items=n_items,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    )
    train_loader, val_loader = build_loaders(train_df, cfg, n_items)
    history = RecsysTrainer(model, train_loader, val_loader, cfg, checkpoint).fit(
        cfg.training.epochs
    )
    model.load_state_dict(torch.load(checkpoint))
    return _register(model, history)


def _register(model: torch.nn.Module, history: TrainingHistory) -> TrainingHistory:
    """Loga métricas e registra o modelo no Registry com alias ``staging``."""
    mlflow.log_metric("best_val_loss", history.best_val_loss)
    mlflow.log_metric("best_epoch", history.best_epoch)
    model.eval()
    example = (torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    info = mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle",
        input_example=example,
        registered_model_name=REGISTERED_MODEL,
    )
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL, "staging", info.registered_model_version
    )
    return history


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC)."""
    cfg = load_config()
    dims = load_dims(cfg.data.processed_dir)
    mlflow.set_tracking_uri(EnvSettings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train-mlp"):
        mlflow.log_params(
            {"model": "mlp", "epochs": cfg.training.epochs, "lr": cfg.training.learning_rate}
        )
        train_model(cfg, dims, Path("models") / MODEL_FILE)
    print(f"[train] MLP treinado e registrado como '{REGISTERED_MODEL}@staging'")


if __name__ == "__main__":
    main()
