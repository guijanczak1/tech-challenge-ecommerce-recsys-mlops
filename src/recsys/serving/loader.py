"""Carregamento local do modelo (imagem de serving self-contained).

Reconstrói o ``RecsysMLP`` a partir do checkpoint (``models/mlp.pt``) e das
dimensões (``data/processed/dims.json``), sem depender de um servidor MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from recsys.config import load_config
from recsys.models.mlp import RecsysMLP

DEFAULT_MODEL_PATH = Path("models/mlp.pt")
DEFAULT_DIMS_PATH = Path("data/processed/dims.json")


def load_local_model(
    model_path: Path = DEFAULT_MODEL_PATH, dims_path: Path = DEFAULT_DIMS_PATH
) -> RecsysMLP:
    """Carrega o MLP do checkpoint local e o coloca em modo de avaliação."""
    cfg = load_config()
    dims = json.loads(dims_path.read_text(encoding="utf-8"))
    model = RecsysMLP(
        dims["n_users"],
        dims["n_items"],
        cfg.model.embedding_dim,
        cfg.model.hidden_dims,
        cfg.model.dropout,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model
