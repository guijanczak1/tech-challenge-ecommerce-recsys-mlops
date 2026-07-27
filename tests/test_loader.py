"""Teste do carregamento local do modelo (imagem self-contained)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from recsys.config import load_config
from recsys.models.mlp import RecsysMLP
from recsys.serving.loader import load_local_model


def test_load_local_model_roundtrip(tmp_path: Path) -> None:
    cfg = load_config()  # usa configs/params.yaml do repo
    model = RecsysMLP(5, 6, cfg.model.embedding_dim, cfg.model.hidden_dims, cfg.model.dropout)
    model_path = tmp_path / "mlp.pt"
    dims_path = tmp_path / "dims.json"
    torch.save(model.state_dict(), model_path)
    dims_path.write_text(json.dumps({"n_users": 5, "n_items": 6}), encoding="utf-8")

    loaded = load_local_model(model_path, dims_path)

    assert isinstance(loaded, RecsysMLP)
    assert len(loaded.recommend(0, 3)) == 3
