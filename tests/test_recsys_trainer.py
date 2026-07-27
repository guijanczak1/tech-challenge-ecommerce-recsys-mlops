"""Teste do trainer neural (treino curto em CPU com dados sintéticos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from recsys.config import EnvSettings, load_config
from recsys.data.dataset import InteractionsDataset
from recsys.models.mlp import RecsysMLP
from recsys.training.recsys_trainer import RecsysTrainer

PARAMS = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"


def _loader(seed: int) -> DataLoader:
    rng = np.random.default_rng(seed)
    users = rng.integers(0, 6, size=40)
    items = rng.integers(0, 8, size=40)
    ds = InteractionsDataset(users, items, n_items=8, negatives=2, seed=seed)
    return DataLoader(ds, batch_size=16, shuffle=True)


def test_trainer_runs_and_checkpoints(tmp_path: Path) -> None:
    cfg = load_config(EnvSettings(params_file=PARAMS))
    model = RecsysMLP(n_users=6, n_items=8, embedding_dim=4, hidden_dims=[8], dropout=0.0)
    ckpt = tmp_path / "model.pt"
    trainer = RecsysTrainer(model, _loader(1), _loader(2), cfg, ckpt)

    history = trainer.fit(epochs=3)

    assert len(history.train_losses) >= 1
    assert len(history.val_losses) == len(history.train_losses)
    assert ckpt.exists()  # melhor checkpoint salvo
    assert history.best_val_loss < float("inf")
