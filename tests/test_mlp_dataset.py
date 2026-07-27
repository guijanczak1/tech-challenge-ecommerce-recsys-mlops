"""Testes do Dataset de interações e do modelo neural (MLP)."""

from __future__ import annotations

import numpy as np
import torch

from recsys.data.dataset import InteractionsDataset, build_samples
from recsys.models import mlp  # noqa: F401  (registra "mlp" na Factory)
from recsys.models.factory import create_model
from recsys.models.mlp import RecsysMLP


def test_build_samples_sizes_and_labels() -> None:
    users = np.array([0, 1, 2])
    items = np.array([5, 6, 7])
    u, i, y = build_samples(users, items, n_items=10, negatives=2, seed=42)
    assert len(u) == len(i) == len(y) == 3 + 3 * 2  # positivos + negativos
    assert set(np.unique(y)) == {0.0, 1.0}
    assert y[:3].tolist() == [1.0, 1.0, 1.0]  # positivos primeiro


def test_dataset_getitem_returns_tensors() -> None:
    ds = InteractionsDataset(np.array([0, 1]), np.array([2, 3]), n_items=5, negatives=1, seed=0)
    assert len(ds) == 4
    user, item, label = ds[0]
    assert user.dtype == torch.int64 and label.dtype == torch.float32


def _model() -> RecsysMLP:
    return RecsysMLP(n_users=6, n_items=8, embedding_dim=4, hidden_dims=[8, 4], dropout=0.0)


def test_forward_output_shape() -> None:
    model = _model()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    out = model(users, items)
    assert out.shape == (3,)


def test_recommend_returns_k_valid_items() -> None:
    recs = _model().recommend(user=0, k=3)
    assert len(recs) == 3
    assert all(0 <= item < 8 for item in recs)


def test_mlp_registered_in_factory() -> None:
    model = create_model("mlp", n_users=6, n_items=8, embedding_dim=4, hidden_dims=[8], dropout=0.0)
    assert isinstance(model, RecsysMLP)
