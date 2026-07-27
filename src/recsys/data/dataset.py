"""Dataset PyTorch de feedback implícito com amostragem de negativos.

Positivos são as interações observadas (rótulo 1). Negativos são itens
amostrados uniformemente (rótulo 0), na proporção ``negatives_per_positive``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def build_samples(
    users: np.ndarray, items: np.ndarray, n_items: int, negatives: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta arrays (usuário, item, rótulo) com positivos + negativos amostrados."""
    rng = np.random.default_rng(seed)
    neg_users = np.repeat(users, negatives)
    neg_items = rng.integers(0, n_items, size=neg_users.size)
    all_users = np.concatenate([users, neg_users])
    all_items = np.concatenate([items, neg_items])
    labels = np.concatenate([np.ones(users.size), np.zeros(neg_users.size)])
    return all_users.astype("int64"), all_items.astype("int64"), labels.astype("float32")


class InteractionsDataset(Dataset):
    """Pares (usuário, item, rótulo) para treino de feedback implícito."""

    def __init__(
        self, users: np.ndarray, items: np.ndarray, n_items: int, negatives: int, seed: int
    ) -> None:
        """Constrói as amostras positivas e negativas."""
        u, i, y = build_samples(users, items, n_items, negatives, seed)
        self._users = torch.from_numpy(u)
        self._items = torch.from_numpy(i)
        self._labels = torch.from_numpy(y)

    def __len__(self) -> int:
        """Número total de amostras (positivas + negativas)."""
        return self._labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna ``(usuário, item, rótulo)`` na posição ``index``."""
        return self._users[index], self._items[index], self._labels[index]
