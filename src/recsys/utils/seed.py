"""Determinismo: fixa seeds de random/numpy/torch quando disponíveis."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Fixa a semente global para reprodutibilidade.

    Args:
        seed: Valor aplicado a ``random``, ``PYTHONHASHSEED`` e, quando
            instalados, a ``numpy`` e ``torch`` (CPU e GPU).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _seed_numpy(seed)
    _seed_torch(seed)


def _seed_numpy(seed: int) -> None:
    """Fixa a semente do numpy se estiver instalado."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover — numpy é dependência obrigatória do projeto
        return
    np.random.seed(seed)


def _seed_torch(seed: int) -> None:
    """Fixa a semente do torch (CPU/GPU) se estiver instalado."""
    try:
        import torch
    except ImportError:  # pragma: no cover — torch é dependência obrigatória do projeto
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
