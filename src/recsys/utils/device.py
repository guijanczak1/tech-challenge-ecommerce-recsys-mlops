"""Resolução do device do PyTorch."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` num ``torch.device``.

    ``auto`` usa CUDA se disponível, senão CPU.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
