"""Testes de src/recsys/utils/seed.py."""

from __future__ import annotations

import os
import random

from recsys.utils.seed import set_seed


def test_set_seed_sets_pythonhashseed() -> None:
    set_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_set_seed_makes_random_deterministic() -> None:
    set_seed(7)
    first = [random.random() for _ in range(3)]
    set_seed(7)
    second = [random.random() for _ in range(3)]
    assert first == second


def test_set_seed_seeds_cuda_when_available(monkeypatch) -> None:
    import torch

    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", calls.append)
    set_seed(99)
    # torch.manual_seed() já semeia CUDA internamente quando disponível; nosso
    # `if torch.cuda.is_available(): manual_seed_all(seed)` explícito soma mais
    # uma chamada — redundante, mas inofensivo (mesma seed nas duas vezes).
    assert calls == [99, 99]
