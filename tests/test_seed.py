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
