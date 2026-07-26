"""Testes das funções de treino/avaliação (sem IO nem MLflow)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import EnvSettings, load_config
from recsys.models.baselines import PopularityRecommender
from recsys.pipeline.evaluate import evaluate_model, per_user_relevant
from recsys.pipeline.train import fit_baseline

PARAMS = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"


def _cfg():
    return load_config(EnvSettings(params_file=PARAMS))


def test_fit_baseline_returns_fitted_popularity() -> None:
    train_df = pd.DataFrame({"itemid": [1, 2, 2, 2, 3, 3], "weight": [1.0] * 6})
    model = fit_baseline(train_df, _cfg())
    assert isinstance(model, PopularityRecommender)
    assert model.ranked_items[0] == 2  # item mais frequente


def test_per_user_relevant_groups_items() -> None:
    test = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 20, 30]})
    assert per_user_relevant(test, "visitorid", "itemid") == {1: {10, 20}, 2: {30}}


def test_evaluate_model_computes_mean_metrics() -> None:
    class StubModel:
        def recommend(self, k: int) -> list[int]:
            return [10, 20, 30][:k]

    test = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 99, 30]})
    metrics = evaluate_model(StubModel(), test, _cfg())
    assert metrics["n_users"] == 2.0
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert metrics["recall_at_k"] == pytest.approx(0.75)  # user1 1/2, user2 1/1
