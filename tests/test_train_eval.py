"""Testes das funções puras de avaliação (sem IO nem MLflow)."""

from __future__ import annotations

import pandas as pd

from recsys.pipeline.evaluate import evaluate_recommender, per_user_relevant


class _StubModel:
    """Recomendador fixo (ignora o usuário) para testar as métricas."""

    def recommend(self, user: int, k: int) -> list[int]:
        return [10, 20, 30][:k]


def test_per_user_relevant_groups_items() -> None:
    test = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 20, 30]})
    assert per_user_relevant(test, "visitorid", "itemid") == {1: {10, 20}, 2: {30}}


def test_evaluate_recommender_returns_four_metrics() -> None:
    relevant = {1: {10, 20}, 2: {30}}
    metrics = evaluate_recommender(_StubModel(), relevant, k=3)
    assert set(metrics) == {"precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_evaluate_recommender_empty_is_zero() -> None:
    metrics = evaluate_recommender(_StubModel(), {}, k=3)
    assert set(metrics) == {"precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"}
    assert all(value == 0.0 for value in metrics.values())
