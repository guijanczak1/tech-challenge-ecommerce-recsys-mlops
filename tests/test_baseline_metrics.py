"""Testes do baseline de popularidade e das métricas top-k."""

from __future__ import annotations

import pandas as pd

from recsys.evaluation.metrics import precision_at_k, recall_at_k
from recsys.models.baselines import PopularityRecommender
from recsys.models.factory import create_model


def test_popularity_ranks_by_weight_sum() -> None:
    interactions = pd.DataFrame(
        {
            "itemid": [1, 1, 2, 2, 2, 3],
            "weight": [1.0, 1.0, 3.0, 3.0, 3.0, 1.0],
        }
    )
    model = PopularityRecommender().fit(interactions, item_col="itemid", weight_col="weight")
    assert model.ranked_items == [2, 1, 3]  # 2:9, 1:2, 3:1
    assert model.recommend(2) == [2, 1]


def test_popularity_is_registered_in_factory() -> None:
    assert isinstance(create_model("popularity"), PopularityRecommender)


def test_precision_at_k() -> None:
    recommended = [10, 20, 30, 40]
    relevant = {20, 40, 99}
    assert precision_at_k(recommended, relevant, k=4) == 0.5  # 2 de 4
    assert precision_at_k(recommended, relevant, k=0) == 0.0


def test_recall_at_k() -> None:
    recommended = [10, 20, 30, 40]
    relevant = {20, 40, 99}
    assert recall_at_k(recommended, relevant, k=4) == 2 / 3  # achou 2 de 3
    assert recall_at_k(recommended, set(), k=4) == 0.0
