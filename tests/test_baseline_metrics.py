"""Testes do baseline de popularidade e das métricas top-k."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import Recommender
from recsys.models.baselines import PopularityRecommender, SvdRecommender
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
    assert model.recommend(user=0, k=2) == [2, 1]  # user ignorado (global)


def test_popularity_is_registered_in_factory() -> None:
    assert isinstance(create_model("popularity"), PopularityRecommender)


def test_svd_recommender_fits_and_recommends() -> None:
    interactions = pd.DataFrame(
        {
            "visitorid": [0, 0, 1, 2, 3, 3],
            "itemid": [1, 2, 2, 3, 4, 5],
            "weight": [1.0, 2.0, 1.0, 3.0, 1.0, 1.0],
        }
    )
    model = SvdRecommender(n_components=3, seed=1).fit(
        interactions, "visitorid", "itemid", "weight", n_users=4, n_items=6
    )
    recs = model.recommend(user=0, k=3)
    assert len(recs) == 3
    assert all(0 <= item < 6 for item in recs)
    assert isinstance(model, Recommender)


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


def test_ndcg_at_k() -> None:
    recommended = [10, 20, 30, 40]
    relevant = {20, 40}  # posições 2 e 4
    assert ndcg_at_k(recommended, relevant, k=4) == pytest.approx(0.6509, abs=1e-3)
    assert ndcg_at_k(recommended, set(), k=4) == 0.0


def test_average_precision_at_k() -> None:
    recommended = [10, 20, 30, 40]
    relevant = {20, 40}  # precisão em 1/2 e 2/4 -> AP = (0.5 + 0.5) / 2
    assert average_precision_at_k(recommended, relevant, k=4) == pytest.approx(0.5)
    assert average_precision_at_k(recommended, set(), k=4) == 0.0
