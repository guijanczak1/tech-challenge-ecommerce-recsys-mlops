"""Testes dos estágios preprocess e feature_eng."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.config import EnvSettings, load_config
from recsys.pipeline.feature_eng import add_weight, temporal_split
from recsys.pipeline.preprocess import build_pipeline

PARAMS = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"


def test_build_pipeline_filters_rare_and_encodes() -> None:
    cfg = load_config(EnvSettings(params_file=PARAMS))  # min_interactions=5
    df = pd.DataFrame(
        {
            "visitorid": [10] * 6 + [99],
            "itemid": [100] * 7,
            "event": ["view"] * 7,
            "timestamp": range(7),
        }
    )
    out = build_pipeline(cfg).fit_transform(df)
    assert set(out["visitorid"]) == {0}  # visitante raro (99) removido, 10 -> 0
    assert set(out["itemid"]) == {0}
    assert len(out) == 6


def test_add_weight_maps_event_types() -> None:
    df = pd.DataFrame({"event": ["view", "addtocart", "transaction", "outro"]})
    out = add_weight(df, "event")
    assert list(out["weight"]) == [1.0, 2.0, 3.0, 1.0]


def test_temporal_split_orders_and_proportions() -> None:
    df = pd.DataFrame({"timestamp": [5, 1, 4, 2, 3], "v": [50, 10, 40, 20, 30]})
    train, test = temporal_split(df, "timestamp", test_size=0.4)
    assert list(train["timestamp"]) == [1, 2, 3]  # 60% mais antigos
    assert list(test["timestamp"]) == [4, 5]  # 40% mais recentes
