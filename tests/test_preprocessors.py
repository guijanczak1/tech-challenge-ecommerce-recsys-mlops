"""Testes de src/recsys/data/preprocessors.py."""

from __future__ import annotations

import pandas as pd

from recsys.data.preprocessors import (
    LabelEncoderPreprocessor,
    MinInteractionsFilter,
    Preprocessor,
    PreprocessorPipeline,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "visitorid": [10, 10, 10, 22, 22, 99],
            "itemid": ["a", "b", "a", "a", "c", "a"],
            "event": ["view", "view", "addtocart", "view", "view", "view"],
        }
    )


def test_label_encoder_maps_to_contiguous_ints() -> None:
    enc = LabelEncoderPreprocessor(columns=["visitorid", "itemid"])
    out = enc.fit_transform(_events())
    assert sorted(out["visitorid"].unique()) == [0, 1, 2]
    assert sorted(out["itemid"].unique()) == [0, 1, 2]
    assert enc.mapping["itemid"] == {"a": 0, "b": 1, "c": 2}


def test_min_interactions_filter_removes_infrequent() -> None:
    filt = MinInteractionsFilter(column="visitorid", min_interactions=2)
    out = filt.fit_transform(_events())
    # visitor 99 tem 1 interação -> removido; 10 (3) e 22 (2) permanecem.
    assert set(out["visitorid"]) == {10, 22}
    assert len(out) == 5


def test_pipeline_chains_steps_in_order() -> None:
    pipeline = PreprocessorPipeline(
        steps=[
            MinInteractionsFilter(column="visitorid", min_interactions=2),
            LabelEncoderPreprocessor(columns=["visitorid", "itemid"]),
        ]
    )
    out = pipeline.fit_transform(_events())
    assert 99 not in set(out["visitorid"])  # filtrado antes de codificar
    assert set(out["visitorid"]) == {0, 1}  # dois visitantes -> índices 0..1
    assert len(out) == 5


def test_concretes_satisfy_protocol() -> None:
    assert isinstance(LabelEncoderPreprocessor(["itemid"]), Preprocessor)
    assert isinstance(MinInteractionsFilter("visitorid", 2), Preprocessor)


def test_pipeline_transform_uses_fitted_steps() -> None:
    pipeline = PreprocessorPipeline(steps=[LabelEncoderPreprocessor(columns=["itemid"])])
    pipeline.fit_transform(_events())  # ajusta os passos
    out = pipeline.transform(_events())  # reaplica sem reajustar
    assert set(out["itemid"]) == {0, 1, 2}
