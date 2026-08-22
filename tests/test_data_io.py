"""Testes de src/recsys/data/synthetic.py e loaders.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import DataConfig
from recsys.data.loaders import ensure_events, load_events, raw_events_path
from recsys.data.synthetic import generate_events

_EXPECTED_COLS = {"timestamp", "visitorid", "event", "itemid", "transactionid"}


def _cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        raw_dir=str(tmp_path),
        processed_dir=str(tmp_path / "processed"),
        events_file="events.csv",
        user_col="visitorid",
        item_col="itemid",
        event_col="event",
        timestamp_col="timestamp",
        min_interactions=1,
        test_size=0.2,
        synthetic={"n_users": 50, "n_items": 20, "n_events": 500},
    )


def test_generate_events_schema_and_size() -> None:
    df = generate_events(n_users=50, n_items=20, n_events=500, seed=42)
    assert set(df.columns) == _EXPECTED_COLS
    assert len(df) == 500
    assert set(df["event"].unique()) <= {"view", "addtocart", "transaction"}
    assert df["timestamp"].is_monotonic_increasing


def test_generate_events_is_deterministic() -> None:
    a = generate_events(50, 20, 300, seed=7)
    b = generate_events(50, 20, 300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_ensure_events_generates_when_missing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = ensure_events(cfg, seed=42)
    assert path == raw_events_path(cfg)
    assert path.exists()
    assert len(pd.read_csv(path)) == 500


def test_ensure_events_keeps_existing_real_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    real = raw_events_path(cfg)
    real.write_text("timestamp,visitorid,event,itemid,transactionid\n1,1,view,1,\n", "utf-8")
    ensure_events(cfg, seed=42)  # não deve sobrescrever
    assert len(pd.read_csv(real)) == 1


def test_load_events_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_events(_cfg(tmp_path))


def test_load_events_returns_dataframe_when_present(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ensure_events(cfg, seed=42)
    df = load_events(cfg)
    assert len(df) == 500
    assert set(df.columns) == _EXPECTED_COLS
