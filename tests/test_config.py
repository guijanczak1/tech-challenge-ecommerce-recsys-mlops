"""Testes de src/recsys/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from recsys.config import (
    AppConfig,
    DataConfig,
    EnvSettings,
    load_config,
    load_params,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS = REPO_ROOT / "configs" / "params.yaml"


def test_load_config_from_repo_params() -> None:
    cfg = load_config(EnvSettings(params_file=PARAMS))
    assert isinstance(cfg, AppConfig)
    assert cfg.seed == 42
    assert cfg.data.user_col == "visitorid"
    assert cfg.data.item_col == "itemid"
    assert cfg.model.name == "mlp"
    assert cfg.training.patience == 3
    assert cfg.evaluation.k == 10


def test_env_settings_default_params_file() -> None:
    assert EnvSettings().params_file == Path("configs/params.yaml")


def test_env_override_params_file(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(_minimal_params_yaml(seed=7), encoding="utf-8")
    cfg = load_config(EnvSettings(params_file=custom))
    assert cfg.seed == 7
    assert cfg.model.embedding_dim == 8


def test_load_params_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nao_existe.yaml")


def test_invalid_test_size_raises() -> None:
    with pytest.raises(ValidationError):
        DataConfig(
            raw_dir="d",
            processed_dir="p",
            events_file="e.csv",
            user_col="u",
            item_col="i",
            event_col="ev",
            timestamp_col="t",
            min_interactions=5,
            test_size=1.5,  # inválido: deve estar em (0, 1)
        )


def test_invalid_device_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    params = yaml.safe_load(_minimal_params_yaml(seed=1))
    params["training"]["device"] = "tpu"  # não permitido
    bad.write_text(yaml.safe_dump(params), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(EnvSettings(params_file=bad))


def _minimal_params_yaml(seed: int) -> str:
    return yaml.safe_dump(
        {
            "seed": seed,
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "events_file": "events.csv",
                "user_col": "visitorid",
                "item_col": "itemid",
                "event_col": "event",
                "timestamp_col": "timestamp",
                "min_interactions": 5,
                "test_size": 0.2,
            },
            "model": {
                "name": "mlp",
                "embedding_dim": 8,
                "hidden_dims": [16, 8],
                "dropout": 0.1,
            },
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 0.01,
                "patience": 2,
                "device": "cpu",
            },
            "evaluation": {"k": 5, "metrics": ["precision_at_k"]},
        }
    )
