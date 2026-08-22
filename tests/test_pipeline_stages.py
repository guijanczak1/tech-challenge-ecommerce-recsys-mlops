"""Testes de orquestração dos estágios do pipeline (CLI/DVC).

Cada teste roda os estágios de ponta a ponta num diretório temporário (dados,
modelo e MLflow isolados) — nunca toca em `models/mlp.pt` nem `mlflow.db` do
repositório, que são o artefato/registro reais usados em produção.
"""

from __future__ import annotations

import json
from pathlib import Path

from recsys.config import AppConfig, load_params
from recsys.data.loaders import ensure_events
from recsys.pipeline import evaluate, feature_eng, prepare, preprocess, train

PARAMS = Path(__file__).resolve().parent.parent / "configs" / "params.yaml"


def _cfg(tmp_path: Path) -> AppConfig:
    """Configuração real do projeto, mas apontando dados/modelo pro tmp_path."""
    raw = load_params(PARAMS)
    raw["data"]["raw_dir"] = str(tmp_path / "raw")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["data"]["min_interactions"] = 1
    raw["data"]["synthetic"] = {"n_users": 15, "n_items": 10, "n_events": 200}
    raw["training"]["epochs"] = 1
    raw["evaluation"]["min_ndcg"] = 0.0  # garante promoção determinística no teste
    return AppConfig(**raw)


def test_prepare_main_generates_synthetic_dataset(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(prepare, "load_config", lambda: cfg)
    prepare.main()
    assert (tmp_path / "raw" / cfg.data.events_file).exists()


def test_preprocess_run_writes_parquet(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    ensure_events(cfg.data, seed=cfg.seed)
    out = preprocess.run(cfg)
    assert out.exists()


def test_preprocess_main(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    ensure_events(cfg.data, seed=cfg.seed)
    monkeypatch.setattr(preprocess, "load_config", lambda: cfg)
    preprocess.main()
    assert (Path(cfg.data.processed_dir) / preprocess.INTERACTIONS_FILE).exists()


def test_feature_eng_run_writes_train_test_and_dims(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    ensure_events(cfg.data, seed=cfg.seed)
    preprocess.run(cfg)
    train_path, test_path = feature_eng.run(cfg)
    assert train_path.exists()
    assert test_path.exists()
    dims = json.loads((Path(cfg.data.processed_dir) / feature_eng.DIMS_FILE).read_text())
    assert dims["n_users"] > 0
    assert dims["n_items"] > 0


def test_feature_eng_main(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    ensure_events(cfg.data, seed=cfg.seed)
    preprocess.run(cfg)
    monkeypatch.setattr(feature_eng, "load_config", lambda: cfg)
    feature_eng.main()
    assert (Path(cfg.data.processed_dir) / feature_eng.TRAIN_FILE).exists()


def _prepared_cfg(tmp_path: Path) -> AppConfig:
    """Config com prepare→preprocess→feature_eng já rodados (fixture manual)."""
    cfg = _cfg(tmp_path)
    ensure_events(cfg.data, seed=cfg.seed)
    preprocess.run(cfg)
    feature_eng.run(cfg)
    return cfg


def test_train_main_trains_and_registers_model(tmp_path, monkeypatch) -> None:
    cfg = _prepared_cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    monkeypatch.setattr(train, "load_config", lambda: cfg)

    train.main()

    assert (tmp_path / "models" / train.MODEL_FILE).exists()


def test_evaluate_main_computes_and_writes_metrics(tmp_path, monkeypatch) -> None:
    cfg = _prepared_cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    train.main()
    monkeypatch.setattr(evaluate, "load_config", lambda: cfg)

    evaluate.main()

    results = json.loads((tmp_path / evaluate.METRICS_FILE).read_text())
    assert {"mlp", "popularity", "svd", "promoted_to_production"} <= set(results)
    assert results["promoted_to_production"] is True  # min_ndcg=0.0 garante isso
