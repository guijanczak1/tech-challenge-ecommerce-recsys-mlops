"""Estágio ``feature_eng``: pondera eventos e faz split temporal treino/teste.

Lê ``interactions.parquet``, atribui um peso implícito por tipo de evento e
separa treino/teste por tempo (os eventos mais recentes viram teste).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.config import AppConfig, load_config
from recsys.pipeline.preprocess import INTERACTIONS_FILE

TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
_EVENT_WEIGHTS = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


def add_weight(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """Adiciona a coluna ``weight`` a partir do tipo de evento."""
    out = df.copy()
    out["weight"] = out[event_col].map(_EVENT_WEIGHTS).fillna(1.0)
    return out


def temporal_split(
    df: pd.DataFrame, timestamp_col: str, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide por tempo: fração final ``test_size`` vira teste."""
    ordered = df.sort_values(timestamp_col)
    cutoff = int(len(ordered) * (1.0 - test_size))
    train = ordered.iloc[:cutoff].reset_index(drop=True)
    test = ordered.iloc[cutoff:].reset_index(drop=True)
    return train, test


def run(cfg: AppConfig) -> tuple[Path, Path]:
    """Executa o feature engineering e devolve os caminhos treino/teste."""
    processed_dir = Path(cfg.data.processed_dir)
    interactions = pd.read_parquet(processed_dir / INTERACTIONS_FILE)
    weighted = add_weight(interactions, cfg.data.event_col)
    train, test = temporal_split(weighted, cfg.data.timestamp_col, cfg.data.test_size)
    train_path, test_path = processed_dir / TRAIN_FILE, processed_dir / TEST_FILE
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train_path, test_path


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC)."""
    train_path, test_path = run(load_config())
    print(f"[feature_eng] treino={train_path} teste={test_path}")


if __name__ == "__main__":
    main()
