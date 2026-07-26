"""Carregamento do dataset bruto de eventos (real ou sintético)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.config import DataConfig
from recsys.data.synthetic import generate_events


def raw_events_path(cfg: DataConfig) -> Path:
    """Caminho do CSV de eventos brutos (``data/raw/<events_file>``)."""
    return Path(cfg.raw_dir) / cfg.events_file


def ensure_events(cfg: DataConfig, seed: int) -> Path:
    """Garante o CSV de eventos: mantém o real se existir, senão gera sintético.

    Args:
        cfg: Configuração de dados (caminhos + parâmetros do sintético).
        seed: Semente do gerador sintético.

    Returns:
        Caminho do CSV de eventos garantido em disco.
    """
    path = raw_events_path(cfg)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_events(
        n_users=cfg.synthetic.n_users,
        n_items=cfg.synthetic.n_items,
        n_events=cfg.synthetic.n_events,
        seed=seed,
    )
    frame.to_csv(path, index=False)
    return path


def load_events(cfg: DataConfig) -> pd.DataFrame:
    """Lê o CSV de eventos brutos como DataFrame.

    Raises:
        FileNotFoundError: Se o arquivo não existir (rode a etapa ``prepare``).
    """
    path = raw_events_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"eventos não encontrados: {path}. Rode a etapa 'prepare'.")
    return pd.read_csv(path)
