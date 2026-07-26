"""Gerador de dados sintéticos no schema do RetailRocket.

Usado como fallback reprodutível quando o dataset real não está em
``data/raw``. Colunas: ``timestamp``, ``visitorid``, ``event``, ``itemid``,
``transactionid`` — iguais às do RetailRocket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EVENTS = ("view", "addtocart", "transaction")
_EVENT_PROBS = (0.85, 0.12, 0.03)
_BASE_TS_MS = 1_433_000_000_000  # ~2015, como o RetailRocket
_WINDOW_MS = 90 * 24 * 3600 * 1000  # 90 dias


def _item_probabilities(n_items: int) -> np.ndarray:
    """Distribuição de popularidade enviesada (Zipf-like) sobre os itens."""
    weights = 1.0 / np.arange(1, n_items + 1)
    return weights / weights.sum()


def _transaction_ids(events: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IDs de transação só para eventos ``transaction`` (NaN caso contrário)."""
    ids = rng.integers(1, events.size + 1, size=events.size).astype(float)
    return np.where(events == "transaction", ids, np.nan)


def generate_events(n_users: int, n_items: int, n_events: int, seed: int) -> pd.DataFrame:
    """Gera eventos user-item sintéticos, ordenados por tempo.

    Args:
        n_users: Número de visitantes distintos.
        n_items: Número de itens distintos.
        n_events: Número total de interações.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame com o schema do RetailRocket.
    """
    rng = np.random.default_rng(seed)
    events = rng.choice(_EVENTS, size=n_events, p=_EVENT_PROBS)
    frame = pd.DataFrame(
        {
            "timestamp": _BASE_TS_MS + rng.integers(0, _WINDOW_MS, size=n_events),
            "visitorid": rng.integers(0, n_users, size=n_events),
            "event": events,
            "itemid": rng.choice(n_items, size=n_events, p=_item_probabilities(n_items)),
            "transactionid": _transaction_ids(events, rng),
        }
    )
    return frame.sort_values("timestamp").reset_index(drop=True)
