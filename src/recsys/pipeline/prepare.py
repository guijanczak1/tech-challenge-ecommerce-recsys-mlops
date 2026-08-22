"""Estágio ``prepare``: garante o dataset bruto em ``data/raw``.

Mantém o RetailRocket real se já estiver presente; senão gera uma amostra
sintética reprodutível (ver ``recsys.data.loaders.ensure_events``).
"""

from __future__ import annotations

from recsys.config import load_config
from recsys.data.loaders import ensure_events


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC)."""
    cfg = load_config()
    path = ensure_events(cfg.data, seed=cfg.seed)
    print(f"[prepare] dataset pronto em {path}")


if __name__ == "__main__":  # pragma: no cover — main() já é testado diretamente
    main()
