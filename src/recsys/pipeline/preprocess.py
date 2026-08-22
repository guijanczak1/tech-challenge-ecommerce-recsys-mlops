"""Estágio ``preprocess``: filtra interações raras e codifica IDs.

Lê ``data/raw`` (via loader), aplica o pipeline de preprocessors (Strategy) e
grava ``data/processed/interactions.parquet``.
"""

from __future__ import annotations

from pathlib import Path

from recsys.config import AppConfig, load_config
from recsys.data.loaders import load_events
from recsys.data.preprocessors import (
    LabelEncoderPreprocessor,
    MinInteractionsFilter,
    PreprocessorPipeline,
)

INTERACTIONS_FILE = "interactions.parquet"


def build_pipeline(cfg: AppConfig) -> PreprocessorPipeline:
    """Monta o pipeline de preprocessors a partir da configuração."""
    data = cfg.data
    return PreprocessorPipeline(
        steps=[
            MinInteractionsFilter(data.user_col, data.min_interactions),
            MinInteractionsFilter(data.item_col, data.min_interactions),
            LabelEncoderPreprocessor([data.user_col, data.item_col]),
        ]
    )


def run(cfg: AppConfig) -> Path:
    """Executa o preprocess e devolve o caminho do parquet gerado."""
    events = load_events(cfg.data)
    processed = build_pipeline(cfg).fit_transform(events)
    out = Path(cfg.data.processed_dir) / INTERACTIONS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(out, index=False)
    return out


def main() -> None:
    """Ponto de entrada do estágio (CLI/DVC)."""
    out = run(load_config())
    print(f"[preprocess] gravado {out}")


if __name__ == "__main__":  # pragma: no cover — main() já é testado diretamente
    main()
