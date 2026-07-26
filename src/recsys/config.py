"""Configuração tipada do projeto (Pydantic Settings).

Combina `.env` (caminhos/segredos, via :class:`EnvSettings`) com
`configs/params.yaml` (hiperparâmetros), expondo :class:`AppConfig` com os
blocos ``data``, ``model``, ``training`` e ``evaluation``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Fonte e pré-processamento dos dados (RetailRocket)."""

    raw_dir: str
    processed_dir: str
    events_file: str
    user_col: str
    item_col: str
    event_col: str
    timestamp_col: str
    min_interactions: int = Field(ge=1)
    test_size: float = Field(gt=0.0, lt=1.0)


class ModelConfig(BaseModel):
    """Hiperparâmetros do modelo de recomendação."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    embedding_dim: int = Field(gt=0)
    hidden_dims: list[int]
    dropout: float = Field(ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Parâmetros do laço de treino."""

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    patience: int = Field(ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class EvaluationConfig(BaseModel):
    """Configuração da avaliação (métricas top-k)."""

    k: int = Field(gt=0)
    metrics: list[str]


class AppConfig(BaseModel):
    """Configuração agregada do projeto, montada a partir do YAML."""

    model_config = ConfigDict(protected_namespaces=())

    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


class EnvSettings(BaseSettings):
    """Valores vindos do ambiente / arquivo `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    params_file: Path = Field(default=Path("configs/params.yaml"), alias="RECSYS_PARAMS_FILE")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")


def load_params(path: Path) -> dict[str, Any]:
    """Lê o YAML de hiperparâmetros.

    Args:
        path: Caminho do arquivo `params.yaml`.

    Returns:
        Dicionário com as seções ``seed``, ``data``, ``model``,
        ``training`` e ``evaluation``.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"params file não encontrado: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(env: EnvSettings | None = None) -> AppConfig:
    """Monta a :class:`AppConfig` a partir do ambiente e do YAML.

    Args:
        env: Configurações de ambiente; se ``None``, são lidas do `.env`.

    Returns:
        Configuração validada e tipada do projeto.
    """
    env = env or EnvSettings()
    params = load_params(env.params_file)
    return AppConfig(**params)
