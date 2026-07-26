"""Pré-processadores de dados (padrão Strategy).

Cada pré-processador é uma estratégia intercambiável que segue o protocolo
:class:`Preprocessor`. :class:`PreprocessorPipeline` compõe estratégias em
ordem. Implementações concretas cobrem o RetailRocket (colunas ``visitorid``,
``itemid``, ``event``, ``timestamp``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Preprocessor(Protocol):
    """Estratégia de pré-processamento (interface estrutural)."""

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Aprende parâmetros a partir de ``df`` e retorna ``self``."""
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação aprendida a ``df``."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encadeia ``fit`` e ``transform``."""
        ...


class BasePreprocessor(ABC):
    """Base que deriva ``fit_transform`` de ``fit`` + ``transform``."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Aprende parâmetros a partir de ``df`` e retorna ``self``."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação aprendida a ``df``."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta e transforma em um passo."""
        return self.fit(df).transform(df)


class LabelEncoderPreprocessor(BasePreprocessor):
    """Mapeia IDs categóricos para inteiros contíguos ``0..n-1`` por coluna."""

    def __init__(self, columns: Sequence[str]) -> None:
        """Inicializa com as colunas a codificar."""
        self._columns = list(columns)
        self._mapping: dict[str, dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame) -> LabelEncoderPreprocessor:
        """Constrói o mapa valor→índice para cada coluna."""
        for col in self._columns:
            uniques = sorted(df[col].unique())
            self._mapping[col] = {value: idx for idx, value in enumerate(uniques)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Substitui os valores das colunas pelos índices aprendidos."""
        out = df.copy()
        for col in self._columns:
            out[col] = out[col].map(self._mapping[col])
        return out

    @property
    def mapping(self) -> dict[str, dict[Any, int]]:
        """Mapa aprendido por coluna (somente leitura após ``fit``)."""
        return self._mapping


class MinInteractionsFilter(BasePreprocessor):
    """Remove linhas cujo valor em ``column`` ocorre menos que o mínimo."""

    def __init__(self, column: str, min_interactions: int) -> None:
        """Inicializa com a coluna alvo e o mínimo de ocorrências."""
        self._column = column
        self._min = min_interactions
        self._keep: set[Any] = set()

    def fit(self, df: pd.DataFrame) -> MinInteractionsFilter:
        """Determina os valores que atingem o mínimo de interações."""
        counts = df[self._column].value_counts()
        self._keep = set(counts[counts >= self._min].index)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mantém apenas as linhas com valores frequentes o suficiente."""
        mask = df[self._column].isin(self._keep)
        return df[mask].reset_index(drop=True)


class PreprocessorPipeline:
    """Compõe pré-processadores aplicando-os em sequência."""

    def __init__(self, steps: Sequence[Preprocessor]) -> None:
        """Inicializa com os passos, na ordem de aplicação."""
        self._steps = list(steps)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta e aplica cada passo, encadeando o resultado."""
        result = df
        for step in self._steps:
            result = step.fit_transform(result)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica cada passo já ajustado, encadeando o resultado."""
        result = df
        for step in self._steps:
            result = step.transform(result)
        return result
