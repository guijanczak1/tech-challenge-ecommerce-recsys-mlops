"""Factory de modelos de recomendação (padrões Factory + Registry).

Builders são registrados por nome com o decorador :func:`register_model` e
instanciados por :func:`create_model`. Isso desacopla o código cliente das
implementações concretas (princípio aberto/fechado): novos modelos entram só
registrando, sem alterar quem consome.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

ModelBuilder = Callable[..., Any]
BuilderT = TypeVar("BuilderT", bound=ModelBuilder)

_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[BuilderT], BuilderT]:
    """Cria um decorador que registra um builder de modelo sob ``name``.

    Args:
        name: Chave única que identifica o modelo.

    Returns:
        Decorador que registra e devolve o próprio builder.

    Raises:
        ValueError: Se ``name`` já estiver registrado.
    """

    def decorator(builder: BuilderT) -> BuilderT:
        if name in _REGISTRY:
            raise ValueError(f"modelo já registrado: {name!r}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def create_model(name: str, **kwargs: Any) -> Any:
    """Instancia o modelo registrado sob ``name``.

    Args:
        name: Chave do modelo desejado.
        **kwargs: Argumentos repassados ao builder registrado.

    Returns:
        A instância criada pelo builder.

    Raises:
        ValueError: Se ``name`` não estiver registrado.
    """
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"modelo desconhecido: {name!r}. Disponíveis: {available_models()}"
        ) from None
    return builder(**kwargs)


def available_models() -> list[str]:
    """Retorna os nomes de modelos registrados, em ordem alfabética."""
    return sorted(_REGISTRY)
