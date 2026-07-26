"""Testes de src/recsys/models/factory.py."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from recsys.models import factory


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Isola o registro global entre os testes."""
    snapshot = dict(factory._REGISTRY)
    yield
    factory._REGISTRY.clear()
    factory._REGISTRY.update(snapshot)


def test_register_and_create_valid() -> None:
    @factory.register_model("dummy")
    def _build(**kwargs: object) -> dict[str, object]:
        return {"kind": "dummy", **kwargs}

    obj = factory.create_model("dummy", units=8)
    assert obj == {"kind": "dummy", "units": 8}


def test_create_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        factory.create_model("inexistente")


def test_duplicate_registration_raises_value_error() -> None:
    @factory.register_model("dup")
    def _first() -> int:
        return 1

    with pytest.raises(ValueError, match="já registrado"):

        @factory.register_model("dup")
        def _second() -> int:
            return 2


def test_available_models_is_sorted() -> None:
    factory._REGISTRY.clear()

    @factory.register_model("beta")
    def _b() -> None: ...

    @factory.register_model("alpha")
    def _a() -> None: ...

    assert factory.available_models() == ["alpha", "beta"]
