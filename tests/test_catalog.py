"""Testes do catálogo fictício de produtos."""

from __future__ import annotations

from recsys.data.catalog import product_name


def test_product_name_is_deterministic() -> None:
    assert product_name(42) == product_name(42)


def test_product_name_has_expected_shape() -> None:
    result = product_name(0)
    assert set(result) == {"name", "category"}
    assert result["category"]
    assert "Mod. 0000" in result["name"]


def test_different_items_can_have_different_categories() -> None:
    categories = {product_name(i)["category"] for i in range(12)}
    assert len(categories) > 1


def test_product_name_stable_across_large_ids() -> None:
    # não deve estourar índice mesmo para IDs bem maiores que o catálogo base
    result = product_name(999_999)
    assert result["name"].endswith("Mod. 999999")
