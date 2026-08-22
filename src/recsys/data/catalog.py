"""Catálogo de produtos FICTÍCIO, só para tornar a demonstração legível.

O RetailRocket (dataset real sugerido no enunciado) não expõe nomes de
produto: os arquivos reais trazem apenas ``itemid`` e propriedades
**hasheadas**, por anonimização comercial. Não há nome real para recuperar,
nem trocando pelo dataset real.

Este módulo gera nomes determinísticos (mesmo ``item_id`` -> sempre o mesmo
nome) a partir de um pequeno catálogo de categorias e modelos plausíveis,
apenas para exibição na API/demo. Não representam produtos reais.
"""

from __future__ import annotations

_CATALOG: dict[str, list[str]] = {
    "Eletrônicos": [
        "Fone de Ouvido Bluetooth",
        "Carregador Portátil",
        "Smartwatch",
        "Caixa de Som",
        "Mouse sem Fio",
    ],
    "Casa e Decoração": [
        "Luminária de Mesa",
        "Jogo de Panelas",
        "Almofada Decorativa",
        "Organizador de Gaveta",
    ],
    "Moda": [
        "Camiseta Estampada",
        "Tênis Casual",
        "Jaqueta Jeans",
        "Bolsa Transversal",
    ],
    "Esporte e Lazer": [
        "Garrafa Térmica",
        "Tapete de Yoga",
        "Corda de Pular",
        "Luva de Treino",
    ],
    "Beleza": [
        "Creme Hidratante",
        "Escova Secadora",
        "Kit de Maquiagem",
        "Perfume",
    ],
    "Livros e Papelaria": [
        "Caderno Universitário",
        "Kit de Canetas",
        "Agenda",
        "Marcador de Página",
    ],
}
_CATEGORIES = list(_CATALOG)


def product_name(item_id: int) -> dict[str, str]:
    """Nome e categoria fictícios e determinísticos para ``item_id``.

    Args:
        item_id: ID do item (mesmo índice usado pelo modelo de recomendação).

    Returns:
        Dicionário ``{"name": ..., "category": ...}``. Fictício — não
        corresponde a um produto real.
    """
    category = _CATEGORIES[item_id % len(_CATEGORIES)]
    templates = _CATALOG[category]
    template = templates[(item_id // len(_CATEGORIES)) % len(templates)]
    return {"name": f"{template} — Mod. {item_id:04d}", "category": category}
