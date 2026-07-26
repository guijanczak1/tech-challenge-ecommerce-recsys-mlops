# tech-challenge-ecommerce-recsys-mlops

Tech Challenge — Pós ML FIAP (Fase 02). Sistema de **recomendação de produtos
de e-commerce** com rede neural PyTorch, pipeline versionado (DVC), experimentos
no MLflow e código seguindo clean code. Dataset: **RetailRocket**.

> Documento em evolução: o uso do modelo/serviço será detalhado ao longo das
> etapas do projeto.

## Requisitos
- **Python 3.12**
- **Poetry** (gerenciamento de dependências)

## Setup do ambiente

```bash
# 1. aponte o Poetry para um Python 3.12 e instale tudo
poetry env use python3.12
poetry install

# 2. instale os hooks de qualidade (ruff)
poetry run pre-commit install

# 3. configure variáveis de ambiente
cp .env.example .env        # (Windows: copy .env.example .env)

# 4. valide o ambiente
poetry run python scripts/validate_env.py
```

## Comandos úteis

```bash
poetry run pytest -q                                          # testes + cobertura
poetry run ruff check . && poetry run ruff format --check .   # lint + formato
```

## Estrutura
`src/recsys/` (config, data, models, training, evaluation, utils) ·
`tests/` · `configs/params.yaml` · `scripts/validate_env.py`.

## Entregas
- Obrigatória: repositório + **vídeo STAR de 5 min**.
- Bônus: deploy em nuvem (AWS) — container acessível por URL.
