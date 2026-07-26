# tech-challenge-ecommerce-recsys-mlops

Tech Challenge — Pós ML FIAP (Fase 02). Sistema de **recomendação de produtos
de e-commerce** com rede neural PyTorch, pipeline versionado (DVC), experimentos
no MLflow e código clean code. Dataset: **RetailRocket**.

> Documento em evolução por etapa. Uso do modelo/serviço será detalhado na
> Etapa 4. Processo e rúbrica: ver [`HARNESS.md`](HARNESS.md); regras de código:
> [`CLAUDE.md`](CLAUDE.md).

## Requisitos
- **Python 3.12**
- **Poetry** (gerenciamento de dependências)

## Setup do ambiente (Etapa 2)

```bash
# 1. aponte o Poetry para um Python 3.12 e instale tudo
poetry env use python3.12
poetry install

# 2. instale os hooks de qualidade (ruff) e de commit semântico
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg

# 3. configure variáveis de ambiente
cp .env.example .env        # (Windows: copy .env.example .env)

# 4. valide o ambiente
poetry run python scripts/validate_env.py
```

## Comandos úteis

```bash
poetry run pytest -q                                          # testes
poetry run ruff check . && poetry run ruff format --check .   # lint + formato
poetry run python scripts/audit_rubric.py --etapa 2           # auditoria da rúbrica
```

## Estrutura
`src/recsys/` (config, data, models, training, evaluation, utils) ·
`tests/` · `configs/params.yaml` · `scripts/` · `docs/plano_commits.md`.

## Entregas
- Obrigatória: repositório + **vídeo STAR de 5 min**.
- Bônus: deploy em nuvem (AWS) — container acessível por URL.
