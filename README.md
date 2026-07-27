# tech-challenge-ecommerce-recsys-mlops

Tech Challenge — Pós ML FIAP (Fase 02). Sistema de **recomendação de produtos
de e-commerce**: rede neural PyTorch (embeddings + MLP), pipeline reprodutível
com **DVC**, experimentos e Model Registry no **MLflow**, containerização com
**Docker** e código em clean code. Dataset: **RetailRocket**.

## Requisitos
- **Python 3.12** · **Poetry** · (opcional) **Docker**

## Setup

```bash
poetry env use python3.12
poetry install
poetry run pre-commit install
cp .env.example .env        # (Windows: copy .env.example .env)
poetry run python scripts/validate_env.py
```

## Pipeline reprodutível (DVC)

```bash
poetry run dvc repro          # prepare → preprocess → feature_eng → train → evaluate
poetry run dvc dag            # visualiza o grafo de estágios
cat metrics.json              # métricas da última avaliação
```

Se `data/raw/events.csv` (RetailRocket real) não existir, o estágio `prepare`
gera uma amostra sintética reprodutível com o mesmo schema.

## Experimentos e Model Registry (MLflow)

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

O `train` registra o modelo como `recsys-mlp` (alias `staging`); o `evaluate`
promove para `production` quando o NDCG@k atinge o mínimo (`configs/params.yaml`).

## Docker

```bash
docker compose up --build     # sobe o servidor MLflow + roda o pipeline de treino
```

## Resultados (amostra sintética)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| MLP (PyTorch) | 0.121 | 0.408 | 0.347 | 0.241 |
| Popularidade | 0.121 | 0.411 | 0.348 | 0.241 |
| SVD (scikit-learn) | 0.074 | 0.252 | 0.212 | 0.137 |

Detalhes, limitações e vieses: [`docs/model_card.md`](docs/model_card.md).

## Qualidade

```bash
poetry run pytest -q                                          # testes + cobertura (>=80%)
poetry run ruff check . && poetry run ruff format --check .   # lint + formato
```

## Estrutura
```
src/recsys/
  config.py              # Pydantic Settings (.env + params.yaml)
  data/       (loaders, synthetic, preprocessors, dataset)
  models/     (factory, base, mlp, baselines)
  training/   (trainer [Template Method], recsys_trainer)
  evaluation/ (metrics: precision/recall/ndcg/map @k)
  pipeline/   (prepare, preprocess, feature_eng, train, evaluate)
configs/params.yaml · dvc.yaml · Dockerfile · docker-compose.yml · docs/model_card.md
```

## Entregas
- Obrigatória: repositório + **vídeo STAR de 5 min**.
- Bônus: deploy em nuvem (AWS) — container acessível por URL.
