# tech-challenge-ecommerce-recsys-mlops

Tech Challenge — Sistema de **recomendação de produtos
de e-commerce**: rede neural PyTorch (embeddings + MLP), pipeline reprodutível
com **DVC**, experimentos e Model Registry no **MLflow**, API de inferência em
**FastAPI**, containerizado com **Docker** e **implantado na AWS (Lambda)**.
Código em clean code (SOLID, Factory/Strategy/Template Method).

## Status

| | |
|---|---|
| Repositório (Etapas 1–4 do enunciado) | ✅ completo |
| Testes | **77 passando**, cobertura **100%** (piso mínimo: 92%) |
| Modelo | `recsys-mlp` v3 — **Production** no MLflow Registry |
| Deploy (bônus) | ✅ **ao vivo** na AWS Lambda — ver [Deploy em produção](#deploy-em-produção-aws) |
| Vídeo STAR (5 min) | pendente |

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

## Estrutura do projeto

```
src/recsys/
  config.py               # Pydantic Settings (.env + configs/params.yaml)
  data/
    synthetic.py           # gera amostra sintética (fallback sem RetailRocket)
    loaders.py              # eventos brutos: real se existir, senão sintético
    preprocessors.py        # Strategy: filtro de interações + label encoding
    dataset.py               # torch Dataset (feedback implícito + negativos)
    catalog.py                # catálogo FICTÍCIO de nomes/categoria (demo)
  models/
    factory.py              # Factory: registro de modelos por decorador
    base.py                   # Protocol Recommender (contrato comum)
    mlp.py                     # RecsysMLP — embeddings + MLP (PyTorch)
    baselines.py             # PopularityRecommender + SvdRecommender
  training/
    trainer.py               # Template Method: BaseTrainer + early stopping
    recsys_trainer.py          # treino do MLP (BCE, device-agnostic)
  evaluation/
    metrics.py                # precision/recall/ndcg/map @k
  pipeline/                  # estágios do DVC (ver dvc.yaml)
    prepare.py · preprocess.py · feature_eng.py · train.py · evaluate.py
  serving/
    api.py                     # FastAPI: /health, /recommend
    loader.py                  # carrega o modelo embutido na imagem
  utils/
    seed.py · device.py       # reprodutibilidade + device-agnostic

tests/                     # espelha src/recsys — 77 testes, cobertura 100%
configs/params.yaml        # hiperparâmetros, seeds, thresholds
deploy/                    # Terraform: ECR + Lambda + Function URL
docs/model_card.md         # performance, limitações e vieses do modelo
Dockerfile                 # treino (usado pelo docker-compose)
Dockerfile.serve           # imagem de serving (self-contained, roda na Lambda)
dvc.yaml                    # pipeline: prepare → preprocess → feature_eng → train → evaluate
```

## Pipeline reprodutível (DVC)

```bash
poetry run dvc repro          # prepare → preprocess → feature_eng → train → evaluate
poetry run dvc dag            # visualiza o grafo de estágios
cat metrics.json               # métricas da última avaliação
```

Se `data/raw/events.csv` (RetailRocket real) não existir, o estágio `prepare`
gera uma amostra sintética reprodutível com o mesmo schema (2.000 usuários,
500 itens, 30.000 interações — seed fixa).

## Modelo e resultados

`recsys-mlp`: embeddings de usuário/item (64 dim) → MLP (128→64) → score de
afinidade. Treino com `BCEWithLogitsLoss`, negativos amostrados, early
stopping. Comparado com dois baselines em 4 métricas top-k:

| Modelo | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|---|
| MLP (PyTorch) | 0.121 | 0.408 | 0.347 | 0.241 |
| Popularidade | 0.121 | 0.411 | 0.348 | 0.241 |
| SVD (scikit-learn) | 0.074 | 0.252 | 0.212 | 0.137 |

> Na amostra **sintética** (popularidade enviesada tipo Zipf), o MLP empata
> com o baseline de popularidade — esperado nesse cenário. Com o RetailRocket
> real, a vantagem da personalização tende a aparecer: troque o arquivo e
> rode `dvc repro` de novo.

Detalhes, limitações e vieses: [`docs/model_card.md`](docs/model_card.md).

## Experimentos e Model Registry (MLflow)

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

O estágio `train` registra o modelo como `recsys-mlp` (alias `staging`); o
`evaluate` promove para `production` quando o NDCG@k atinge o mínimo
configurado (`configs/params.yaml`).

## API de inferência (serving)

```bash
poetry run uvicorn recsys.serving.api:app --reload
```

- `GET /health` — liveness check.
- `GET /recommend?user=<id>&k=<n>` — top-k recomendações para o usuário.

```json
{
  "user": 1, "k": 5,
  "items": [
    {"id": 0, "name": "Fone de Ouvido Bluetooth — Mod. 0000", "category": "Eletrônicos"}
  ]
}
```

`name`/`category` vêm de um **catálogo fictício de demonstração**
(`src/recsys/data/catalog.py`) — o RetailRocket não expõe nomes de produto
reais, só IDs e propriedades anonimizadas/hasheadas. `id` é sempre o item
real recomendado pelo modelo.

## Docker (ambiente local)

```bash
docker compose up --build     # sobe o servidor MLflow + roda o pipeline de treino
```

## Deploy em produção (AWS)

Bônus do enunciado: container acessível por URL pública. Implantado em
**AWS Lambda** (imagem de container, self-contained) + **Function URL**
pública — escolhido por ter free tier **perpétuo** (1M requisições/mês),
diferente de EC2/App Runner. Guia completo de deploy: [`deploy/README.md`](deploy/README.md).

```bash
curl "https://kordlmw2h4cfpovpex3gghwica0uzrnl.lambda-url.sa-east-1.on.aws/health"
curl "https://kordlmw2h4cfpovpex3gghwica0uzrnl.lambda-url.sa-east-1.on.aws/recommend?user=1&k=5"
```

**Custo:** invocações Lambda ficam no free tier perpétuo. O armazenamento da
imagem no ECR **não** é gratuito para sempre (só nos primeiros 12 meses de
conta nova) — custa ~US$0,10/GB-mês; vale limpar digests antigos de tempos em
tempos (`aws ecr batch-delete-image`).

## Qualidade

```bash
poetry run pytest -q                                          # 77 testes, cobertura >=92% (piso do CI)
poetry run ruff check . && poetry run ruff format --check .   # lint + formato
```

## Entregas (enunciado)
- ✅ Repositório GitHub (clean code, Poetry, Docker, DVC, MLflow).
- ⏳ **Vídeo de 5 min (método STAR)** — pendente.
- ✅ Bônus: deploy em nuvem — container acessível por URL pública (ver acima).
