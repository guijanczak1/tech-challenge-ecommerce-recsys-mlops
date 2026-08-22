# Model Card — Recomendador de E-commerce (recsys-mlp)

## Visão geral
- **Modelo:** `recsys-mlp` — rede neural de embeddings de usuário e item seguida
  de um MLP, treinada em PyTorch para recomendação por feedback implícito.
- **Tarefa:** recomendar top-k itens para um usuário a partir do histórico de
  navegação (dataset RetailRocket: eventos `view`/`addtocart`/`transaction`).
- **Registro:** MLflow Model Registry como `recsys-mlp`, promovido de
  `staging` para `production` quando atinge o NDCG@k mínimo configurado.

## Arquitetura
- Embeddings de usuário e item (`embedding_dim`), concatenados e processados por
  um MLP (`hidden_dims`, `dropout`) que produz um score por par (usuário, item).
- Treino com `BCEWithLogitsLoss` sobre positivos observados + negativos
  amostrados uniformemente (`negatives_per_positive`), `Adam`, early stopping
  por paciência e checkpoint do melhor modelo.

## Dados de treino
- RetailRocket (real) quando presente em `data/raw`; caso contrário, amostra
  **sintética reprodutível** com o mesmo schema (seed fixa) para permitir
  `dvc repro` offline.
- Pré-processamento: filtro de interações mínimas + label encoding de IDs;
  split treino/teste **temporal** (eventos mais recentes viram teste).

## Métricas (avaliação top-k, k configurável)
Comparação com baselines (valores da amostra sintética; variam com o dataset real):

| Modelo | Precision@k | Recall@k | NDCG@k | MAP@k |
|---|---|---|---|---|
| MLP (PyTorch) | 0.121 | 0.408 | 0.347 | 0.241 |
| Popularidade | 0.121 | 0.411 | 0.348 | 0.241 |
| SVD (sklearn) | 0.074 | 0.252 | 0.212 | 0.137 |

> Na amostra **sintética** (popularidade Zipf), o baseline de popularidade é
> muito forte e o MLP praticamente empata. Em dados reais do RetailRocket
> espera-se vantagem do modelo neural pela personalização. Reproduza com
> `dvc repro` e inspecione no MLflow.

## Serving (API)
- `GET /recommend` retorna cada item como `{id, name, category}`. **`name` e
  `category` vêm de um catálogo fictício de demonstração**
  (`src/recsys/data/catalog.py`), determinístico por `item_id` — o
  RetailRocket (dataset real) não expõe nomes de produto, apenas IDs e
  propriedades hasheadas por anonimização. Não representam produtos reais.

## Uso pretendido
- Gerar listas de recomendação por usuário em contexto de e-commerce.
- **Não** indicado para decisões sensíveis (crédito, etc.) nem como única fonte
  de ranqueamento sem testes A/B.

## Limitações e vieses
- **Viés de popularidade:** feedback implícito reforça itens já populares;
  itens novos/nicho (cold start) são sub-representados.
- **Cold start:** usuários/itens não vistos no treino não têm embedding.
- **Negativos amostrados:** podem incluir falsos negativos (item relevante
  tratado como negativo), adicionando ruído.
- **Dados sintéticos** no fallback não refletem padrões reais de compra.

## Reprodutibilidade
- Seeds fixas (`config.seed`), `poetry.lock`, pipeline DVC versionado e
  experimentos rastreados no MLflow. Ver `README.md`.
