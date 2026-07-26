# syntax=docker/dockerfile:1

# --- Stage 1: builder (instala dependências num venv isolado) ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app
RUN pip install "poetry>=1.8"

# Camada de dependências (cacheável): instala deps antes de copiar o código,
# então mudanças no código não reinstalam o torch/mlflow/etc.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Camada de código: copia o pacote e instala a raiz (rápido, deps já prontas).
COPY src ./src
COPY configs ./configs
RUN poetry install --only main

# --- Stage 2: runtime (imagem enxuta, só o venv + código) ---
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Padrão: roda a avaliação do baseline. Sobrescreva o command no compose/CLI.
CMD ["python", "-m", "recsys.pipeline.evaluate"]
