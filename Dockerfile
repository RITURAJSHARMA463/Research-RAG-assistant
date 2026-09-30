# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# REQS selects the dependency profile:
#   requirements.txt        -> full fidelity (local embeddings + cross-encoder, ~2GB RAM)
#   requirements-cloud.txt   -> lightweight (API embeddings, no torch, fits ~512MB-1GB)
ARG REQS=requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --upgrade pip && pip install -r ${REQS}

COPY . .

# OPTIONAL (full-fidelity profile only): pre-download the HF models at build time so
# the first request doesn't pay a ~1GB download. Uncomment if using requirements.txt.
# RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')" && \
#     python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

EXPOSE 8000

# Render/Railway/Fly inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
