FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv venv && uv pip install -r pyproject.toml

# Pré-télécharger le modèle dans l'image (cold start réduit de ~2 min → ~20 s)
RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-Reranker-4B')"

COPY handler.py .

CMD [".venv/bin/python", "handler.py"]
