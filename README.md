# Handlers RunPod Serverless — MIIP

Deux endpoints GPU pour le RAG (mêmes contrats que `back_end/src/infrastructure/ia_providers/runpod.py`) :

| Dossier | Modèle | Input → Output | GPU conseillé |
|---|---|---|---|
| `embeddings/` | Qwen3-Embedding-4B (vLLM) | `{texts, is_query?}` → `{embeddings}` | 24 Go (L4 / A5000 / 4090) |
| `rerank/` | Qwen3-Reranker-4B (transformers) | `{query, documents}` → `{scores}` | 24 Go |

## Déploiement (par endpoint)
1. Pousser ce dossier dans un repo GitHub.
2. RunPod Console → Serverless → New Endpoint → **GitHub Repo** → pointer le sous-dossier (Dockerfile).
3. GPU 24 Go · Max workers: 2 · Idle timeout: 60 s (POC).
4. Récupérer l'endpoint ID → `Miip/.env` (`RUNPOD_EMBED_ENDPOINT_ID` / `RUNPOD_RERANK_ENDPOINT_ID`).

## Concurrence (POC)
Un job = un batch (32 textes pour embed, ≤20 docs pour rerank). RunPod met en file
et scale les workers (max 2). L'ingestion des 10 clients (~15-30 k chunks) ≈ 500-1000
jobs séquentiels → ~20-40 min avec 1 worker. Suffisant pour le POC ; en prod dédiée,
on passera sur un serving vLLM persistant (débit continu, pas de cold start).
