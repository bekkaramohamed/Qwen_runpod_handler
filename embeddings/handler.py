# runpod_handlers/embeddings/handler.py
"""Handler RunPod Serverless — embeddings denses Qwen3-Embedding-4B via vLLM.

Contrat (aligné sur src/infrastructure/ia_providers/runpod.py) :
  input  : {"texts": ["...", ...], "is_query": false}
  output : {"embeddings": [[...2560 floats...], ...], "status": "success"}

`is_query=true` ajoute l'instruction de recherche recommandée par Qwen pour les
REQUÊTES (les documents s'encodent bruts) — gain de pertinence ~1-3 %.
"""

import runpod
from vllm import LLM

# Chargé une seule fois au démarrage du worker (cold start ~1-2 min)
llm = LLM(
    model="Qwen/Qwen3-Embedding-4B",
    task="embed",
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def handler(job):
    inputs = job["input"]
    texts = inputs.get("texts")
    if not texts or not isinstance(texts, list):
        return {"error": "champ 'texts' (liste) manquant", "status": "failed"}

    try:
        if inputs.get("is_query"):
            texts = [QUERY_INSTRUCTION + t for t in texts]
        outputs = llm.embed(texts)
        return {
            "embeddings": [o.outputs.embedding for o in outputs],
            "status": "success",
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
