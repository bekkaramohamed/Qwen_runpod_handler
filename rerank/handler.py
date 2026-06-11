# runpod_handlers/rerank/handler.py
"""Handler RunPod Serverless — reranking Qwen3-Reranker-4B (transformers).

Contrat (aligné sur src/infrastructure/ia_providers/runpod.py) :
  input  : {"query": "...", "documents": ["...", ...]}
  output : {"scores": [0.92, 0.13, ...], "status": "success"}

Qwen3-Reranker est un LLM causal utilisé en juge binaire : pour chaque paire
(query, doc) on lit P("yes") sur le dernier token — code du model card officiel.
transformers suffit pour le POC (≤20 docs/requête) ; passage vLLM si besoin de débit.
"""

import torch
import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-Reranker-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto"
).eval()

TOKEN_YES = tokenizer.convert_tokens_to_ids("yes")
TOKEN_NO = tokenizer.convert_tokens_to_ids("no")
MAX_LENGTH = 4096

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _format_pair(query: str, doc: str) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}{SUFFIX}"


@torch.no_grad()
def _scores(query: str, documents: list[str]) -> list[float]:
    pairs = [_format_pair(query, d) for d in documents]
    inputs = tokenizer(
        pairs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(model.device)
    logits = model(**inputs).logits[:, -1, :]
    yes_no = torch.stack([logits[:, TOKEN_NO], logits[:, TOKEN_YES]], dim=1)
    probs = torch.nn.functional.log_softmax(yes_no, dim=1)
    return probs[:, 1].exp().tolist()  # P("yes")


def handler(job):
    inputs = job["input"]
    query = inputs.get("query")
    documents = inputs.get("documents")
    if not query or not documents:
        return {"error": "champs 'query' et 'documents' requis", "status": "failed"}
    try:
        return {"scores": _scores(query, documents), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
