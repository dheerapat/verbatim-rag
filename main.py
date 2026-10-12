import json
import urllib.request

import torch
from transformers import AutoModel

WIKI_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=Japan"
)
EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
RERANK_MODEL = "jinaai/jina-reranker-v3.5"
HIGHLIGHT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 3


def load_context():
    # ponytail: fetched on every run; add a disk cache if the network is slow.
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "verbatim-rag-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"]


def chunk(text, min_chars=150):
    """Split the article on paragraph lines. Drop short headings."""
    return [line.strip() for line in text.split("\n") if len(line.strip()) >= min_chars]


def main():
    context = load_context()
    chunks = chunk(context)
    print(f"Context: Wikipedia 'Japan' -> {len(chunks)} chunks\n")

    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    # ponytail: bf16 on CPU is ~3x slower here; use float32 unless a GPU is present.
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    reranker = AutoModel.from_pretrained(RERANK_MODEL, dtype=dtype, trust_remote_code=True)
    highlighter = AutoModel.from_pretrained(HIGHLIGHT_MODEL, trust_remote_code=True)

    doc_embs = embed.encode(texts=chunks, task="retrieval", prompt_name="document")

    for q in QUESTIONS:
        print(f"Q: {q}")

        # Stage 1: retrieve candidates by embedding similarity.
        q_emb = embed.encode(texts=[q], task="retrieval", prompt_name="query")
        scores = (q_emb @ doc_embs.T).squeeze(0)
        top = torch.topk(scores, k=min(TOP_K_RETRIEVE, len(chunks))).indices.tolist()
        candidates = [chunks[i] for i in top]
        print(f"  retrieve: {len(candidates)} candidates, best cos={scores[top[0]]:.2f}")

        # Stage 2: rerank candidates with the cross-encoder.
        ranked = reranker.rerank(q, candidates, top_n=TOP_K_RERANK)
        print(f"  rerank:   best={ranked[0]['relevance_score']:.2f}")

        # Stage 3: highlight the exact evidence span in the top chunks.
        evidence = "\n".join(r["document"] for r in ranked)
        result = highlighter.process(question=q, context=evidence, threshold=0.2)
        if not result["spans"]:
            print("  answer:   (no answer found)\n")
            continue
        best = max(result["spans"], key=lambda s: s["score"])
        print(f"  answer:   [{best['score']:.2f}] {best['text'].strip()}\n")


QUESTIONS = [
    "What is the capital of Japan?",
    "What is the currency of Japan?",
    "What are the warrior nobility of Japan called?",
    "What are the five main islands of Japan?",
    "What is the capital of France?",
]

main()
