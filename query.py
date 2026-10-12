import os
import sys

import chromadb
import torch
from transformers import AutoModel

EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
RERANK_MODEL = "jinaai/jina-reranker-v3.5"
HIGHLIGHT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
CHROMA_PATH = "./chroma-data"
COLLECTION = "japan-demo"
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = 3

QUESTIONS = [
    "What is the capital of Japan?",
    "What is the currency of Japan?",
    "What are the warrior nobility of Japan called?",
    "What are the five main islands of Japan?",
    "What is the capital of France?",
]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print(f"Collection '{COLLECTION}' not found. Run: uv run ingest.py")
        return

    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    # ponytail: bf16 on CPU is ~3x slower here; use float32 unless a GPU is present.
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    reranker = AutoModel.from_pretrained(RERANK_MODEL, dtype=dtype, trust_remote_code=True)
    highlighter = AutoModel.from_pretrained(HIGHLIGHT_MODEL, trust_remote_code=True)

    for q in (sys.argv[1:] or QUESTIONS):
        print(f"Q: {q}")

        # Stage 1: retrieve candidates from the vector store.
        q_emb = embed.encode(texts=[q], task="retrieval", prompt_name="query")
        res = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=TOP_K_RETRIEVE,
            include=["documents", "distances"],
        )
        candidates = res["documents"][0]
        print(f"  retrieve: {len(candidates)} candidates, best dist={res['distances'][0][0]:.3f}")

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


main()
