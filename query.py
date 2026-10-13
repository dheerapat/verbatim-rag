import os
import sys
import time
import warnings

import chromadb
import torch
import transformers
from transformers import AutoModel

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
RERANK_MODEL = "jinaai/jina-reranker-v3.5"
HIGHLIGHT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
CHROMA_PATH = "./chroma-data"
COLLECTION = "wiki-multi"
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = 3

QUESTIONS = [
    "What is the capital of Japan?",
    "What is the currency of Japan?",
    "What are the warrior nobility of Japan called?",
    "What are the five main islands of Japan?",
    "What is the capital of France?",
    "What is the longest river in Egypt?",
    "Who wrote the play 'Romeo and Juliet'?",
    "What is the official language of Brazil?",
    "Which country has the most time zones?",
    "What is the Holocaust memorial in Berlin called?",
    "Who invented the telephone?",
    "What year did the Berlin Wall fall?",
    "What is the tallest mountain in the world?",
    "What did Einstein win a Nobel Prize for?",
    "What is the Bodhisattva of compassion called in Japanese Buddhism?",
]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print(f"Collection '{COLLECTION}' not found. Run: uv run ingest.py")
        return

    if torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    else:
        dtype, device = torch.float32, "cpu"
    print(f"Models on: {device}\n")
    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True).to(device)
    reranker = AutoModel.from_pretrained(
        RERANK_MODEL, dtype=dtype, device_map=device, trust_remote_code=True
    )
    highlighter = AutoModel.from_pretrained(
        HIGHLIGHT_MODEL, device_map=device, trust_remote_code=True
    )

    for q in sys.argv[1:] or QUESTIONS:
        print(f"Q: {q}")

        # Stage 1: retrieve candidates from the vector store.
        t0 = time.perf_counter()
        q_emb = embed.encode(texts=[q], task="retrieval", prompt_name="query")
        res = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=TOP_K_RETRIEVE,
            include=["documents", "distances", "metadatas"],
        )
        documents = res["documents"]
        distances = res["distances"]
        metadatas = res.get("metadatas")
        assert documents is not None and distances is not None
        candidates = documents[0]
        best_title = metadatas[0][0]["title"] if metadatas else "?"
        dt = (time.perf_counter() - t0) * 1000
        print(
            f"  retrieve: {len(candidates)} candidates, best dist={distances[0][0]:.3f}"
            f" [{best_title}] ({dt:.0f}ms)"
        )

        # Stage 2: rerank candidates with the cross-encoder.
        t0 = time.perf_counter()
        ranked = reranker.rerank(q, candidates, top_n=TOP_K_RERANK)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  rerank:   best={ranked[0]['relevance_score']:.2f} ({dt:.0f}ms)")

        # Stage 3: highlight the exact evidence span in the top chunks.
        t0 = time.perf_counter()
        evidence = "\n".join(r["document"] for r in ranked)
        result = highlighter.process(question=q, context=evidence, threshold=0.2)
        dt = (time.perf_counter() - t0) * 1000
        if not result["spans"]:
            print(f"  answer:   (no answer found) ({dt:.0f}ms)\n")
            continue
        best = max(result["spans"], key=lambda s: s["score"])
        print(
            f"  answer:   [{best['score']:.2f}] {best['text'].strip()} ({dt:.0f}ms)\n"
        )


main()
