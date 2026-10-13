"""Verbatim RAG core: retrieve, rerank, and highlight exact evidence spans.

CLI: uv run python -m core.rag "What is the capital of Japan?"
"""

import os
import sys
import time
import warnings
from pathlib import Path

import chromadb
import torch
import transformers
from transformers import AutoModel

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
RERANK_MODEL = "jinaai/jina-reranker-v3.5"
HIGHLIGHT_MODEL = "KRLabsOrg/verbatim-rag-modern-bert-v2"
# Repo-root chroma-data, independent of the working directory.
CHROMA_PATH = os.environ.get(
    "CHROMA_PATH", str(Path(__file__).resolve().parents[1] / "chroma-data")
)
COLLECTION = "wiki-multi"
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK = 3
THRESHOLD = 0.2

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


def load():
    """Load the collection and all three models once. Returns None if no data."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - a missing collection is expected
        print(f"Collection '{COLLECTION}' not found. Run: uv run python -m core.ingest")
        return None

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
    return {
        "embed": embed,
        "reranker": reranker,
        "highlighter": highlighter,
        "collection": collection,
    }


def answer(question, models):
    """Run retrieve -> rerank -> highlight. Returns a dict with the exact span.

    The span offsets are relative to the source paragraph, so a UI can mark the
    answer inside the original text.
    """
    collection = models["collection"]
    t0 = time.perf_counter()
    q_emb = models["embed"].encode(
        texts=[question], task="retrieval", prompt_name="query"
    )
    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=TOP_K_RETRIEVE,
        include=["documents", "distances", "metadatas"],
    )
    candidates = res["documents"][0]
    distances = res["distances"][0]
    metadatas = res["metadatas"][0]
    retrieve_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ranked = models["reranker"].rerank(question, candidates, top_n=TOP_K_RERANK)
    rerank_ms = (time.perf_counter() - t0) * 1000

    # Join the top chunks for the highlighter, but remember where each chunk
    # starts so a span can be mapped back to one source paragraph.
    parts, offsets = [], []
    pos = 0
    for r in ranked:
        doc = r["document"]
        offsets.append((pos, pos + len(doc), r["index"]))
        parts.append(doc)
        pos += len(doc) + 1  # newline separator
    evidence = "\n".join(parts)

    t0 = time.perf_counter()
    result = models["highlighter"].process(
        question=question, context=evidence, threshold=THRESHOLD
    )
    highlight_ms = (time.perf_counter() - t0) * 1000

    out = {
        "question": question,
        "answer": None,
        "score": None,
        "title": metadatas[0]["title"],
        "heading": metadatas[0]["heading"],
        "paragraph": candidates[0],
        "start": None,
        "end": None,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "highlight_ms": highlight_ms,
        "best_distance": float(distances[0]),
        "best_rerank": float(ranked[0]["relevance_score"]),
    }
    if not result["spans"]:
        return out

    best = max(result["spans"], key=lambda s: s["score"])
    for start, end, idx in offsets:
        if start <= best["start"] < end:
            meta = metadatas[idx]
            paragraph = candidates[idx]
            # Clamp a span that crosses a chunk boundary to this paragraph.
            span_start = best["start"] - start
            span_end = min(best["end"], end) - start
            # Trim surrounding whitespace so the offsets match the answer text.
            while span_start < span_end and paragraph[span_start].isspace():
                span_start += 1
            while span_end > span_start and paragraph[span_end - 1].isspace():
                span_end -= 1
            out.update(
                answer=paragraph[span_start:span_end],
                score=float(best["score"]),
                title=meta["title"],
                heading=meta["heading"],
                paragraph=paragraph,
                start=span_start,
                end=span_end,
            )
            break
    return out


def main(models=None):
    models = models or load()
    if models is None:
        return
    for q in sys.argv[1:] or QUESTIONS:
        r = answer(q, models)
        print(f"Q: {r['question']}")
        print(
            f"  retrieve: {TOP_K_RETRIEVE} candidates, "
            f"best dist={r['best_distance']:.3f} [{r['title']}] "
            f"({r['retrieve_ms']:.0f}ms)"
        )
        print(f"  rerank:   best={r['best_rerank']:.2f} ({r['rerank_ms']:.0f}ms)")
        if r["answer"] is None:
            print(f"  answer:   (no answer found) ({r['highlight_ms']:.0f}ms)\n")
        else:
            print(
                f"  answer:   [{r['score']:.2f}] {r['answer']} ({r['highlight_ms']:.0f}ms)\n"
            )


if __name__ == "__main__":
    main()
