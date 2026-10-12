# verbatim-rag demo

A small demo of **verbatim RAG**: retrieval that answers a question with an exact
quote from the source text.

A normal RAG system generates an answer. This demo returns the supporting span
itself. The user can read the source words and check the answer.

## Pipeline

1. **Ingest** (offline, one time): split the article into chunks, embed each chunk,
   store the vectors in Chroma.
2. **Retrieve** (per query): embed the query, get the top-k chunks from Chroma.
3. **Rerank**: score the candidates with a cross-encoder, keep the top 3.
4. **Highlight**: find the exact evidence span in the top chunks.

```
question
   |
   v
embed query --> Chroma top-k --> rerank top-3 --> highlight --> exact span
```

## Models

| Stage | Model |
|-------|-------|
| Embed | `jinaai/jina-embeddings-v5-text-nano` |
| Rerank | `jinaai/jina-reranker-v3.5` |
| Highlight | `KRLabsOrg/verbatim-rag-modern-bert-v2` |

## Setup

```bash
uv sync
```

## Usage

Ingest the article one time:

```bash
uv run ingest.py
```

Ask the default questions:

```bash
uv run query.py
```

Ask one question:

```bash
uv run query.py "What is the currency of Japan?"
```

Change the number of retrieved chunks:

```bash
TOP_K_RETRIEVE=3 uv run query.py
```

## Example

```
Q: What is the capital of Japan?
  retrieve: 10 candidates, best dist=0.262
  rerank:   best=0.48
  answer:   [0.92] Tokyo is the country's capital and largest city.

Q: What are the five main islands of Japan?
  retrieve: 10 candidates, best dist=0.222
  rerank:   best=0.69
  answer:   [0.99] The country's five main islands, from north to south, are Hokkaido, Honshu, Shikoku, Kyushu and Okinawa.

Q: What is the capital of France?
  retrieve: 10 candidates, best dist=0.687
  rerank:   best=-0.06
  answer:   (no answer found)
```

The last question is not in the context. The highlighter returns no span.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Fetch, chunk, embed, and store the context. |
| `query.py` | Retrieve, rerank, and highlight for each question. |
| `main.py` | Old single-file version. Kept for reference. |

The vector store is a local Chroma `PersistentClient` at `./chroma-data`
(gitignored). No server and no Docker are needed.

## Notes

- The context is the Wikipedia article "Japan": about 52,000 chars and 88 chunks.
- On CPU the reranker is the slow part. One query is about 12s. A GPU removes this cost.
- `TOP_K_RETRIEVE=10` is the default. A small value is fast but can miss the
  correct chunk. The reranker cannot fix a miss in stage 1.

## Other use cases

Verbatim RAG fits every task where the user must see the source words.

1. **Legal contract review.** Answer "What is the termination notice period?" and
   highlight the exact clause. No paraphrase can change a legal term.
2. **Clinical guidelines.** Answer "What is the first-line treatment?" and quote the
   exact recommendation line. A clinician can verify it at once.
3. **Financial filings.** Answer "What was the 2024 revenue?" and highlight the exact
   figure in the 10-K. The reader sees the number in its original sentence.
4. **Technical support.** Answer "How do I reset the device?" and quote the exact step
   in the manual. The user reads the real instruction.
5. **Grounded LLM answers.** Put the highlighted span next to a generated answer.
   The user checks the claim against the source. This reduces hallucination.
