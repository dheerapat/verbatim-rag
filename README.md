# verbatim-rag demo

<img width="744" height="642" alt="Recording 2026-10-13 112028" src="https://github.com/user-attachments/assets/8bdc9ffb-3734-43e4-aafa-4071b881f1a7" />

A small demo of **verbatim RAG**: retrieval that answers a question with an exact
quote from the source text.

A normal RAG system generates an answer. This demo returns the supporting span
itself. The user can read the source words and check the answer.

## Pipeline

1. **Ingest** (offline, one time): fetch each Wikipedia page, split it into chunks,
   attach each section heading to its paragraph, embed each chunk, store the
   vectors in LanceDB.
2. **Retrieve** (per query): embed the query, get the top-k chunks from LanceDB.
3. **Rerank**: score the candidates with a cross-encoder, keep the top 3.
4. **Highlight**: find the exact evidence span in the top chunks.

```
question
   |
   v
embed query --> LanceDB top-k --> rerank top-3 --> highlight --> exact span
```

## Models

| Stage | Model |
|-------|-------|
| Embed | `Alibaba-NLP/gte-modernbert-base` |
| Rerank | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Highlight | `KRLabsOrg/verbatim-rag-modern-bert-v2` |

## Setup

The project needs Python 3.14 or newer (see `.python-version`) and Node.js.

```bash
uv sync
cd frontend && npm install && cd ..
```

## Project structure

| Directory | Role |
|-----------|------|
| `core/` | The RAG logic. `ingest.py`, `rag.py`, `embed.py` and `vector_store.py`. No HTTP. |
| `backend/` | FastAPI app in `main.py`. Calls `core/`. |
| `frontend/` | React 19 + Vite + Astryx UI. Calls the API. |

## Usage

Ingest the default page set (9 country articles, 1,103 chunks) with a 1 s delay
between requests:

```bash
uv run python -m core.ingest
```

Add extra pages to the set:

```bash
uv run python -m core.ingest Rome Nairobi Switzerland
```

Ask the default questions:

```bash
uv run python -m core.rag
```

Ask one question:

```bash
uv run python -m core.rag "What is the currency of Japan?"
```

Change the number of retrieved chunks:

```bash
TOP_K_RETRIEVE=20 uv run python -m core.rag
```

## S3 storage with RustFS

The default local LanceDB URI keeps the demo self-contained. For an S3-backed table
using RustFS, copy `.env.example` to `.env`, replace both credentials with unique
values, and start RustFS:

```bash
cp .env.example .env
# Edit .env and set RUSTFS_ACCESS_KEY and RUSTFS_SECRET_KEY.
docker compose up -d
```

RustFS exposes its S3 API at <http://localhost:9100> and its Console at
<http://localhost:9101> by default. The `rustfs-create-bucket` one-shot service uses
AWS CLI to wait for RustFS and create the `verbatim-rag` bucket automatically; it
succeeds without visiting the Console and safely skips creation if the bucket exists.

If you copied an earlier `.env.example`, update the existing `.env` (copying the
template again will not overwrite it):

```dotenv
RUSTFS_S3_PORT=9100
RUSTFS_CONSOLE_PORT=9101
RUSTFS_BUCKET=verbatim-rag
LANCEDB_S3_ENDPOINT=http://localhost:9100
```

The host ports are configurable with
`RUSTFS_S3_PORT` and `RUSTFS_CONSOLE_PORT`; Compose maps them to RustFS's internal
ports 9000 and 9001. If a port is already allocated, choose unused host ports and
keep `LANCEDB_S3_ENDPOINT` in sync with the S3 host port. For example, use `9200` for
the S3 port and set `LANCEDB_S3_ENDPOINT=http://localhost:9200` in `.env`, then run
`docker compose up -d --force-recreate`.

Then ingest and query with the env file loaded into each command:

```bash
uv run --env-file .env python -m core.ingest
uv run --env-file .env python -m core.rag "What is the capital of Japan?"
uv run --env-file .env uvicorn backend.main:app --reload
```

Use the same `--env-file .env` option when starting the web app in production.

The example config points `LANCEDB_URI` to `s3://verbatim-rag/lance-data` and
`LANCEDB_S3_ENDPOINT` to the local RustFS service. Set the endpoint to the URL
reachable from the Python process; when the app runs in another container, use the
RustFS service name and container port instead of `localhost`. The Compose file binds
the RustFS ports to loopback only. Use strong secrets and TLS before making RustFS
reachable beyond a trusted local network. LanceDB's local S3-backed OSS client stores
the table in the bucket; the AWS CLI init service provisions the bucket.

After moving from the old Chroma store, run ingestion again to recreate the table.
The ignored `chroma-data/` directory is left untouched as a backup; this app no longer
reads it.

## Web UI

The UI is React 19 + Vite + Astryx. The server is FastAPI. It wraps the same
pipeline in `core/rag.py`.

Development (two terminals):

```bash
uv run uvicorn backend.main:app --reload
cd frontend && npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8000.

Production build (one server):

```bash
cd frontend && npm run build && cd ..
uv run uvicorn backend.main:app
```

Open <http://localhost:8000>. FastAPI serves `frontend/dist/` and `/api` together.

The page shows:

- the answer span and its score,
- the time for each stage (retrieve, rerank, highlight),
- the source paragraph with the span marked in place,
- the source title and section, with a link to Wikipedia.

When the highlighter finds no span, the page shows an empty state with the best
rerank score. When the request fails, the page shows the HTTP status.

The page cannot mark text on wikipedia.org. No link can highlight an arbitrary
span there. So the page renders the exact source paragraph from the corpus and
marks the span. The Wikipedia link opens the section for a second check.

Frontend files:

- `frontend/index.html`, `frontend/vite.config.ts`, `frontend/tsconfig.json` — Vite and TypeScript.
- `frontend/src/main.tsx` — mounts the app inside the Astryx `Theme`.
- `frontend/src/App.tsx` — the page. Uses Astryx `Layout`, `Heading`, `Text`, `TextInput`, `Button`, `Card`, `Citation` and `EmptyState`.
- `frontend/src/styles.css` — Astryx base CSS and the `.answer-mark` rule.
- `frontend/astryx-themes.js` — generated by `npx astryx theme add neutral`.

## Example

Corpus: 9 Wikipedia country articles, 1,103 chunks. Default `TOP_K_RETRIEVE=10`.

Questions in the corpus get an exact quote and a score. Questions outside the
corpus get a low rerank score and no span. Real output on CPU:

```
Q: What is the capital of Japan?
  retrieve: 10 candidates, best dist=0.186 [Japan] (229ms)
  rerank:   best=2.81 (2526ms)
  answer:   [0.92] Tokyo is the country's capital and largest city. (2288ms)

Q: What is the capital of France?
  retrieve: 10 candidates, best dist=0.176 [France] (148ms)
  rerank:   best=2.51 (2979ms)
  answer:   [0.85] Its capital, largest city and main cultural and economic centre is Paris, with a metropolitan population of over 13 million. (590ms)

Q: Who invented the telephone?
  retrieve: 10 candidates, best dist=0.395 [Brazil] (90ms)
  rerank:   best=1.47 (2555ms)
  answer:   (no answer found) (418ms)
```

The title in brackets after `best dist` shows the page of the best candidate.
The last question is not in the corpus. The highlighter returns no span.

The default `RERANK_THRESHOLD=2.0` rejects low-relevance candidates before
highlighting. Only candidates at or above that score reach the highlighter,
which still uses its 0.2 span threshold. Override the rerank cutoff when
calibrating against a different corpus, for example `RERANK_THRESHOLD=1.8`.

## Files

| File | Purpose |
|------|---------|
| `core/ingest.py` | Fetch pages, chunk, embed, and store in LanceDB. |
| `core/rag.py` | Retrieve from LanceDB, rerank, and highlight. The shared pipeline. |
| `core/embed.py` | Embedding helpers: CLS pooling and L2 normalization. |
| `core/vector_store.py` | Configure local or S3-compatible LanceDB connections. |
| `backend/main.py` | FastAPI server. Serves `/api` and the built UI. |
| `frontend/src/App.tsx` | React + Astryx page. |

By default the vector store is a local LanceDB table at `./lance-data` (gitignored),
so no server or Docker is needed. To use object storage instead, configure `LANCEDB_URI`
with an S3 URI and set the S3 credentials and endpoint; the RustFS setup below provides
an S3-compatible service for local development.

## Notes

- The embedded text is `title + heading + paragraph`. The stored document is the
  bare paragraph, so the highlight always returns an exact span of the source.
- Embedding runs in batches of 64. One call over the full corpus exceeds VRAM.
- Wikipedia serves one full-text extract per request, so ingest loops over the
  titles with a 1 s delay between requests (`FETCH_DELAY`). A shorter delay
  causes HTTP 429 errors. The script skips a failed page and fills it on the
  next run, because it embeds only chunks that are not stored yet.
- With a small corpus, near-miss distractors surface. "Most time zones" answers
  with Brazil, but France has more. More pages reduce this.
- `TOP_K_RETRIEVE=10` is the default. A small value is fast but can miss the
  correct chunk. The reranker cannot fix a miss in stage 1.
- On CPU the reranker is the slow part: about 2 to 4 s for 10 candidates. The
  highlighter takes about 0.4 to 2.6 s, and retrieval takes about 0.1 to 0.2 s.
  A GPU removes most of this cost. The first query of a run is slower; this is
  one-time CUDA warm-up.

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
