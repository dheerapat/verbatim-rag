"""Ingest Wikipedia pages into the Chroma store.

CLI: uv run python -m core.ingest [ExtraTitle ...]
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import torch
import chromadb
import transformers

from core.embed import encode, load_embed_model

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

WIKI_TITLES = [
    "Japan",
    "France",
    "Brazil",
    "Egypt",
    "India",
    "China",
    "Germany",
    "Australia",
    "Canada",
    "Kenya",
    "Peru",
    "Indonesia",
    "Italy",
    "Mexico",
    "Nigeria",
    "Poland",
    "South Korea",
    "Spain",
    "Sweden",
    "Thailand",
    "Turkey",
    "Vietnam",
    "Argentina",
    "Morocco",
    "Greece",
    "Portugal",
]
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"
CHROMA_PATH = os.environ.get(
    "CHROMA_PATH", str(Path(__file__).resolve().parents[1] / "chroma-data")
)
COLLECTION = "wiki-multi"
FETCH_DELAY = 1.0  # seconds between Wikipedia requests (lower = HTTP 429)


def fetch_page(title):
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&prop=extracts&explaintext=1&format=json&redirects=1"
        f"&titles={urllib.parse.quote(title)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "verbatim-rag-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return next(iter(data["query"]["pages"].values()))["extract"]


def chunk(text, min_chars=150):
    """Split the article on paragraph lines.

    Short lines are section headings. Each heading is attached to the next
    paragraph so retrieval sees the section context. Returns (heading, paragraph)
    pairs; the stored document stays the exact paragraph for verbatim highlight.
    """
    out = []
    heading = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) < min_chars:
            heading = line
        else:
            out.append((heading, line))
            heading = ""
    return out


def main():
    # Wikipedia serves one full-text extract per request, so we loop.
    # Extra titles on the command line: uv run python -m core.ingest Corsica Rome
    titles = WIKI_TITLES + [t for t in sys.argv[1:] if t not in WIKI_TITLES]

    pages = []
    for i, title in enumerate(titles, 1):
        try:
            extract = fetch_page(title)
        except Exception as e:  # noqa: BLE001 - a failed fetch skips the page
            print(f"[{i}/{len(titles)}] {title}: FAILED ({e})")
            continue
        pairs = chunk(extract)
        pages.append((title, pairs))
        print(f"[{i}/{len(titles)}] {title}: {len(pairs)} chunks")
        time.sleep(FETCH_DELAY)

    triples = [(title, h, p) for title, ps in pages for h, p in ps]
    if not triples:
        print("No chunks. Check network/pages.")
        return

    print(f"\nIngesting {len(triples)} chunks from {len(pages)} pages...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": EMBED_MODEL},
    )
    # The stored embeddings are only valid for the model that produced them.
    stored_model = (collection.metadata or {}).get("embed_model")
    if stored_model != EMBED_MODEL:
        print(f"Embedding model changed; re-embedding {collection.count()} chunks.")
        client.delete_collection(COLLECTION)
        collection = client.get_or_create_collection(
            COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embed_model": EMBED_MODEL},
        )
    # Ids must not depend on which pages succeeded in one run: use the
    # per-page chunk position, not the global triple index.
    ids = [f"{title}-{j}" for title, pairs in pages for j in range(len(pairs))]
    # Incremental: only embed chunks this run has not stored yet, so a re-run
    # after Wikipedia 429s fills the gaps without re-embedding everything.
    have = set(collection.get(ids=ids, limit=len(ids) + 1)["ids"] if ids else [])
    todo = [
        (i, t)
        for i, (t, cid) in enumerate(zip(triples, ids))
        if cid not in have
    ]
    if not todo:
        print(f"Already ingested: {collection.count()} chunks. Nothing to do.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(todo)}/{len(triples)} chunks on {device}...")
    embed = load_embed_model(EMBED_MODEL, device=device)
    # Embed title + heading + paragraph for retrieval; store the bare paragraph
    # as the document so highlight returns an exact span of the source text.
    embed_texts = [
        f"{title}\n{h}\n{p}" if h else f"{title}\n{p}" for _, (title, h, p) in todo
    ]
    # Encode in batches; a single call over hundreds of chunks exceeds VRAM.
    embeddings = []
    for start in range(0, len(embed_texts), 64):
        batch = embed_texts[start : start + 64]
        embeddings.extend(encode(embed, batch))
        print(f"  embedded {min(start + 64, len(embed_texts))}/{len(embed_texts)}")
    collection.upsert(
        ids=[ids[i] for i, _ in todo],
        documents=[p for _, (_, _, p) in todo],
        metadatas=[{"title": t, "heading": h} for _, (t, h, _) in todo],
        embeddings=embeddings,
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
