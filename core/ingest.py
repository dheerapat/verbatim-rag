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

import chromadb
import torch
import transformers
from transformers import AutoModel

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
EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
CHROMA_PATH = os.environ.get(
    "CHROMA_PATH", str(Path(__file__).resolve().parents[1] / "chroma-data")
)
COLLECTION = "wiki-multi"
FETCH_DELAY = 0.25  # seconds between Wikipedia requests


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
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )
    if collection.count() == len(triples):
        print(f"Already ingested: {len(triples)} chunks. Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(triples)} chunks on {device}...")
    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True).to(device)
    # Embed title + heading + paragraph for retrieval; store the bare paragraph
    # as the document so highlight returns an exact span of the source text.
    embed_texts = [
        f"{title}\n{h}\n{p}" if h else f"{title}\n{p}" for title, h, p in triples
    ]
    # Encode in batches; a single call over hundreds of chunks exceeds VRAM.
    embeddings = []
    for start in range(0, len(embed_texts), 64):
        batch = embed_texts[start : start + 64]
        embeddings.extend(
            embed.encode(texts=batch, task="retrieval", prompt_name="document")
        )
        print(f"  embedded {min(start + 64, len(embed_texts))}/{len(embed_texts)}")
    collection.upsert(
        ids=[f"{title}-{i}" for i, (title, _, _) in enumerate(triples)],
        documents=[p for _, _, p in triples],
        metadatas=[{"title": t, "heading": h} for t, h, _ in triples],
        embeddings=[e.tolist() for e in embeddings],
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
