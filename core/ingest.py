"""Ingest Wikipedia pages into the LanceDB vector store.

CLI: uv run python -m core.ingest [ExtraTitle ...]
"""

import json
import sys
import time
import urllib.parse
import urllib.request
import warnings

import torch
import transformers
from lancedb.index import BTree

from core.embed import encode, load_embed_model
from core.vector_store import connect_database

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
]
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"
TABLE_NAME = "wiki-multi"
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
    db = connect_database()
    table = db.open_table(TABLE_NAME) if TABLE_NAME in db.list_tables().tables else None

    # Stored vectors are only valid for the model that produced them. A changed
    # model requires rebuilding this local LanceDB OSS table.
    if table is not None:
        sample = table.search().select(["embed_model"]).limit(1).to_list()
        stored_model = sample[0]["embed_model"] if sample else EMBED_MODEL
        if stored_model != EMBED_MODEL:
            print(f"Embedding model changed; rebuilding {table.count_rows()} chunks.")
            db.drop_table(TABLE_NAME)
            table = None

    # IDs are based on each page's chunk position, independent of fetch success.
    ids = [f"{title}-{j}" for title, pairs in pages for j in range(len(pairs))]
    # Only query the IDs in this run. The result is explicitly bounded.
    if table is None:
        have = set()
    elif ids:
        quoted_ids = ", ".join("'" + value.replace("'", "''") + "'" for value in ids)
        found = (
            table.search()
            .where(f"id IN ({quoted_ids})")
            .select(["id"])
            .limit(len(ids))
            .to_list()
        )
        have = {row["id"] for row in found}
    else:
        have = set()

    todo = [(i, t) for i, (t, cid) in enumerate(zip(triples, ids)) if cid not in have]
    if not todo:
        count = table.count_rows() if table is not None else 0
        print(f"Already ingested: {count} chunks. Nothing to do.")
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
    rows = [
        {
            "id": ids[index],
            "vector": vector,
            "document": paragraph,
            "title": title,
            "heading": heading,
            "embed_model": EMBED_MODEL,
        }
        for (index, (title, heading, paragraph)), vector in zip(todo, embeddings)
    ]
    if table is None:
        table = db.create_table(TABLE_NAME, data=rows)
        table.create_index("id", config=BTree())
    else:
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
    table.optimize()
    print(f"Stored {table.count_rows()} chunks in {db.uri}.")


if __name__ == "__main__":
    main()
