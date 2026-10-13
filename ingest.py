import json
import urllib.request
import warnings

import chromadb
import torch
import transformers
from transformers import AutoModel

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

WIKI_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=Japan"
)
EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
CHROMA_PATH = "./chroma-data"
COLLECTION = "japan-demo"


def load_context():
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "verbatim-rag-demo/1.0"})
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
    pairs = chunk(load_context())
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )
    if collection.count() == len(pairs):
        print(f"Already ingested: {len(pairs)} chunks. Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(pairs)} chunks on {device}...")
    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True).to(device)
    # Embed heading + paragraph for retrieval; store the bare paragraph as the
    # document so highlight returns an exact span of the source text.
    embed_texts = [f"{h}\n{p}" if h else p for h, p in pairs]
    embeddings = embed.encode(texts=embed_texts, task="retrieval", prompt_name="document")
    collection.upsert(
        ids=[f"chunk-{i}" for i in range(len(pairs))],
        documents=[p for _, p in pairs],
        metadatas=[{"heading": h} for h, _ in pairs],
        embeddings=embeddings.tolist(),
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_PATH}.")


main()
