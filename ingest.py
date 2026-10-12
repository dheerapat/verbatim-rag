import json
import urllib.request

import chromadb
from transformers import AutoModel

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
    """Split the article on paragraph lines. Drop short headings."""
    return [line.strip() for line in text.split("\n") if len(line.strip()) >= min_chars]


def main():
    chunks = chunk(load_context())
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )
    if collection.count() >= len(chunks):
        print(f"Already ingested: {collection.count()} chunks. Nothing to do.")
        return

    print(f"Embedding {len(chunks)} chunks...")
    embed = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    embeddings = embed.encode(texts=chunks, task="retrieval", prompt_name="document")
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
    )
    print(f"Stored {collection.count()} chunks in {CHROMA_PATH}.")


main()
