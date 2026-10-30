"""Step 4 of the pipeline: semantic search over all chunks (brute force with numpy)."""
import json

import numpy as np
from google.genai import types

# Reuse settings from embed.py instead of copying them. This works because of the
# `if __name__ == "__main__":` guard: importing embed.py does NOT start re-embedding.
from embed import CHUNKS_PATH, DIMENSIONS, IDS_PATH, MODEL, VECTORS_PATH, client

TOP_K = 5


def load_index():
    """Load the vectors plus the chunk (text + metadata) that belongs to each row."""
    vectors = np.load(VECTORS_PATH)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # make sure every vector has length 1
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        by_id = {c["chunk_id"]: c for c in map(json.loads, f)}
    return vectors, [by_id[i] for i in ids]


def embed_query(question: str) -> np.ndarray:
    """Queries use the 'task: ... | query: ...' format, documents used 'title: ... | text: ...'."""
    response = client.models.embed_content(
        model=MODEL,
        contents=f"task: search result | query: {question}",
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
    )
    vector = np.array(response.embeddings[0].values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def search(question: str, vectors: np.ndarray, chunks: list, k: int = TOP_K):
    """Score every chunk against the question and return the k best."""
    scores = vectors @ embed_query(question)  # one cosine similarity per chunk
    best = np.argsort(-scores)[:k]            # indices of the highest scores
    return [(float(scores[i]), chunks[i]) for i in best]


def main():
    vectors, chunks = load_index()
    print(f"Loaded {len(chunks):,} chunks. Ask a question (press Enter on an empty line to quit).")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            break
        for score, c in search(question, vectors, chunks):
            print(f"\n{score:.3f} | {c['company']} | {c['item']} {c['section_title']} | {c['chunk_id']}")
            print("   " + c["text"][:300] + "...")


if __name__ == "__main__":
    main()