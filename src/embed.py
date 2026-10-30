"""Step 3 of the pipeline: embed every chunk with Vertex AI and save the vectors."""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from google import genai
from google.genai import errors, types

PROJECT = "tenk-risk-analyst-emira"
LOCATION = "global"
MODEL = "gemini-embedding-2"
DIMENSIONS = 768
MAX_WORKERS = 4   # how many requests we send in parallel
MAX_RETRIES = 8

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CACHE_PATH = Path("data/processed/embeddings_cache.jsonl")  # one line per finished chunk
VECTORS_PATH = Path("data/processed/vectors.npy")
IDS_PATH = Path("data/processed/vector_ids.json")

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def document_prompt(chunk: dict) -> str:
    """gemini-embedding-2's document format. The title tells the model where the text came from."""
    title = f"{chunk['company']} 10-K ({chunk['filing_date'][:4]}), {chunk['item']}: {chunk['section_title']}"
    return f"title: {title} | text: {chunk['text']}"


def embed_one(text: str) -> list[float]:
    """Embed one text. If Google says 'slow down' (429) or has a hiccup (5xx), wait and retry."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
            )
            return response.embeddings[0].values
        except (errors.APIError, httpx.TransportError) as e:
            code = getattr(e, "code", None)  # network errors have no HTTP code
            retryable = code is None or code == 429 or code >= 500
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt + random.random()  # exponential backoff: ~1s, 2s, 4s, 8s... plus jitter
            print(f"  {code or type(e).__name__} error, retrying in {wait:.1f}s")
            time.sleep(wait)


def load_cache() -> dict:
    """Chunks embedded in an earlier run, so a crash or rerun never pays for the same chunk twice."""
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {r["chunk_id"]: r["vector"] for r in rows}


def main():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    cache = load_cache()
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"{len(chunks):,} chunks total, {len(cache):,} already embedded, {len(todo):,} to go")

    start = time.time()
    with CACHE_PATH.open("a", encoding="utf-8") as cache_file, ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(embed_one, document_prompt(c)): c["chunk_id"] for c in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            chunk_id = futures[future]
            vector = future.result()
            cache[chunk_id] = vector
            cache_file.write(json.dumps({"chunk_id": chunk_id, "vector": vector}) + "\n")
            if done % 100 == 0 or done == len(todo):
                print(f"  {done:,}/{len(todo):,} embedded ({time.time() - start:.0f}s)")

    # Save as one matrix: row i of vectors.npy belongs to vector_ids[i]
    ids = [c["chunk_id"] for c in chunks]
    vectors = np.array([cache[i] for i in ids], dtype=np.float32)
    np.save(VECTORS_PATH, vectors)
    IDS_PATH.write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved {vectors.shape[0]:,} vectors of size {vectors.shape[1]} to {VECTORS_PATH}")


if __name__ == "__main__":
    main()