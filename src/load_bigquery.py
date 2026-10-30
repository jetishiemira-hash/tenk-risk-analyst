"""Step 4a of the pipeline: load chunks + their vectors into BigQuery, our vector database."""
import json

import numpy as np
from google.cloud import bigquery

from embed import CHUNKS_PATH, IDS_PATH, PROJECT, VECTORS_PATH

DATASET = "tenk"
TABLE = f"{PROJECT}.{DATASET}.chunks"

# Explicit schema: structured metadata columns + the text + the embedding vector
SCHEMA = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING"),
    bigquery.SchemaField("company", "STRING"),
    bigquery.SchemaField("filing_date", "DATE"),
    bigquery.SchemaField("item", "STRING"),
    bigquery.SchemaField("section_title", "STRING"),
    bigquery.SchemaField("url", "STRING"),
    bigquery.SchemaField("text", "STRING"),
    bigquery.SchemaField("n_words", "INT64"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),  # REPEATED = an array of numbers
]


def main():
    bq = bigquery.Client(project=PROJECT)

    dataset = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset.location = "US"
    bq.create_dataset(dataset, exists_ok=True)

    vectors = np.load(VECTORS_PATH)
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        by_id = {c["chunk_id"]: c for c in map(json.loads, f)}

    rows = []
    for chunk_id, vector in zip(ids, vectors):
        c = by_id[chunk_id]
        rows.append({
            "chunk_id": chunk_id,
            "ticker": c["ticker"],
            "company": c["company"],
            "filing_date": c["filing_date"],
            "item": c["item"],
            "section_title": c["section_title"],
            "url": c["url"],
            "text": c["text"],
            "n_words": c["n_words"],
            "embedding": [float(x) for x in vector],
        })

    # WRITE_TRUNCATE replaces the table each run, so re-running never creates duplicates
    job_config = bigquery.LoadJobConfig(schema=SCHEMA, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    bq.load_table_from_json(rows, TABLE, job_config=job_config).result()  # .result() waits for the job

    table = bq.get_table(TABLE)
    print(f"Loaded {table.num_rows:,} rows into {TABLE}")


if __name__ == "__main__":
    main()