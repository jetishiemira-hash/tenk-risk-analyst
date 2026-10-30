"""Step 4b of the pipeline: semantic search inside BigQuery with VECTOR_SEARCH + metadata filters."""
from google.cloud import bigquery

from load_bigquery import PROJECT, TABLE
from search import embed_query

bq = bigquery.Client(project=PROJECT)


def retrieve(question: str, k: int = 5, ticker: str | None = None) -> list[dict]:
    """Return the k chunks closest in meaning to the question, optionally only from one company."""
    params = [bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embed_query(question).tolist())]
    where = ""
    if ticker:
        where = "WHERE ticker = @ticker"  # a hard filter: only this company's chunks are searched
        params.append(bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper()))

    sql = f"""
        SELECT base.chunk_id, base.ticker, base.company, base.filing_date,
               base.item, base.section_title, base.url, base.text, distance
        FROM VECTOR_SEARCH(
            (SELECT * FROM `{TABLE}` {where}),
            'embedding',
            (SELECT @query_embedding AS embedding),
            top_k => {int(k)},
            distance_type => 'COSINE'
        )
        ORDER BY distance
    """
    rows = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    # cosine distance = 1 - cosine similarity, so we convert back to the scores we're used to
    return [dict(row) | {"score": 1 - row["distance"], "filing_date": str(row["filing_date"])} for row in rows]


def parse_question(raw: str) -> tuple[str, str | None]:
    """'WMT: how do they hedge?' -> ('how do they hedge?', 'WMT'). No prefix -> search all companies."""
    prefix, sep, rest = raw.partition(":")
    if sep and prefix.strip().isalpha() and len(prefix.strip()) <= 5:
        return rest.strip(), prefix.strip().upper()
    return raw.strip(), None


def main():
    print("Ask a question. Start with a ticker to filter, e.g. 'WMT: interest rate risk'. Empty line quits.")
    while True:
        raw = input("\nQuestion: ").strip()
        if not raw:
            break
        question, ticker = parse_question(raw)
        for r in retrieve(question, ticker=ticker):
            print(f"\n{r['score']:.3f} | {r['company']} | {r['item']} {r['section_title']} | {r['chunk_id']}")
            print("   " + r["text"][:300] + "...")


if __name__ == "__main__":
    main()