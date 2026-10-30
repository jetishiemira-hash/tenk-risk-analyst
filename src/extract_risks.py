"""Extract a structured risk table from every company's Risk Factors:
company | risk cluster | risk | evidence quote | source chunk, with every quote checked against the filing."""
import re
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from google.cloud import bigquery
from google.genai import types
from pydantic import BaseModel, Field

from answer import MODEL, generate_with_retry
from load_bigquery import PROJECT, TABLE

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "JNJ", "XOM", "WMT", "TSLA", "NKE"]
CHUNKS_PER_CALL = 6  # how many excerpts the model reads per request
OUT_DIR = Path("outputs")

# A fixed taxonomy: the model must pick one of these, so results are comparable across companies
RiskCluster = Literal[
    "Supply chain & operations",
    "Regulatory & legal",
    "Litigation",
    "Cybersecurity & data privacy",
    "Macroeconomic",
    "Geopolitical & trade",
    "Market & financial (rates, FX, credit)",
    "Competition",
    "Technology & AI",
    "Climate & environmental",
    "People & talent",
    "Reputation & brand",
    "Other",
]


# Pydantic models define the exact JSON shape Gemini must return (structured output)
class RiskSignal(BaseModel):
    cluster: RiskCluster
    risk: str = Field(description="One-sentence plain-English summary of this specific risk")
    evidence_quote: str = Field(description="ONE sentence copied word for word from the excerpt")
    chunk_id: str = Field(description="The id of the excerpt the quote was copied from")


class Extraction(BaseModel):
    risks: list[RiskSignal]


PROMPT = """Below are excerpts from the Risk Factors section of {company}'s 10-K, each starting with its [id].
List the distinct risks they disclose. For each risk:
- choose exactly one cluster,
- summarize the risk in one plain-English sentence,
- copy ONE supporting sentence word for word as evidence, and give the id of the excerpt it came from.
Skip boilerplate such as forward-looking-statement disclaimers. Only include risks stated in the text.

{excerpts}"""


def normalize(text: str) -> str:
    """Make quote matching robust to curly quotes, extra spaces and capitalization."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def fetch_risk_chunks(bq: bigquery.Client, ticker: str) -> list[dict]:
    """Get ALL Risk Factors chunks for a company. No vector search: extraction needs full coverage."""
    sql = f"""
        SELECT chunk_id, company, text FROM `{TABLE}`
        WHERE ticker = @ticker AND item = 'Item 1A'
        ORDER BY chunk_id
    """
    config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
    return [dict(row) for row in bq.query(sql, job_config=config).result()]


def extract_company(bq: bigquery.Client, ticker: str) -> list[dict]:
    chunks = fetch_risk_chunks(bq, ticker)
    by_id = {c["chunk_id"]: c for c in chunks}
    rows = []

    for start in range(0, len(chunks), CHUNKS_PER_CALL):
        batch = chunks[start:start + CHUNKS_PER_CALL]
        excerpts = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in batch)
        response = generate_with_retry(
            model=MODEL,
            contents=PROMPT.format(company=batch[0]["company"], excerpts=excerpts),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Extraction,  # Gemini must return JSON matching this shape
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        result = response.parsed or Extraction.model_validate_json(response.text)

        for s in result.risks:
            source = by_id.get(s.chunk_id)
            # Grounding check: does the quote really appear, word for word, in the chunk it claims?
            verified = bool(source) and normalize(s.evidence_quote) in normalize(source["text"])
            rows.append({
                "ticker": ticker,
                "company": batch[0]["company"],
                "cluster": s.cluster,
                "risk": s.risk,
                "evidence": s.evidence_quote,
                "source": s.chunk_id,
                "quote_verified": verified,
            })
        time.sleep(5)  # be gentle with rate limits

    return rows


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or TICKERS  # e.g. `python src/extract_risks.py WMT NVDA`
    bq = bigquery.Client(project=PROJECT)

    rows = []
    for ticker in tickers:
        company_rows = extract_company(bq, ticker)
        print(f"{ticker}: {len(company_rows)} risks extracted")
        rows.extend(company_rows)

    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "evidence"])  # overlap can repeat a quote
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "risk_table.csv", index=False)

    # Company x cluster matrix (verified risks only): a quick view of who is exposed to what
    matrix = df[df["quote_verified"]].pivot_table(
        index="company", columns="cluster", values="risk", aggfunc="count", fill_value=0
    )
    matrix.to_csv(OUT_DIR / "risk_matrix.csv")

    print(f"\n{len(df)} risks, quote accuracy {df['quote_verified'].mean():.0%}")
    print(matrix.to_string())
    print(f"\nSaved {OUT_DIR / 'risk_table.csv'} and {OUT_DIR / 'risk_matrix.csv'}")


if __name__ == "__main__":
    main()