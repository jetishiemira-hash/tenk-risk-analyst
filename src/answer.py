"""Step 5 of the pipeline: RAG. Retrieve chunks from BigQuery, then have Gemini answer with citations."""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from google.genai import types

from bq_search import parse_question, retrieve
from embed import client

MODEL = "gemini-3.6-flash"
TOP_K = 6
# Approximate list prices in USD per 1M tokens (Gemini API standard tier, through Dec 31, 2026)
PRICE_INPUT_PER_M = 0.75
PRICE_OUTPUT_PER_M = 3.75   # output price also applies to "thinking" tokens
LOG_PATH = Path("logs/requests.jsonl")

SYSTEM_INSTRUCTION = """You are a financial analyst assistant answering questions about SEC 10-K filings.
Rules:
- Use ONLY the excerpts provided. Do not use outside knowledge.
- Cite every claim with its source id in square brackets, e.g. [WMT-2026-Item7A-000].
- If the excerpts do not contain the answer, reply: "The provided filings don't cover this."
- Be concise: a short paragraph or a few bullet points."""

CITATION_PATTERN = re.compile(r"\b[A-Z]+-\d{4}-Item\w+-\d{3}\b")


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Put each retrieved chunk in the prompt with its id, so the model can cite it."""
    sources = "\n\n".join(
        f"[{c['chunk_id']}] {c['company']} 10-K filed {c['filing_date']}, {c['item']} {c['section_title']}\n{c['text']}"
        for c in chunks
    )
    return f"Excerpts:\n\n{sources}\n\nQuestion: {question}"


def answer(question: str, ticker: str | None = None) -> dict:
    t0 = time.perf_counter()
    chunks = retrieve(question, k=TOP_K, ticker=ticker)
    t1 = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),  # less thinking = faster + cheaper
        ),
    )
    t2 = time.perf_counter()

    # LLM-native metrics: tokens, speed, cost
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    answer_tokens = usage.candidates_token_count or 0
    thinking_tokens = usage.thoughts_token_count or 0
    output_tokens = answer_tokens + thinking_tokens
    cost = input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M

    # Citation check: did the model only cite chunks we actually gave it?
    text = response.text or ""
    cited = set(CITATION_PATTERN.findall(text))
    retrieved_ids = {c["chunk_id"] for c in chunks}

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "ticker_filter": ticker,
        "model": MODEL,
        "answer": text,
        "retrieved": [c["chunk_id"] for c in chunks],
        "cited": sorted(cited),
        "invalid_citations": sorted(cited - retrieved_ids),
        "retrieval_ms": round((t1 - t0) * 1000),
        "generation_ms": round((t2 - t1) * 1000),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "output_tokens_per_sec": round(output_tokens / (t2 - t1), 1) if t2 > t1 else None,
        "cost_usd": round(cost, 6),
    }

    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:  # one JSON line per request = simple tracing
        f.write(json.dumps(result) + "\n")
    return result


def main():
    print("Ask about the 10-Ks. Prefix a ticker to filter, e.g. 'TSLA: supply chain risks'. Empty line quits.")
    while True:
        raw = input("\nQuestion: ").strip()
        if not raw:
            break
        question, ticker = parse_question(raw)
        r = answer(question, ticker)
        print(f"\n{r['answer']}\n")
        print(f"Sources cited: {', '.join(r['cited']) or 'none'}")
        if r["invalid_citations"]:
            print(f"WARNING: cited chunks that were not retrieved: {r['invalid_citations']}")
        print(
            f"[retrieval {r['retrieval_ms']} ms | generation {r['generation_ms']} ms | "
            f"{r['input_tokens']} in / {r['output_tokens']} out tokens ({r['thinking_tokens']} thinking) | "
            f"{r['output_tokens_per_sec']} tok/s | ${r['cost_usd']:.5f}]"
        )


if __name__ == "__main__":
    main()