"""Step 6 of the pipeline: run a fixed test set through the RAG system and score it."""
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from answer import MODEL, TOP_K, answer

QUESTIONS_PATH = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")
REFUSAL_PHRASES = ["don't cover", "do not cover"]


def is_relevant(chunk: dict, expected: dict) -> bool:
    """A chunk counts as relevant if it's from the right company (and section, if given)
    and contains at least one expected keyword."""
    if chunk["ticker"] != expected["ticker"]:
        return False
    if expected.get("item") and chunk["item"] != expected["item"]:
        return False
    text = f" {chunk['text'].lower()} "
    return any(k.lower() in text for k in expected["keywords"])


def score(q: dict, r: dict) -> dict:
    refused = any(p in r["answer"].lower() for p in REFUSAL_PHRASES)
    row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"], "refused": refused,
           "invalid_citations": r["invalid_citations"], "cost_usd": r["cost_usd"],
           "latency_ms": r["retrieval_ms"] + r["generation_ms"], "answer": r["answer"]}
    if q["answerable"]:
        relevant = [is_relevant(c, q["expected"]) for c in r["chunks"]]
        first = relevant.index(True) + 1 if any(relevant) else None
        by_id = {c["chunk_id"]: c for c in r["chunks"]}
        row |= {
            "hit": first is not None,                     # did retrieval find ANY relevant chunk?
            "reciprocal_rank": 1 / first if first else 0,  # 1.0 if it was ranked first, 0.5 if second...
            "cited_relevant": any(is_relevant(by_id[c], q["expected"]) for c in r["cited"] if c in by_id),
        }
    return row


def pct(values) -> str:
    values = list(values)
    return f"{sum(values) / len(values):.0%}" if values else "n/a"


def main():
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    rows = []
    for q in questions:
        r = answer(q["question"], q["ticker"])
        row = score(q, r)
        rows.append(row)
        if q["answerable"]:
            status = f"hit={'Y' if row['hit'] else 'N'} rr={row['reciprocal_rank']:.2f} cited_ok={'Y' if row['cited_relevant'] else 'N'}"
            status += "  REFUSED!" if row["refused"] else ""
        else:
            status = "refused (good)" if row["refused"] else "ANSWERED A TRAP!"
        print(f"{q['id']}  {status:<40} {q['question']}")
        time.sleep(2)  # be gentle with rate limits

    answerable = [r for r in rows if r["answerable"]]
    traps = [r for r in rows if not r["answerable"]]
    summary = {
        "model": MODEL,
        "top_k": TOP_K,
        "n_questions": len(rows),
        "retrieval_hit_rate": pct(r["hit"] for r in answerable),
        "mrr": round(statistics.mean(r["reciprocal_rank"] for r in answerable), 3),
        "cited_relevant_rate": pct(r["cited_relevant"] for r in answerable),
        "wrongly_refused": pct(r["refused"] for r in answerable),
        "trap_refusal_rate": pct(r["refused"] for r in traps),
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in rows),
        "avg_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows)),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:<32} {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved details to {out}")


if __name__ == "__main__":
    main()