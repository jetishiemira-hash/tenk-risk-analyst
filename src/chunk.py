"""Step 2 of the pipeline: split 10-K sections into chunks ready for embedding."""
import json
import re
import statistics
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")

MAX_WORDS = 400        # upper limit per chunk
OVERLAP_PARAGRAPHS = 1  # repeat the last paragraph at the start of the next chunk


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines and tidy whitespace (10-Ks are full of non-breaking spaces)."""
    text = text.replace("\xa0", " ")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    """Rare case: one paragraph is longer than a whole chunk, so cut it into word windows."""
    words = paragraph.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_section(text: str) -> list[str]:
    """Pack whole paragraphs into chunks of up to MAX_WORDS, never cutting mid-paragraph."""
    paragraphs = []
    for p in split_paragraphs(text):
        paragraphs.extend(split_long_paragraph(p, MAX_WORDS) if len(p.split()) > MAX_WORDS else [p])

    chunks, current = [], []
    for p in paragraphs:
        if current and len(" ".join(current + [p]).split()) > MAX_WORDS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS else []
            # if the overlap plus the new paragraph is still too long, drop the overlap
            if len(" ".join(current + [p]).split()) > MAX_WORDS:
                current = []
        current.append(p)
    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for path in sorted(RAW_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        year = doc["filing_date"][:4]
        for section in doc["sections"]:
            for i, text in enumerate(chunk_section(section["text"])):
                # A short header so each chunk "knows" where it came from, even on its own
                header = f"{doc['company']} | 10-K filed {doc['filing_date']} | {section['item']}: {section['title']}"
                all_chunks.append({
                    "chunk_id": f"{doc['ticker']}-{year}-{section['item'].replace(' ', '')}-{i:03d}",
                    "ticker": doc["ticker"],
                    "company": doc["company"],
                    "filing_date": doc["filing_date"],
                    "item": section["item"],
                    "section_title": section["title"],
                    "url": doc["url"],
                    "text": text,
                    "embed_text": f"{header}\n\n{text}",
                    "n_words": len(text.split()),
                })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    sizes = [c["n_words"] for c in all_chunks]
    print(f"{len(all_chunks):,} chunks from {len({c['ticker'] for c in all_chunks})} companies")
    print(f"words per chunk: median {statistics.median(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")


if __name__ == "__main__":
    main()
