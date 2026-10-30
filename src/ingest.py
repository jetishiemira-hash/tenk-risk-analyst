import json
from pathlib import Path

from edgar import Company, set_identity


set_identity("Emira Jetishi jetishiemira@gmail.com")

# 10 companies across sectors, so the system has to handle different kinds of risk
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "JNJ", "XOM", "WMT", "TSLA", "NKE"]

# The 10-K sections that carry the most risk-related information
SECTIONS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 7A": "Market Risk",
}

OUT_DIR = Path("data/raw")


def fetch_10k(ticker: str) -> dict:
    """Get the latest 10-K for one company and pull out the sections we want."""
    filing = Company(ticker).get_filings(form="10-K", amendments=False).latest()
    tenk = filing.obj()  # parses the filing so we can ask for sections by name

    sections = []
    for item, title in SECTIONS.items():
        text = tenk[item]
        if not text:
            print(f"  ! {ticker}: {item} not found")
            continue
        sections.append({"item": item, "title": title, "text": text})

    # Metadata is just as important as text: it's what makes citations possible later
    return {
        "ticker": ticker,
        "company": filing.company,
        "filing_date": str(filing.filing_date),
        "accession_no": filing.accession_no,
        "url": filing.homepage_url,
        "sections": sections,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        try:
            doc = fetch_10k(ticker)
        except Exception as e:  # one bad filing shouldn't stop the whole run
            print(f"  ! {ticker} failed: {e}")
            continue
        path = OUT_DIR / f"{ticker}.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        words = sum(len(s["text"].split()) for s in doc["sections"])
        print(f"  saved {len(doc['sections'])} sections, {words:,} words, filed {doc['filing_date']}")


if __name__ == "__main__":
    main()