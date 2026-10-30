"""Step 7 of the pipeline: a web app for the 10-K Risk Analyst, deployed on Cloud Run."""
import logging

import streamlit as st

from answer import MODEL, answer

logging.basicConfig(level=logging.INFO, format="%(message)s")  # request logs -> stdout -> Cloud Logging

COMPANIES = {
    "All companies": None,
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Amazon": "AMZN", "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ", "ExxonMobil": "XOM", "Walmart": "WMT", "Tesla": "TSLA", "Nike": "NKE",
}
EXAMPLES = [
    "How does Walmart manage interest rate risk?",
    "Which company is most affected by export controls on advanced chips?",
    "What climate-related risks does ExxonMobil describe?",
    "What does Johnson & Johnson say about talc litigation?",
]
MAX_QUESTIONS_PER_SESSION = 15  # protects the cloud budget on a public demo

st.set_page_config(page_title="10-K Risk Analyst", layout="centered")
st.title("10-K Risk Analyst")
st.caption(
    "Ask about the latest SEC 10-K filings of 10 public companies. Answers use only the filings, "
    f"with citations. Retrieval: BigQuery vector search · Generation: {MODEL} on Vertex AI."
)

company = st.selectbox("Company", list(COMPANIES))
question = st.text_input("Question", placeholder="e.g. How does Walmart manage interest rate risk?")
with st.expander("Example questions"):
    for example in EXAMPLES:
        st.markdown(f"- {example}")

st.session_state.setdefault("asked", 0)

if st.button("Ask", type="primary", disabled=not question.strip()):
    if st.session_state.asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the question limit for this demo session. Refresh the page to start over.")
        st.stop()
    st.session_state.asked += 1

    with st.spinner("Searching the filings..."):
        r = answer(question.strip(), COMPANIES[company])

    st.markdown(r["answer"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{(r['retrieval_ms'] + r['generation_ms']) / 1000:.1f} s")
    col2.metric("Tokens in / out", f"{r['input_tokens']:,} / {r['output_tokens']:,}")
    col3.metric("Speed", f"{r['output_tokens_per_sec']} tok/s")
    col4.metric("Cost", f"${r['cost_usd']:.4f}")

    if r["invalid_citations"]:
        st.warning(f"The model cited sources that weren't retrieved: {', '.join(r['invalid_citations'])}")

    st.subheader("Sources")
    cited = set(r["cited"])
    for c in r["chunks"]:
        tag = "CITED · " if c["chunk_id"] in cited else ""
        with st.expander(f"{tag}{c['company']} · {c['item']} {c['section_title']} · similarity {c['score']:.3f}"):
            st.write(c["text"])
            st.markdown(f"`{c['chunk_id']}` · [Filing on SEC EDGAR]({c['url']})")