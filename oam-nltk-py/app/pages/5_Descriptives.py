"""Step 5 -- Descriptive statistics.

Supporting charts and tables: term frequency, dictionary coverage by
category, document-level heatmap, and sentiment scores.  These
complement the network graph with quantitative summaries.
"""
from __future__ import annotations

import streamlit as st

from oam_nltk.salience import dictionary_coverage
from oam_nltk.sentiment import score_sentiment
from oam_nltk import viz

st.title("Descriptives")
st.markdown(
    "Summary statistics and charts showing how your dictionary terms "
    "distribute across the corpus.  These support the network view with "
    "quantitative detail."
)

if "corpus" not in st.session_state or "tokens" not in st.session_state.get("corpus", {}).columns:
    st.warning("Load and preprocess a corpus first (step 1: Upload).")
    st.stop()
if "dictionary" not in st.session_state:
    st.warning("Select a dictionary first (step 2: Dictionary).")
    st.stop()

df = st.session_state["corpus"]
d = st.session_state["dictionary"]

# --- coverage ----------------------------------------------------------------
st.subheader("Dictionary coverage")
st.markdown(
    "How often each dictionary term appears across your documents.  "
    "The bar chart shows the most frequent terms; the pie chart shows "
    "the share of each category; the heatmap shows coverage per document."
)

cov = dictionary_coverage(df["tokens"].tolist(), d, doc_ids=df["name"].tolist())

if cov.empty:
    st.info("No dictionary terms found in the corpus.  Try a different "
            "dictionary or check your preprocessing settings.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(viz.bar_top_terms(cov, top_n=25), use_container_width=True)
with c2:
    st.plotly_chart(viz.pie_category_coverage(cov), use_container_width=True)

st.plotly_chart(viz.heatmap_doc_category(cov), use_container_width=True)

with st.expander("Coverage table"):
    st.dataframe(cov, use_container_width=True)

# --- sentiment ---------------------------------------------------------------
st.divider()
st.subheader("Sentiment")
st.markdown(
    "VADER sentiment scoring per document.  Compound scores range from "
    "-1 (most negative) to +1 (most positive).  Useful for spotting "
    "which documents are predominantly positive, negative, or neutral."
)

with st.spinner("Scoring sentiment..."):
    sent = score_sentiment(df["text"].fillna("").tolist(),
                           doc_ids=df["name"].tolist())
st.plotly_chart(viz.sentiment_bar(sent), use_container_width=True)

with st.expander("Sentiment table"):
    st.dataframe(sent, use_container_width=True)
