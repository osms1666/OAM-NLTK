"""OAM-NLTK Dashboard -- Home page."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="OAM-NLTK",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OAM-NLTK")
st.subheader("Network Systems Modelling of Large Text Bases")
st.caption("Hu & Skorge | 2025-2026 | Python edition")

st.markdown("""
This toolkit lets you upload a collection of documents (PDF, DOCX, TXT),
apply a thematic dictionary, run topic modelling and entity extraction,
and explore the results as an **interactive network graph** that you can
click, drag, reshape and export.

**Workflow -- follow the pages in order:**

1. **Upload** -- load your documents and run preprocessing.
2. **Dictionary** -- pick a built-in thematic lexicon or create your own.
3. **Network** -- the main output.  Build a co-occurrence graph, switch
   between five layouts (force, spider-web, earth, radar, hierarchical),
   drag nodes, and export to Gephi or HTML.
4. **Topics and Entities** -- LDA topic model and spaCy named-entity
   extraction, shown as charts you can cross-reference with the network.
5. **Descriptives** -- supporting bar charts, coverage tables and sentiment
   scores.
6. **Timeline** -- replay the network across time periods if your documents
   have dates.

Use the sidebar to navigate.  Every page shares the same in-memory corpus
and dictionary, so you only load once.
""")

st.divider()

with st.sidebar:
    st.header("Session")
    if "corpus" in st.session_state:
        n = len(st.session_state["corpus"])
        st.success(f"Corpus: {n} documents loaded")
    else:
        st.info("No corpus loaded yet.")
    if "dictionary" in st.session_state:
        d = st.session_state["dictionary"]
        st.success(f"Dictionary: {d.name} ({len(d.all_terms)} terms)")
    else:
        st.info("No dictionary selected.")
