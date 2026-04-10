"""Step 3 -- Network graph (the centrepiece).

This page builds a co-occurrence network from your corpus and dictionary,
then renders it as an interactive graph you can click, drag, zoom and
reshape.  Nodes are terms; edges connect terms that appear near each
other across your documents, weighted by PMI (pointwise mutual
information) so that meaningful associations rise above noise.

Five layout modes let you see the same data from different angles:

  Force-directed    Standard physics simulation -- hubs pull neighbours.
  Spider-web        Concentric rings -- highest-degree nodes at centre.
  Earth             Geographic placement if documents have country tags.
  Radar             Sectors by dictionary category, like a radar chart.
  Hierarchical      Top-down tree from the most central node outwards.

Export the graph as GEXF (for Gephi), interactive HTML, or CSV
node/edge lists.
"""
from __future__ import annotations

import io

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from oam_nltk.salience import pmi_cooccurrence
from oam_nltk.network import build_graph, layout_positions, ALL_LAYOUTS

st.title("Network graph")

# --- guards ------------------------------------------------------------------
if "corpus" not in st.session_state or "tokens" not in st.session_state.get("corpus", {}).columns:
    st.warning("Load and preprocess a corpus first (step 1: Upload).")
    st.stop()
if "dictionary" not in st.session_state:
    st.warning("Select a dictionary first (step 2: Dictionary).")
    st.stop()

df = st.session_state["corpus"]
d = st.session_state["dictionary"]

# --- controls ----------------------------------------------------------------
st.markdown(
    "Adjust the parameters below to control how the graph is built.  "
    "**Restrict to dictionary** limits nodes to your dictionary keywords.  "
    "**Window** is how many tokens apart two words can be to count as "
    "co-occurring.  **Min PMI** filters out weak/random associations."
)

with st.sidebar:
    st.header("Graph parameters")
    restrict = st.checkbox("Restrict vocabulary to dictionary", value=True,
                           help="Only include terms from your dictionary as nodes.")
    window = st.slider("Co-occurrence window", 2, 20, 5,
                       help="How many tokens apart two words can be.")
    min_count = st.slider("Min pair count", 1, 20, 3,
                          help="Pairs seen fewer times than this are dropped.")
    min_pmi = st.slider("Min PMI", -2.0, 10.0, 0.5, step=0.1,
                        help="Higher = only strong associations.")
    top_edges = st.slider("Max edges", 50, 5000, 1000, step=50,
                          help="Cap on total number of edges shown.")

# --- layout selector ---------------------------------------------------------
st.subheader("Layout")
st.markdown(
    "Choose how the nodes are arranged.  You can switch at any time "
    "without recomputing the graph."
)
layout_key = st.radio(
    "Layout mode",
    [k for k, _ in ALL_LAYOUTS],
    format_func=lambda k: dict(ALL_LAYOUTS)[k],
    horizontal=True,
    label_visibility="collapsed",
)

# --- build -------------------------------------------------------------------
@st.cache_data(show_spinner="Computing co-occurrences...")
def _cooc(tokens_ser, vocab_tuple, window, min_count, top_edges):
    tokens_list = list(tokens_ser)
    return pmi_cooccurrence(
        tokens_list, window=window, min_count=min_count,
        vocabulary=list(vocab_tuple) if vocab_tuple else None,
        top_n=top_edges,
    )

vocab = tuple(d.all_terms) if restrict else None
co = _cooc(df["tokens"], vocab, window, min_count, top_edges)

node_meta: dict[str, dict] = {}
for cat, terms in d.categories.items():
    for t in terms:
        node_meta[t.lower()] = {"category": cat}
if "country" in df.columns:
    for country in df["country"].dropna().unique():
        node_meta.setdefault(str(country).lower(), {}).setdefault("country", str(country).lower())

result = build_graph(co, node_metadata=node_meta, min_pmi=min_pmi,
                     min_count=min_count, top_edges=top_edges)
layout_positions(result, layout=layout_key)

# --- metrics -----------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Nodes", len(result.graph))
c2.metric("Edges", result.graph.number_of_edges())
c3.metric("Layout", dict(ALL_LAYOUTS).get(layout_key, layout_key))

if len(result.graph) == 0:
    st.warning(
        "The graph is empty.  Try lowering Min PMI, lowering Min pair count, "
        "or switching off 'Restrict vocabulary to dictionary' to include "
        "all terms."
    )
    st.stop()

# --- render ------------------------------------------------------------------
html = result.to_pyvis_html(height="800px")
components.html(html, height=820, scrolling=True)

# --- export ------------------------------------------------------------------
st.divider()
st.subheader("Export")
st.markdown(
    "Download the graph for use in other tools.  GEXF opens in Gephi; "
    "HTML is a self-contained interactive file you can share; CSV gives "
    "you raw node and edge tables."
)

col_a, col_b, col_c = st.columns(3)

buf = io.BytesIO()
nx.write_gexf(result.graph, buf)
col_a.download_button("Download GEXF (Gephi)", buf.getvalue(),
                      file_name="oam_network.gexf", mime="application/xml")
col_b.download_button("Download HTML", html.encode("utf-8"),
                      file_name="oam_network.html", mime="text/html")
col_c.download_button("Download edges CSV",
                      result.edges_csv().encode("utf-8"),
                      file_name="oam_edges.csv", mime="text/csv")

# --- edge table --------------------------------------------------------------
with st.expander("Edge table (raw data)"):
    st.dataframe(result.edge_meta, use_container_width=True)

with st.expander("Node table"):
    st.dataframe(result.node_meta, use_container_width=True)
