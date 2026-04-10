"""Step 6 -- Timeline replay.

If your documents have dates (parsed from filenames like
'NDC_Kenya_2020.pdf' or from a 'date' column), this page builds a
separate network graph for each time period and lets you step through
them to see how connections evolve.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from oam_nltk.temporal import time_sliced_graphs
from oam_nltk.network import layout_positions, ALL_LAYOUTS

st.title("Timeline")
st.markdown(
    "Step through the network across time periods.  Each slice builds "
    "a separate co-occurrence graph from only the documents in that "
    "period, so you can watch themes emerge, grow and fade."
)

if "corpus" not in st.session_state:
    st.warning("Load a corpus first (step 1: Upload).")
    st.stop()
if "dictionary" not in st.session_state:
    st.warning("Select a dictionary first (step 2: Dictionary).")
    st.stop()

df = st.session_state["corpus"]
d = st.session_state["dictionary"]

if "date" not in df.columns or df["date"].notna().sum() == 0:
    st.info(
        "No dates detected in your documents.  For timeline analysis, "
        "include a year in the filename (e.g. 'Report_Kenya_2020.pdf') "
        "or ensure the corpus DataFrame has a 'date' column."
    )
    st.stop()

# --- controls ----------------------------------------------------------------
col1, col2 = st.columns(2)
freq = col1.selectbox("Time bucket", ["Y", "Q", "M"],
                      format_func={"Y": "Year", "Q": "Quarter", "M": "Month"}.get)
layout_key = col2.selectbox("Layout", [k for k, _ in ALL_LAYOUTS],
                            format_func=lambda k: dict(ALL_LAYOUTS)[k])

# --- compute -----------------------------------------------------------------
@st.cache_data(show_spinner="Building time slices...")
def _slices(_df_hash, freq, dict_name):
    return time_sliced_graphs(df, freq=freq, dictionary=d)

slices = _slices(id(df), freq, d.name)

if not slices:
    st.warning("No non-empty time slices found.  Check that your documents "
               "have parseable dates.")
    st.stop()

periods = sorted(slices.keys())
period = st.select_slider("Period", options=periods, value=periods[-1])

result = slices[period]
layout_positions(result, layout=layout_key)

c1, c2, c3 = st.columns(3)
c1.metric("Period", period)
c2.metric("Nodes", len(result.graph))
c3.metric("Edges", result.graph.number_of_edges())

if len(result.graph) == 0:
    st.info("This period has no co-occurrences above the threshold.")
else:
    components.html(result.to_pyvis_html(height="750px"), height=780, scrolling=True)
