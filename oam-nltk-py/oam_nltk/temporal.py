"""Time-slice graphs for temporal replay of the network."""
from __future__ import annotations

import logging

import pandas as pd

from .network import GraphResult, build_graph
from .preprocess import Preprocessor
from .salience import pmi_cooccurrence

log = logging.getLogger(__name__)


def time_sliced_graphs(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    date_col: str = "date",
    freq: str = "Y",
    dictionary=None,
    window: int = 5,
    min_count: int = 2,
    top_edges: int = 500,
) -> dict[str, GraphResult]:
    """Build one graph per time period.

    Returns {period_label: GraphResult}.
    """
    if date_col not in df.columns:
        log.warning("No '%s' column found. Cannot build time slices.", date_col)
        return {}

    work = df.dropna(subset=[date_col]).copy()
    if work.empty:
        return {}

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    if work.empty:
        return {}

    work["_period"] = work[date_col].dt.to_period(freq).astype(str)
    pp = Preprocessor()
    vocab = dictionary.all_terms if dictionary is not None else None
    node_meta: dict[str, dict] = {}
    if dictionary is not None:
        for cat, terms in dictionary.categories.items():
            for t in terms:
                node_meta[t.lower()] = {"category": cat}

    out: dict[str, GraphResult] = {}
    for period, chunk in work.groupby("_period", sort=True):
        tokens_col = "tokens" if "tokens" in chunk.columns else text_col
        if tokens_col == "tokens":
            tokens = chunk["tokens"].tolist()
        else:
            tokens = [pp.tokenize(t) for t in chunk[text_col].fillna("")]

        if not any(tokens):
            continue
        co = pmi_cooccurrence(tokens, window=window, min_count=min_count,
                              vocabulary=vocab, top_n=top_edges)
        if co.empty:
            continue
        out[str(period)] = build_graph(co, node_metadata=node_meta, top_edges=top_edges)

    return out
