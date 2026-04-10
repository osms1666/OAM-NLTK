"""Plotly charts for the descriptives panel."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bar_top_terms(df: pd.DataFrame, top_n: int = 20, title: str = "Top terms"):
    top = df.groupby("term", as_index=False)["n"].sum().nlargest(top_n, "n")
    if top.empty:
        return go.Figure().update_layout(title=title)
    return (px.bar(top, x="n", y="term", orientation="h", title=title,
                   color="n", color_continuous_scale="Viridis")
              .update_layout(yaxis=dict(categoryorder="total ascending")))


def pie_category_coverage(df: pd.DataFrame, title: str = "Category share"):
    agg = df.groupby("category", as_index=False)["n"].sum()
    if agg.empty:
        return go.Figure().update_layout(title=title)
    return px.pie(agg, names="category", values="n", title=title, hole=0.35)


def heatmap_doc_category(df: pd.DataFrame, title: str = "Coverage heatmap"):
    pivot = (df.groupby(["doc", "category"], as_index=False)["log_tf"].sum()
               .pivot(index="doc", columns="category", values="log_tf")
               .fillna(0))
    if pivot.empty:
        return go.Figure().update_layout(title=title)
    return px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis",
                     title=title)


def sentiment_bar(sent_df: pd.DataFrame, title: str = "Document sentiment"):
    if sent_df.empty:
        return go.Figure().update_layout(title=title)
    return px.bar(sent_df, x="doc", y="compound", color="label", title=title,
                  color_discrete_map={"positive": "#2ecc71",
                                      "neutral": "#95a5a6",
                                      "negative": "#e74c3c"})


def topics_bar(top_words: pd.DataFrame, top_n: int = 8):
    if top_words.empty:
        return go.Figure().update_layout(title="LDA topics")
    return (px.bar(top_words.groupby("topic").head(top_n),
                   x="weight", y="term", color="topic", facet_col="topic",
                   facet_col_wrap=2, orientation="h",
                   title="LDA topics -- top terms")
              .update_yaxes(matches=None)
              .update_layout(showlegend=False, height=600))
