"""Sentiment scoring (VADER).

Returns a DataFrame: doc, neg, neu, pos, compound, label.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import pandas as pd

log = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(columns=["doc", "neg", "neu", "pos", "compound", "label"])


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def score_sentiment(
    texts: Sequence[str],
    doc_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    doc_ids = list(doc_ids) if doc_ids else [f"doc{i}" for i in range(len(texts))]
    try:
        analyser = _vader()
    except Exception as exc:
        log.warning("VADER unavailable: %s", exc)
        return _EMPTY
    rows = []
    for doc_id, text in zip(doc_ids, texts):
        try:
            s = analyser.polarity_scores(text or "")
            rows.append((doc_id, s["neg"], s["neu"], s["pos"],
                         round(s["compound"], 4), _classify(s["compound"])))
        except Exception:
            rows.append((doc_id, 0, 0, 0, 0, "neutral"))
    return pd.DataFrame(rows, columns=["doc", "neg", "neu", "pos", "compound", "label"])
