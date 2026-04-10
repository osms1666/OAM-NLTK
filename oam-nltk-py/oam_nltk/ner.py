"""Named entity extraction via spaCy.

Returns a long-format DataFrame: doc, entity, label, count.
Falls back gracefully if spaCy or the requested model is not installed.
"""
from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache
from typing import Sequence

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "LAW", "NORP", "EVENT"}

_EMPTY = pd.DataFrame(columns=["doc", "entity", "label", "n"])


@lru_cache(maxsize=4)
def _load(model: str):
    import spacy
    return spacy.load(model)


def extract_entities(
    texts: Sequence[str],
    doc_ids: Sequence[str] | None = None,
    model: str = "en_core_web_sm",
    labels: set[str] | None = None,
    max_chars: int = 500_000,
) -> pd.DataFrame:
    """Run spaCy NER over a list of texts.

    Each text is truncated to *max_chars* to avoid memory issues on
    very large PDFs.  Returns an empty DataFrame (not an error) when
    spaCy is unavailable.
    """
    labels = labels or DEFAULT_LABELS
    doc_ids = list(doc_ids) if doc_ids else [f"doc{i}" for i in range(len(texts))]

    try:
        nlp = _load(model)
    except Exception as exc:
        log.warning("Could not load spaCy model '%s': %s.  "
                    "Install it with: python -m spacy download %s", model, exc, model)
        return _EMPTY

    rows = []
    for doc_id, text in zip(doc_ids, texts):
        if not text:
            continue
        try:
            doc = nlp(text[:max_chars])
        except Exception as exc:
            log.warning("NER failed for %s: %s", doc_id, exc)
            continue
        ents = [(ent.text.strip(), ent.label_)
                for ent in doc.ents if ent.label_ in labels]
        for (ent_text, label), n in Counter(ents).items():
            rows.append((doc_id, ent_text, label, n))

    if not rows:
        return _EMPTY
    return pd.DataFrame(rows, columns=["doc", "entity", "label", "n"])
