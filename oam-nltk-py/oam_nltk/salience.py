"""Salience and co-occurrence measures.

- term_frequency / log_tf -- per-document word counts.
- tf_idf               -- corpus-level TF-IDF via scikit-learn.
- pmi_cooccurrence     -- sliding-window PMI (edges for the network graph).
- dictionary_coverage  -- how much of a dictionary appears in each document.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import pandas as pd

log = logging.getLogger(__name__)


def term_frequency(tokens: Sequence[str],
                   vocabulary: Iterable[str] | None = None) -> dict[str, int]:
    counts = Counter(tokens)
    if vocabulary is not None:
        vocab = {v.lower() for v in vocabulary}
        return {w: c for w, c in counts.items() if w.lower() in vocab}
    return dict(counts)


def log_tf(tokens: Sequence[str],
           vocabulary: Iterable[str] | None = None) -> dict[str, float]:
    return {w: math.log1p(c) for w, c in term_frequency(tokens, vocabulary).items()}


def tf_idf(corpus_tokens: Sequence[Sequence[str]],
           vocabulary: Iterable[str] | None = None,
           max_features: int = 5000) -> pd.DataFrame:
    """Corpus-level TF-IDF. Returns doc x term DataFrame."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        log.error("scikit-learn is required for tf_idf.")
        return pd.DataFrame()
    docs = [" ".join(toks) for toks in corpus_tokens]
    vocab = sorted({v.lower() for v in vocabulary}) if vocabulary else None
    vec = TfidfVectorizer(vocabulary=vocab, max_features=max_features, lowercase=True)
    X = vec.fit_transform(docs)
    return pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())


def pmi_cooccurrence(
    corpus_tokens: Sequence[Sequence[str]],
    window: int = 5,
    min_count: int = 3,
    vocabulary: Iterable[str] | None = None,
    top_n: int | None = 3000,
) -> pd.DataFrame:
    """Sliding-window PMI co-occurrence.

    Returns DataFrame: word1, word2, count, pmi.
    """
    vocab = {v.lower() for v in vocabulary} if vocabulary else None
    word_counts: Counter[str] = Counter()
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_words = 0
    total_pairs = 0

    for tokens in corpus_tokens:
        tokens_lower = [t.lower() for t in tokens]
        total_words += len(tokens_lower)
        word_counts.update(tokens_lower)
        n = len(tokens_lower)
        for i in range(n):
            w1 = tokens_lower[i]
            for j in range(i + 1, min(i + window + 1, n)):
                w2 = tokens_lower[j]
                if w1 == w2:
                    continue
                pair = (w1, w2) if w1 < w2 else (w2, w1)
                pair_counts[pair] += 1
                total_pairs += 1

    if total_pairs == 0 or total_words == 0:
        return pd.DataFrame(columns=["word1", "word2", "count", "pmi"])

    rows = []
    for (w1, w2), c in pair_counts.items():
        if c < min_count:
            continue
        if vocab is not None and (w1 not in vocab or w2 not in vocab):
            continue
        p_xy = c / total_pairs
        p_x = word_counts[w1] / total_words
        p_y = word_counts[w2] / total_words
        if p_x == 0 or p_y == 0:
            continue
        pmi = math.log(p_xy / (p_x * p_y))
        rows.append((w1, w2, c, round(pmi, 4)))

    df = pd.DataFrame(rows, columns=["word1", "word2", "count", "pmi"])
    df = df.sort_values("pmi", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df


def dictionary_coverage(
    corpus_tokens: Sequence[Sequence[str]],
    dictionary,
    doc_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Long-form DataFrame: doc, category, term, n, log_tf."""
    doc_ids = list(doc_ids) if doc_ids else [f"doc{i}" for i in range(len(corpus_tokens))]
    term_to_cat = {t.lower(): c
                   for c, ts in dictionary.categories.items() for t in ts}
    rows = []
    for doc_id, tokens in zip(doc_ids, corpus_tokens):
        counts = Counter(t.lower() for t in tokens)
        for term, cat in term_to_cat.items():
            c = counts.get(term, 0)
            if c == 0:
                continue
            rows.append((doc_id, cat, term, c, round(math.log1p(c), 4)))
    if not rows:
        return pd.DataFrame(columns=["doc", "category", "term", "n", "log_tf"])
    return pd.DataFrame(rows, columns=["doc", "category", "term", "n", "log_tf"])
