"""LDA topic modelling via Gensim.

Wraps Gensim's LdaModel with convenience methods for fitting, extracting
top words per topic, document-topic distributions, and coherence scoring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class LDAModel:
    num_topics: int = 8
    passes: int = 10
    random_state: int = 1234
    no_below: int = 2
    no_above: float = 0.85

    # Populated after .fit()
    _dictionary: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)
    _corpus_bow: list = field(default_factory=list, repr=False)

    def fit(self, corpus_tokens: Sequence[Sequence[str]]) -> "LDAModel":
        from gensim import corpora
        from gensim.models import LdaModel as _Lda

        self._dictionary = corpora.Dictionary(list(corpus_tokens))
        self._dictionary.filter_extremes(
            no_below=self.no_below, no_above=self.no_above)

        if len(self._dictionary) == 0:
            log.warning("Dictionary is empty after filtering. "
                        "Try lowering no_below or raising no_above.")
            return self

        self._corpus_bow = [self._dictionary.doc2bow(t) for t in corpus_tokens]
        self._model = _Lda(
            corpus=self._corpus_bow,
            id2word=self._dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def top_words(self, n: int = 10) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame(columns=["topic", "term", "weight"])
        rows = []
        for tid in range(self.num_topics):
            for term, weight in self._model.show_topic(tid, topn=n):
                rows.append((tid, term, round(float(weight), 5)))
        return pd.DataFrame(rows, columns=["topic", "term", "weight"])

    def doc_topics(self, doc_ids: Sequence[str] | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame(columns=["doc", "topic", "gamma"])
        ids = list(doc_ids) if doc_ids else [f"doc{i}" for i in range(len(self._corpus_bow))]
        rows = []
        for doc_id, bow in zip(ids, self._corpus_bow):
            dist = dict(self._model.get_document_topics(bow, minimum_probability=0))
            for tid in range(self.num_topics):
                rows.append((doc_id, tid, round(float(dist.get(tid, 0.0)), 4)))
        return pd.DataFrame(rows, columns=["doc", "topic", "gamma"])

    def coherence(self, corpus_tokens: Sequence[Sequence[str]],
                  measure: str = "c_v") -> float:
        if not self.is_fitted:
            return 0.0
        from gensim.models import CoherenceModel
        cm = CoherenceModel(
            model=self._model, texts=list(corpus_tokens),
            dictionary=self._dictionary, coherence=measure,
        )
        return round(float(cm.get_coherence()), 4)

    def pyldavis_html(self) -> str | None:
        """Render pyLDAvis as an HTML string, or None on failure."""
        if not self.is_fitted:
            return None
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
            vis = gensimvis.prepare(self._model, self._corpus_bow, self._dictionary)
            return pyLDAvis.prepared_data_to_html(vis)
        except Exception as exc:
            log.warning("pyLDAvis rendering failed: %s", exc)
            return None
