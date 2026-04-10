"""Core unit tests (no heavy model downloads needed)."""
from __future__ import annotations

from oam_nltk.dictionaries import load_dictionary, list_dictionaries
from oam_nltk.preprocess import Preprocessor
from oam_nltk.salience import log_tf, pmi_cooccurrence, dictionary_coverage
from oam_nltk.network import build_graph, layout_positions


CORPUS = [
    "Women and youth in Kenya face climate change impacts and need adaptation support.",
    "Farmers and pastoralists in Ethiopia suffer from drought and food insecurity.",
    "Governance and participation are critical for climate finance delivery.",
    "Children and refugees in Somalia require health and education services.",
]


def test_list_dictionaries_excludes_removed():
    names = list_dictionaries()
    assert "oam" in names
    assert "removed" not in names
    assert "epstein" not in names


def test_load_oam_dictionary():
    d = load_dictionary("oam")
    assert "Demographic Groups" in d.categories
    assert "women" in d.all_terms


def test_preprocessor_fast():
    pp = Preprocessor()
    toks = pp.tokenize(CORPUS[0])
    assert "women" in toks
    assert "kenya" in toks
    assert "and" not in toks  # stopword removed


def test_preprocessor_dataframe():
    import pandas as pd
    pp = Preprocessor()
    df = pd.DataFrame({"text": CORPUS})
    df = pp.add_to_dataframe(df)
    assert "tokens" in df.columns
    assert "clean_text" in df.columns
    assert all(df["n_tokens_clean"] > 0)


def test_dictionary_coverage():
    pp = Preprocessor()
    corpus_tokens = [pp.tokenize(t) for t in CORPUS]
    d = load_dictionary("oam")
    cov = dictionary_coverage(corpus_tokens, d,
                              doc_ids=[f"doc{i}" for i in range(len(CORPUS))])
    assert not cov.empty
    assert set(cov.columns) == {"doc", "category", "term", "n", "log_tf"}


def test_pmi_returns_dataframe():
    pp = Preprocessor()
    tokens = [pp.tokenize(t) for t in CORPUS]
    co = pmi_cooccurrence(tokens, window=10, min_count=1)
    assert {"word1", "word2", "count", "pmi"} <= set(co.columns)


def test_graph_builds_and_all_layouts_work():
    pp = Preprocessor()
    tokens = [pp.tokenize(t) for t in CORPUS]
    co = pmi_cooccurrence(tokens, window=10, min_count=1)
    g = build_graph(co, min_count=1, top_edges=200)
    assert len(g.graph) > 0
    for layout in ("force", "spider", "earth", "radar", "hierarchical"):
        pos = layout_positions(g, layout=layout)
        assert len(pos) == len(g.graph)


def test_empty_cooccurrence_gives_empty_graph():
    import pandas as pd
    co = pd.DataFrame(columns=["word1", "word2", "count", "pmi"])
    g = build_graph(co)
    assert len(g.graph) == 0
