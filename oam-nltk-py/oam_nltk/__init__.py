"""OAM-NLTK -- Network Systems Modelling of large text bases.

Python port and expansion of the R OAM NLTK (Hu & Skorge, 2025).
"""
__version__ = "0.2.0"

from .ingestion import load_folder, load_document
from .preprocess import preprocess, Preprocessor
from .dictionaries import load_dictionary, list_dictionaries, Dictionary
from .salience import term_frequency, log_tf, tf_idf, pmi_cooccurrence
from .topics import LDAModel
from .ner import extract_entities
from .sentiment import score_sentiment
from .network import build_graph, layout_positions, GraphResult
from .temporal import time_sliced_graphs

__all__ = [
    "load_folder", "load_document",
    "preprocess", "Preprocessor",
    "load_dictionary", "list_dictionaries", "Dictionary",
    "term_frequency", "log_tf", "tf_idf", "pmi_cooccurrence",
    "LDAModel",
    "extract_entities",
    "score_sentiment",
    "build_graph", "layout_positions", "GraphResult",
    "time_sliced_graphs",
]
