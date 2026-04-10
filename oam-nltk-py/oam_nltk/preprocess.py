"""Text cleaning, tokenisation, stop-word removal (multilingual).

Performance notes
-----------------
- Language detection is done ONCE per document at the DataFrame level, not
  per-token.  The slow ``langdetect`` call is wrapped with a fast-path that
  skips detection when the caller already supplies a language column.
- Stopword sets are frozen and cached so the NLTK lookup happens only once
  per language across the whole session.
- Lemmatisation is OFF by default.  It is the single slowest step and is
  rarely needed for co-occurrence / topic modelling.  Turn it on explicitly
  if you need it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s]", flags=re.UNICODE)
_NUMBERS = re.compile(r"\b\d+\b")


# ---------------------------------------------------------------------------
# Stopwords — cached per language, downloaded once silently
# ---------------------------------------------------------------------------

@lru_cache(maxsize=12)
def _stopwords(lang: str) -> frozenset[str]:
    _LANG_MAP = {
        "en": "english", "fr": "french", "es": "spanish",
        "ar": "arabic", "de": "german", "pt": "portuguese",
        "it": "italian", "nl": "dutch", "sw": "english",
    }
    try:
        from nltk.corpus import stopwords as nltk_sw
        return frozenset(nltk_sw.words(_LANG_MAP.get(lang, "english")))
    except LookupError:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords as nltk_sw
            return frozenset(nltk_sw.words(_LANG_MAP.get(lang, "english")))
        except Exception:
            return frozenset()
    except Exception:
        return frozenset()


def _detect_language(text: str) -> str:
    """Best-effort language detection.  Returns ISO-639-1 code or 'en'."""
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect
        return detect(text[:3000])
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """Fast, configurable text preprocessor.

    The default settings (lowercase + strip punctuation + strip numbers +
    remove English stopwords) are tuned for the OAM topic-modelling and
    network pipeline.  They run entirely on regexes and set lookups — no
    heavy NLP models required.
    """
    lowercase: bool = True
    strip_punct: bool = True
    strip_numbers: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    language: str | None = None      # None = autodetect once per doc
    extra_stopwords: set[str] = field(default_factory=set)

    def clean(self, text: str) -> str:
        """Return whitespace-normalised text with optional lowercasing,
        punctuation removal and number removal."""
        if not text:
            return ""
        if self.lowercase:
            text = text.lower()
        if self.strip_punct:
            text = _NON_ALNUM.sub(" ", text)
        if self.strip_numbers:
            text = _NUMBERS.sub(" ", text)
        return _WHITESPACE.sub(" ", text).strip()

    def tokenize(self, text: str, lang: str | None = None) -> list[str]:
        """Clean text and return a list of filtered tokens."""
        lang = lang or self.language or "en"
        tokens = self.clean(text).split()
        if self.remove_stopwords:
            sw = _stopwords(lang) | self.extra_stopwords
            tokens = [t for t in tokens if t not in sw]
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return tokens

    # ---- batch API (operates on a whole DataFrame) -------------------------

    def add_to_dataframe(self, df: pd.DataFrame,
                         text_col: str = "text") -> pd.DataFrame:
        """Add ``clean_text``, ``tokens``, ``n_tokens_clean`` columns.

        If the DataFrame already has a ``language`` column it is used
        directly, skipping the expensive per-document detection.
        """
        df = df.copy()

        # Detect languages up-front (once), only where missing
        if "language" not in df.columns:
            df["language"] = df[text_col].fillna("").map(
                lambda t: _detect_language(t[:3000]))
        else:
            df["language"] = df["language"].fillna("en")

        df["clean_text"] = df[text_col].fillna("").map(self.clean)

        tokens_col: list[list[str]] = []
        for text, lang in zip(df[text_col].fillna(""), df["language"]):
            try:
                tokens_col.append(self.tokenize(text, lang))
            except Exception as exc:
                log.warning("tokenize error: %s", exc)
                tokens_col.append([])
        df["tokens"] = tokens_col
        df["n_tokens_clean"] = df["tokens"].map(len)
        return df


def preprocess(texts: Iterable[str], **kwargs) -> list[list[str]]:
    """One-shot helper: tokenise a list of raw texts."""
    pp = Preprocessor(**kwargs)
    return [pp.tokenize(t) for t in texts]
