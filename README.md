# OAM-NLTK (Python)

**Operational Analysis and Mapping - Natural Language Toolkit**

A Python port and expansion of the R-based OAM NLTK (Hu & Skorge, 2025).

OAM-NLTK is a network systems modeling toolkit for large text bases.
It maps the configuration of text as data across time and space and
renders the result as an interactive, draggable network graph of
concepts, entities, and themes.

---

## Quick start

```bash
git clone https://github.com/santiweide/oam-nltk.git
cd oam-nltk
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Download NLTK stopwords (runs once)
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Launch the dashboard
oam-nltk dashboard
```

Open http://localhost:8501 and follow the pages in order:
Upload, Dictionary, Network, Topics, Descriptives, Timeline.

### Optional extras

```bash
# OCR for scanned PDFs (also needs Tesseract binary installed)
pip install -e ".[ocr]"

# spaCy NER (also run: python -m spacy download en_core_web_sm)
pip install -e ".[spacy]"

# pyLDAvis interactive topic explorer
pip install -e ".[lda-viz]"

# Everything at once
pip install -e ".[all]"
```

---

## What it does

1. **Ingestion** -- reads PDF, DOCX, TXT, HTML.  Falls back to OCR for
   scanned PDFs if Tesseract is installed.

2. **Preprocessing** -- fast regex-based tokenisation, stopword removal
   (multilingual: en/fr/es/ar/de/pt), number and punctuation stripping.
   No heavy NLP models required for the core pipeline.

3. **Dictionaries** -- thematic lexicons mapping categories to keywords.
   Seven built-in dictionaries (OAM, environmental, governance, gender,
   conflict, health, economy).  Create and save your own from the
   dashboard or as JSON files.

4. **Network graph** -- the centrepiece.  Builds a co-occurrence graph
   weighted by PMI (pointwise mutual information).  Five switchable
   layouts: force-directed, spider-web, earth (geographic), radar
   (category sectors), hierarchical.  Rendered with pyvis -- click,
   drag, zoom, pin nodes.  Export to GEXF (Gephi), HTML, or CSV.

5. **Topics** -- Gensim LDA with configurable topic count, coherence
   scoring, and optional pyLDAvis.

6. **NER** -- spaCy named-entity extraction (people, organisations,
   places, dates).

7. **Descriptives** -- bar charts, pie charts, heatmaps, sentiment
   scoring (VADER).

8. **Timeline** -- time-slider replay of the network across
   years/quarters/months.

---

## Project layout

```
oam-nltk/
  oam_nltk/                 core library
    ingestion.py            document loaders + OCR
    preprocess.py           tokenisation, cleaning
    dictionaries/           built-in JSON lexicons + user/ folder
    salience.py             TF, TF-IDF, PMI co-occurrence
    topics.py               Gensim LDA
    ner.py                  spaCy NER
    sentiment.py            VADER sentiment
    network.py              graph construction + 5 layouts + pyvis
    temporal.py             time-slice graphs
    viz.py                  plotly charts
    cli.py                  click CLI
  app/                      Streamlit dashboard
    Home.py
    pages/
      1_Upload.py
      2_Dictionary.py
      3_Network.py          <-- the centrepiece
      4_Topics_and_Entities.py
      5_Descriptives.py
      6_Timeline.py
  tests/
  pyproject.toml
  requirements.txt
```

---

## CLI

```bash
oam-nltk ingest ./my_corpus --out corpus.parquet
oam-nltk analyse corpus.parquet --dictionary oam --topics 8 --out results/
oam-nltk graph corpus.parquet --dictionary environmental --layout spider \
  --export graph.gexf --html graph.html
oam-nltk dicts
oam-nltk dashboard
```

---

## Dictionaries

Built-in:

| Name          | Focus                                     |
|---------------|-------------------------------------------|
| oam           | OAM coverage lexicon (demographic, geographic, thematic, ownership) |
| environmental | Climate, biodiversity, adaptation, finance |
| governance    | Institutions, rule of law, participation   |
| gender        | Women's empowerment, GBV, rights           |
| conflict      | Peace, security, fragility                 |
| health        | Public health systems                      |
| economy       | Macro, trade, labour, development          |

Create your own in the Dictionary page of the dashboard or drop a JSON
file into `oam_nltk/dictionaries/user/`.

NB: This OAM-NLTK is a port of the original text modeling methodology developed by Hu & Skorge (2025) for OAM Consulting. This version is a simplified version that may be utilized by anyone under the MIT License.

---

## Citation

Hu, M. & Skorge, O. (2025). The OAM Natural Language Toolkit.
Python port (2026): same authors.

## License

MIT
