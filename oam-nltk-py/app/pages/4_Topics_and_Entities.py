"""Step 4 -- Topic modelling (LDA) and named-entity extraction.

LDA discovers latent topics across your corpus.  Each topic is a cluster
of words that tend to appear together.  You control how many topics to
extract.

Named-entity recognition (NER) uses spaCy to find people, organisations,
places, dates and other real-world entities mentioned in the text.  This
is useful for mapping who/what/where across documents.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from oam_nltk.topics import LDAModel
from oam_nltk.ner import extract_entities
from oam_nltk import viz

st.title("Topics and entities")

if "corpus" not in st.session_state or "tokens" not in st.session_state.get("corpus", {}).columns:
    st.warning("Load and preprocess a corpus first (step 1: Upload).")
    st.stop()

df = st.session_state["corpus"]

# ---- LDA --------------------------------------------------------------------
st.subheader("LDA topic model")
st.markdown(
    "Latent Dirichlet Allocation groups words into topics based on how "
    "they co-occur across documents.  Use the slider to set the number "
    "of topics.  Start with 5-10 for small corpora, 10-20 for large ones."
)

col1, col2 = st.columns(2)
k = col1.slider("Number of topics", 2, 25, 8,
                help="More topics = finer granularity, but harder to interpret.")
passes = col2.slider("Training passes", 1, 30, 10,
                     help="More passes = better convergence, slower fitting.")

if st.button("Fit topic model", type="primary"):
    with st.spinner("Fitting LDA..."):
        lda = LDAModel(num_topics=k, passes=passes)
        lda.fit(df["tokens"].tolist())
    if lda.is_fitted:
        st.session_state["lda"] = lda
        st.success("Topic model fitted.")
    else:
        st.error("Could not fit topics.  The vocabulary may be too small "
                 "after filtering.  Try loading more documents.")

if "lda" in st.session_state:
    lda = st.session_state["lda"]
    top_words = lda.top_words(10)
    st.plotly_chart(viz.topics_bar(top_words), use_container_width=True)

    with st.expander("Document-topic distribution"):
        st.dataframe(lda.doc_topics(df["name"].tolist()), use_container_width=True)

    with st.expander("pyLDAvis (interactive topic explorer)"):
        html = lda.pyldavis_html()
        if html:
            components.html(html, height=800, scrolling=True)
        else:
            st.info("pyLDAvis could not render.  Install pyLDAvis for this feature.")

# ---- NER --------------------------------------------------------------------
st.divider()
st.subheader("Named-entity recognition")
st.markdown(
    "Extract real-world entities (people, organisations, places, dates) "
    "from your documents using spaCy.  You need a spaCy model installed "
    "(run `python -m spacy download en_core_web_sm`)."
)

model = st.selectbox(
    "spaCy model",
    ["en_core_web_sm", "en_core_web_lg", "fr_core_news_sm", "es_core_news_sm"],
    help="Choose the model matching your corpus language.",
)
if st.button("Extract entities"):
    with st.spinner("Running NER..."):
        ner = extract_entities(
            df["text"].fillna("").tolist(),
            doc_ids=df["name"].tolist(),
            model=model,
        )
    if ner.empty:
        st.warning("No entities found.  Check that the spaCy model is installed.")
    else:
        st.session_state["ner"] = ner
        st.success(f"Found {len(ner)} entity mentions.")

if "ner" in st.session_state:
    ner = st.session_state["ner"]
    st.plotly_chart(viz.treemap_entities(ner), use_container_width=True)
    with st.expander("Entity table"):
        st.dataframe(ner, use_container_width=True)
