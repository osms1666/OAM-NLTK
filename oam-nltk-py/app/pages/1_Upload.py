"""Step 1 -- Upload documents and run preprocessing.

This page loads your corpus into memory.  Supported formats: PDF (with
OCR fallback for scanned documents), DOCX, TXT, HTML, and Markdown.
After loading, preprocessing tokenises the text, removes stopwords and
punctuation, and prepares the data for all downstream analysis.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from oam_nltk.ingestion import load_folder, load_document
from oam_nltk.preprocess import Preprocessor

st.title("Upload and preprocess")
st.markdown(
    "Load your documents here.  You can point to a folder on disk, drag "
    "and drop individual files, or use the built-in sample.  Once loaded, "
    "run preprocessing to tokenise the text before moving to the next step."
)

# --- input methods -----------------------------------------------------------
tab_folder, tab_files = st.tabs(["Folder path", "Upload files"])

with tab_folder:
    st.markdown(
        "Enter the **absolute path** to a folder containing your documents.  "
        "Sub-folders are included by default."
    )
    folder = st.text_input("Folder path", placeholder="/Users/you/Documents/corpus")
    col_a, col_b = st.columns(2)
    recursive = col_a.checkbox("Include sub-folders", value=True)
    ocr = col_b.checkbox("OCR fallback for image PDFs", value=True)
    if st.button("Load folder", type="primary", disabled=not folder):
        with st.spinner("Reading files..."):
            try:
                df = load_folder(folder, recursive=recursive, ocr=ocr)
                if df.empty:
                    st.error("No supported files found in that folder.")
                else:
                    st.session_state["corpus"] = df
                    st.session_state.pop("_preprocessed", None)
                    st.success(f"Loaded {len(df)} documents.")
            except Exception as exc:
                st.error(f"Error loading folder: {exc}")

with tab_files:
    st.markdown("Drag and drop one or more files (PDF, DOCX, TXT, HTML, MD).")
    uploads = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads and st.button("Load uploaded files", type="primary"):
        rows = []
        with tempfile.TemporaryDirectory() as td:
            for up in uploads:
                p = Path(td) / up.name
                p.write_bytes(up.getbuffer())
                try:
                    rows.append(load_document(p).to_dict())
                except Exception as exc:
                    st.warning(f"Skipped {up.name}: {exc}")
        if rows:
            st.session_state["corpus"] = pd.DataFrame(rows)
            st.session_state.pop("_preprocessed", None)
            st.success(f"Loaded {len(rows)} documents.")
        else:
            st.error("No files could be read.")

# --- preprocessing -----------------------------------------------------------
if "corpus" in st.session_state:
    df = st.session_state["corpus"]

    st.divider()
    st.subheader("Corpus preview")
    preview_cols = [c for c in df.columns if c != "text"]
    st.dataframe(df[preview_cols].head(50), use_container_width=True)

    st.divider()
    st.subheader("Preprocessing")
    st.markdown(
        "Preprocessing cleans and tokenises every document.  The defaults "
        "work well for most English-language corpora.  Adjust if needed, "
        "then press **Run**."
    )

    c1, c2 = st.columns(2)
    lower = c1.checkbox("Lowercase", value=True)
    punct = c1.checkbox("Strip punctuation", value=True)
    nums = c2.checkbox("Strip numbers", value=True)
    stop = c2.checkbox("Remove stopwords", value=True)
    extra = st.text_input(
        "Additional stopwords (comma-separated)",
        placeholder="e.g. annex, appendix, page",
    )

    if st.button("Run preprocessing", type="primary"):
        pp = Preprocessor(
            lowercase=lower,
            strip_punct=punct,
            strip_numbers=nums,
            remove_stopwords=stop,
            extra_stopwords={w.strip() for w in extra.split(",") if w.strip()},
        )
        with st.spinner("Tokenising documents..."):
            st.session_state["corpus"] = pp.add_to_dataframe(df)
            st.session_state["_preprocessed"] = True
        st.success(
            f"Done.  Average {st.session_state['corpus']['n_tokens_clean'].mean():.0f} "
            f"tokens per document after cleaning."
        )

    if st.session_state.get("_preprocessed"):
        st.info("Preprocessing complete.  Continue to **Dictionary** in the sidebar.")
