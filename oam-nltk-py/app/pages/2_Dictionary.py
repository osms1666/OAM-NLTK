"""Step 2 -- Choose or create a thematic dictionary.

A dictionary is a set of categories, each containing a list of keywords.
For example the "oam" dictionary has categories like "Demographic Groups"
(women, youth, farmers...) and "Thematic Areas" (governance, health,
climate change...).

The dictionary determines which terms the network graph highlights and
how nodes are colour-grouped.  You can use a built-in dictionary, edit
one, or create your own from scratch.
"""
from __future__ import annotations

import json

import streamlit as st

from oam_nltk.dictionaries import (
    Dictionary,
    list_dictionaries,
    load_dictionary,
    save_user_dictionary,
)

st.title("Dictionary")
st.markdown(
    "Select a thematic dictionary below.  The dictionary controls which "
    "terms are tracked in the coverage analysis and how nodes are grouped "
    "in the network graph.  You can edit any dictionary or build a new one."
)

names = list_dictionaries()

# --- select ------------------------------------------------------------------
st.subheader("Built-in dictionaries")
st.markdown(
    "Each dictionary ships with categories and keywords tailored to a "
    "domain.  Pick one as a starting point -- you can always edit it."
)

for name in names:
    try:
        d = load_dictionary(name)
        n_terms = len(d.all_terms)
        n_cats = len(d.categories)
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{name}** -- {d.description}")
        col1.caption(f"{n_cats} categories, {n_terms} terms")
        if col2.button("Use", key=f"use_{name}"):
            st.session_state["dictionary"] = d
            st.success(f"Selected: {name}")
    except Exception:
        continue

# --- edit / create -----------------------------------------------------------
st.divider()
st.subheader("Edit or create a dictionary")
st.markdown(
    "Modify the categories and keywords below.  Each text box is one "
    "category -- keywords are separated by commas.  Add a new category "
    "at the bottom.  Press **Save** when done."
)

# Start from selected dictionary or empty
base = st.session_state.get("dictionary", Dictionary(name="custom", categories={}))

edited: dict[str, list[str]] = {}
for cat, terms in base.categories.items():
    txt = st.text_area(
        cat,
        value=", ".join(terms),
        height=70,
        help=f"Comma-separated keywords for the '{cat}' category.",
    )
    parsed = [t.strip() for t in txt.split(",") if t.strip()]
    if parsed:
        edited[cat] = parsed

# New category
st.markdown("**Add a new category**")
new_cat = st.text_input("Category name", placeholder="e.g. Technology")
new_terms = st.text_input(
    "Keywords (comma-separated)",
    placeholder="e.g. digital, AI, blockchain, fintech",
)
if new_cat and new_terms:
    edited[new_cat] = [t.strip() for t in new_terms.split(",") if t.strip()]

save_name = st.text_input("Dictionary name", value=base.name,
                          help="Name used when saving.  Existing names overwrite.")

if st.button("Save and use this dictionary", type="primary"):
    if not edited:
        st.error("Add at least one category with keywords.")
    else:
        new_d = Dictionary(name=save_name, categories=edited,
                           description=f"Custom dictionary: {save_name}")
        save_user_dictionary(new_d)
        st.session_state["dictionary"] = new_d
        st.success(f"Saved and selected '{save_name}' ({len(new_d.all_terms)} terms).")

# --- upload JSON -------------------------------------------------------------
st.divider()
st.subheader("Import from JSON")
st.markdown(
    "Upload a JSON file with the structure "
    '`{"name": "...", "categories": {"Cat1": ["word1", "word2"], ...}}`.'
)
up = st.file_uploader("Dictionary JSON file", type=["json"])
if up is not None:
    try:
        data = json.loads(up.read().decode("utf-8"))
        d = Dictionary.from_dict(data)
        if st.button(f"Import and use '{d.name}'"):
            save_user_dictionary(d)
            st.session_state["dictionary"] = d
            st.success(f"Imported '{d.name}'.")
    except Exception as exc:
        st.error(f"Could not parse JSON: {exc}")

# --- status ------------------------------------------------------------------
st.divider()
if "dictionary" in st.session_state:
    d = st.session_state["dictionary"]
    st.info(f"Active dictionary: **{d.name}** ({len(d.all_terms)} terms).  "
            "Continue to **Network** in the sidebar.")
else:
    st.warning("No dictionary selected yet.  Pick or create one above.")
