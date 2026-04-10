"""Command-line interface."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

from . import load_folder, load_dictionary, list_dictionaries, Preprocessor
from .salience import pmi_cooccurrence, dictionary_coverage
from .topics import LDAModel
from .network import build_graph, layout_positions


@click.group()
@click.version_option()
def main() -> None:
    """OAM-NLTK -- Network Systems Modelling of large text bases."""


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--out", type=click.Path(), default="corpus.parquet", show_default=True)
@click.option("--no-ocr", is_flag=True, help="Disable OCR fallback.")
def ingest(folder: str, out: str, no_ocr: bool) -> None:
    """Load documents from FOLDER and save as parquet."""
    df = load_folder(folder, ocr=not no_ocr)
    pp = Preprocessor()
    df = pp.add_to_dataframe(df)
    df.to_parquet(out, index=False)
    click.echo(f"Loaded and preprocessed {len(df)} documents -> {out}")


@main.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.option("--dictionary", "dict_name", default="oam", show_default=True)
@click.option("--topics", "n_topics", type=int, default=8, show_default=True)
@click.option("--out", type=click.Path(), default="results", show_default=True)
def analyse(corpus: str, dict_name: str, n_topics: int, out: str) -> None:
    """Run coverage analysis and LDA on CORPUS."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(corpus)
    dictionary = load_dictionary(dict_name)
    cov = dictionary_coverage(df["tokens"].tolist(), dictionary,
                              doc_ids=df["name"].tolist())
    cov.to_csv(out_dir / "coverage.csv", index=False)
    lda = LDAModel(num_topics=n_topics).fit(df["tokens"].tolist())
    if lda.is_fitted:
        lda.top_words(10).to_csv(out_dir / "topics.csv", index=False)
        lda.doc_topics(df["name"].tolist()).to_csv(out_dir / "doc_topics.csv", index=False)
    click.echo(f"Results saved to {out_dir}/")


@main.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.option("--dictionary", "dict_name", default="oam", show_default=True)
@click.option("--layout", default="force",
              type=click.Choice(["force", "spider", "earth", "radar", "hierarchical"]))
@click.option("--min-pmi", type=float, default=0.5, show_default=True)
@click.option("--min-count", type=int, default=3, show_default=True)
@click.option("--top-edges", type=int, default=1500, show_default=True)
@click.option("--export", type=click.Path(), default="graph.gexf", show_default=True)
@click.option("--html", type=click.Path(), default=None)
def graph(corpus: str, dict_name: str, layout: str, min_pmi: float,
          min_count: int, top_edges: int, export: str, html: str | None) -> None:
    """Build and export a network graph from CORPUS."""
    df = pd.read_parquet(corpus)
    dictionary = load_dictionary(dict_name)
    co = pmi_cooccurrence(df["tokens"].tolist(), window=5, min_count=min_count,
                          vocabulary=dictionary.all_terms, top_n=top_edges)
    node_meta = {t.lower(): {"category": cat}
                 for cat, ts in dictionary.categories.items() for t in ts}
    result = build_graph(co, node_metadata=node_meta, min_pmi=min_pmi,
                         min_count=min_count, top_edges=top_edges)
    layout_positions(result, layout=layout)
    result.to_gexf(export)
    click.echo(f"GEXF -> {export}  ({len(result.graph)} nodes, "
               f"{result.graph.number_of_edges()} edges)")
    if html:
        Path(html).write_text(result.to_pyvis_html(), encoding="utf-8")
        click.echo(f"HTML -> {html}")


@main.command("dicts")
def dicts_cmd() -> None:
    """List available dictionaries."""
    for name in list_dictionaries():
        d = load_dictionary(name)
        click.echo(f"  {name:20s}  {len(d.all_terms):4d} terms  {d.description}")


@main.command()
@click.option("--port", type=int, default=8501, show_default=True)
def dashboard(port: int) -> None:
    """Launch the Streamlit dashboard."""
    home = Path(__file__).resolve().parent.parent / "app" / "Home.py"
    if not home.exists():
        click.echo(f"Cannot find dashboard at {home}", err=True)
        sys.exit(1)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(home),
                    "--server.port", str(port)])


if __name__ == "__main__":
    main()
