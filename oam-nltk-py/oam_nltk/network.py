"""Network Systems Model — the cosmograph.

Builds a NetworkX graph from PMI co-occurrences and renders it as an
interactive pyvis HTML that you can click, drag, zoom and pin.

Five switchable layouts:

  force        Standard force-directed (spring)
  spider       Concentric radial — hubs at the centre
  earth        Nodes placed at geographic coordinates (by country tag)
  radar        Equal-angle sectors grouped by dictionary category
  hierarchical Top-down tree by graph centrality
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)

Layout = Literal["force", "spider", "earth", "radar", "hierarchical"]

ALL_LAYOUTS: list[tuple[str, str]] = [
    ("force",        "Force-directed"),
    ("spider",       "Spider-web (radial)"),
    ("earth",        "Earth (geographic)"),
    ("radar",        "Radar (category sectors)"),
    ("hierarchical", "Hierarchical (tree)"),
]

# Minimal geo lookup for the Earth layout.  Extend freely.
_GEO: dict[str, tuple[float, float]] = {
    "kenya": (-1.29, 36.82), "ethiopia": (9.03, 38.74), "somalia": (2.04, 45.34),
    "uganda": (0.35, 32.58), "tanzania": (-6.79, 39.21), "rwanda": (-1.94, 30.06),
    "burundi": (-3.38, 29.36), "south sudan": (4.85, 31.58), "sudan": (15.5, 32.56),
    "egypt": (30.04, 31.23), "nigeria": (9.08, 7.49), "ghana": (5.56, -0.2),
    "senegal": (14.69, -17.44), "mali": (12.65, -8.0), "morocco": (33.97, -6.85),
    "france": (48.85, 2.35), "germany": (52.52, 13.4), "uk": (51.51, -0.13),
    "united kingdom": (51.51, -0.13), "united states": (38.9, -77.04),
    "usa": (38.9, -77.04), "china": (39.9, 116.41), "india": (28.61, 77.21),
    "brazil": (-15.78, -47.93), "south africa": (-25.75, 28.19),
}


@dataclass
class GraphResult:
    graph: nx.Graph
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    node_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    edge_meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_gexf(self, path: str) -> None:
        nx.write_gexf(self.graph, path)

    def to_pyvis_html(self, height: str = "800px") -> str:
        """Render a drag-and-drop pyvis network as a self-contained HTML string."""
        from pyvis.network import Network

        net = Network(height=height, width="100%",
                      bgcolor="#0e1117", font_color="#fafafa",
                      notebook=False, directed=False,
                      cdn_resources="in_line")
        net.barnes_hut(gravity=-20000, central_gravity=0.3,
                       spring_length=100, spring_strength=0.04, damping=0.9)

        for n, d in self.graph.nodes(data=True):
            x, y = self.positions.get(n, (None, None))
            title = f"<b>{n}</b>"
            for k in ("category", "degree"):
                if k in d:
                    title += f"<br>{k}: {d[k]}"
            net.add_node(
                n, label=n, title=title,
                value=float(d.get("degree", 1)),
                group=str(d.get("category", "")),
                x=None if x is None else float(x) * 600,
                y=None if y is None else float(y) * 600,
                physics=(x is None),
            )
        for u, v, d in self.graph.edges(data=True):
            net.add_edge(u, v,
                         value=float(d.get("weight", 1.0)),
                         title=f"PMI {d.get('pmi', 0):.2f} | count {d.get('count', 0)}")
        return net.generate_html(notebook=False)

    def nodes_csv(self) -> str:
        return self.node_meta.to_csv(index=False)

    def edges_csv(self) -> str:
        return self.edge_meta.to_csv(index=False)


# --- graph construction -------------------------------------------------------

def build_graph(
    cooccurrence: pd.DataFrame,
    *,
    node_metadata: dict[str, dict] | None = None,
    min_pmi: float = 0.0,
    min_count: int = 1,
    top_edges: int | None = 1500,
) -> GraphResult:
    """Build a NetworkX graph from a PMI co-occurrence DataFrame.

    Expected columns: word1, word2, count, pmi.
    """
    if cooccurrence.empty:
        return GraphResult(graph=nx.Graph())

    df = cooccurrence.copy()
    df = df[(df["pmi"] >= min_pmi) & (df["count"] >= min_count)]
    df = df.sort_values("pmi", ascending=False)
    if top_edges:
        df = df.head(top_edges)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row["word1"], row["word2"],
            pmi=float(row["pmi"]),
            count=int(row["count"]),
            weight=max(float(row["pmi"]), 0.01),
        )

    if node_metadata:
        for n in G.nodes:
            for k, v in node_metadata.get(n, {}).items():
                G.nodes[n][k] = v

    for n, deg in G.degree():
        G.nodes[n]["degree"] = int(deg)

    nodes_df = pd.DataFrame([{"node": n, **d} for n, d in G.nodes(data=True)])
    edges_df = pd.DataFrame([{"source": u, "target": v, **d}
                              for u, v, d in G.edges(data=True)])
    return GraphResult(graph=G, node_meta=nodes_df, edge_meta=edges_df)


# --- layouts ------------------------------------------------------------------

def layout_positions(result: GraphResult, layout: Layout = "force",
                     seed: int = 42) -> dict[str, tuple[float, float]]:
    G = result.graph
    if len(G) == 0:
        result.positions = {}
        return result.positions

    if layout == "force":
        pos = nx.spring_layout(G, seed=seed, k=1.2 / math.sqrt(max(len(G), 1)))

    elif layout == "spider":
        degs = dict(G.degree())
        max_d = max(degs.values()) or 1
        nodes_sorted = sorted(G.nodes(), key=lambda n: -degs[n])
        rings = 6
        pos = {}
        for n in nodes_sorted:
            r = 1 - (degs[n] / max_d)
            ring = int(r * (rings - 1))
            members = [m for m in nodes_sorted
                       if int((1 - degs[m] / max_d) * (rings - 1)) == ring]
            k = members.index(n)
            theta = 2 * math.pi * k / max(len(members), 1)
            rr = (ring + 1) / rings
            pos[n] = (rr * math.cos(theta), rr * math.sin(theta))

    elif layout == "earth":
        pos = {}
        for n, d in G.nodes(data=True):
            key = str(d.get("country", n)).lower()
            lat, lon = _GEO.get(key, (None, None))
            if lat is not None:
                pos[n] = (lon / 180.0, lat / 90.0)
            else:
                idx = abs(hash(n)) % 360
                rr = 0.2 + (idx % 20) / 25
                pos[n] = (rr * math.cos(math.radians(idx)),
                          rr * math.sin(math.radians(idx)))

    elif layout == "radar":
        cats: dict[str, list[str]] = {}
        for n, d in G.nodes(data=True):
            cats.setdefault(str(d.get("category", "other")), []).append(n)
        cat_order = sorted(cats)
        n_cats = max(len(cat_order), 1)
        pos = {}
        for i, cat in enumerate(cat_order):
            base = 2 * math.pi * i / n_cats
            for k, n in enumerate(cats[cat]):
                r = 0.3 + 0.7 * (k + 1) / max(len(cats[cat]), 1)
                pos[n] = (r * math.cos(base + k * 0.03),
                          r * math.sin(base + k * 0.03))

    elif layout == "hierarchical":
        root = max(G.degree, key=lambda x: x[1])[0]
        layers: dict[int, list[str]] = {}
        for n, depth in nx.single_source_shortest_path_length(G, root).items():
            layers.setdefault(depth, []).append(n)
        max_depth = max(layers) if layers else 1
        pos = {}
        for depth, nodes in layers.items():
            y = 1 - 2 * depth / max(max_depth, 1)
            for i, n in enumerate(nodes):
                x = (i - len(nodes) / 2) / max(len(nodes), 1) * 2
                pos[n] = (x, y)
        for n in G.nodes:
            pos.setdefault(n, (0.0, 0.0))
    else:
        raise ValueError(f"Unknown layout: {layout}")

    result.positions = {n: (float(x), float(y)) for n, (x, y) in pos.items()}
    return result.positions
