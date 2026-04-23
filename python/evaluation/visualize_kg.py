"""
CLI utility to visualize a persisted knowledge graph for one PDF.

Usage (from `python/`):
  python -m evaluation.visualize_kg --pdf_id eval_fixture_acme
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterable

import networkx as nx

from config import INDICES_DIR
from retrieval.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_COMMUNITY_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
]


def _community_color(community_id: object) -> str:
    try:
        idx = int(community_id)
    except (TypeError, ValueError):
        idx = 0
    return _COMMUNITY_PALETTE[idx % len(_COMMUNITY_PALETTE)]


def _top_degree_subgraph(graph: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
    if max_nodes <= 0 or graph.number_of_nodes() <= max_nodes:
        return graph.copy()

    ordered_nodes = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    keep = ordered_nodes[:max_nodes]
    return graph.subgraph(keep).copy()


def _truncate_predicates(predicates: Iterable[str], limit: int = 3) -> str:
    preds = [p for p in predicates if p]
    if not preds:
        return ""
    if len(preds) <= limit:
        return " / ".join(preds)
    return " / ".join(preds[:limit]) + f" (+{len(preds) - limit} more)"


def _node_tooltip(name: str, attrs: dict) -> str:
    community_id = attrs.get("community_id", "N/A")
    source_count = len(attrs.get("sources", []) or [])
    return (
        f"Entity: {name}\n"
        f"Community: {community_id}\n"
        f"Source refs: {source_count}"
    )


def build_pyvis_html(graph: nx.DiGraph, out_path: str) -> None:
    from pyvis.network import Network  # Lazy import; optional dependency.

    net = Network(height="850px", width="100%", directed=True, cdn_resources="in_line")
    net.barnes_hut()

    for node, attrs in graph.nodes(data=True):
        net.add_node(
            node,
            label=str(node),
            title=_node_tooltip(str(node), attrs),
            value=max(1, graph.degree(node)),
            color=_community_color(attrs.get("community_id", 0)),
        )

    for u, v, attrs in graph.edges(data=True):
        predicates = attrs.get("predicates", []) or []
        label = _truncate_predicates(predicates)
        edge_kwargs = {"title": label} if label else {}
        net.add_edge(u, v, **edge_kwargs)

    net.write_html(out_path, open_browser=False, notebook=False)


def build_png(graph: nx.DiGraph, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Empty knowledge graph", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    pos = nx.spring_layout(graph, seed=42)
    node_colors = [_community_color(graph.nodes[n].get("community_id", 0)) for n in graph.nodes()]
    node_sizes = [220 + 45 * graph.degree(n) for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.35, arrows=True, arrowsize=10, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    edge_labels = {}
    for u, v, attrs in graph.edges(data=True):
        predicates = attrs.get("predicates", []) or []
        label = _truncate_predicates(predicates)
        if label:
            edge_labels[(u, v)] = label
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, alpha=0.75, ax=ax)

    ax.set_title("Knowledge Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reports",
        "kg",
    )
    parser = argparse.ArgumentParser(description="Render persisted KG as HTML and/or PNG.")
    parser.add_argument("--pdf_id", required=True, help="PDF ID used for persisted graph filename.")
    parser.add_argument(
        "--indices-dir",
        default=INDICES_DIR,
        help="Directory containing persisted {pdf_id}_graph.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "png", "both"],
        default="both",
        help="Which artifact(s) to generate.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Maximum number of top-degree nodes to render.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    graph_path = os.path.join(args.indices_dir, f"{args.pdf_id}_graph.json")
    if not os.path.isfile(graph_path):
        print(f"Graph file not found: {graph_path}", file=sys.stderr)
        return 1

    kg = KnowledgeGraph()
    if not kg.load(graph_path):
        print(f"Failed to load graph file: {graph_path}", file=sys.stderr)
        return 1

    subgraph = _top_degree_subgraph(kg.graph, args.max_nodes)
    os.makedirs(args.output_dir, exist_ok=True)

    html_path = os.path.join(args.output_dir, f"{args.pdf_id}_kg.html")
    png_path = os.path.join(args.output_dir, f"{args.pdf_id}_kg.png")

    output = {
        "html": None,
        "png": None,
        "num_nodes": subgraph.number_of_nodes(),
        "num_edges": subgraph.number_of_edges(),
    }

    try:
        if args.format in ("html", "both"):
            build_pyvis_html(subgraph, html_path)
            output["html"] = html_path
        if args.format in ("png", "both"):
            build_png(subgraph, png_path)
            output["png"] = png_path
    except Exception as exc:
        logger.error(f"Visualization failed: {exc}", exc_info=True)
        print(f"Visualization failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
