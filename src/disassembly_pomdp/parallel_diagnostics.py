"""Rejected parallel attempts, kept outside the reachable sequence graph."""

import copy
import textwrap
from pathlib import Path

import networkx as nx


def record_rejection(graph, state, operations, reason, details=None):
    """Each failed call gets its own leaf; a leaf is not a reachable product state."""
    state = tuple(state)
    source = "state:" + repr(state)
    if source not in graph:
        graph.add_node(
            source,
            kind="source_state",
            parts_present=list(state),
            n_parts_present=len(state),
        )
    target = f"attempt:{graph.number_of_edges()}"
    names = [op.get("target_part") for op in operations]
    graph.add_node(target, kind="rejected_attempt", target_parts=names, reason=reason)
    graph.add_edge(
        source,
        target,
        rejected=True,
        target_parts=names,
        operations=copy.deepcopy(operations),
        reason=reason,
        details=copy.deepcopy(details or {}),
        label=" + ".join(str(n) for n in names),
    )


def plot_rejected_parallel_graph(
    graph: nx.MultiDiGraph, path: str | Path | None = None
):
    """Plot every rejection as a red dashed branch from its actual source state.

    Width is bounded and repeated states are shared. Very large reports should
    be explored through their JSON export rather than the overview image.
    """
    import matplotlib.pyplot as plt

    attempts = graph.number_of_edges()
    # Keep PNG dimensions bounded; every attempt remains in the graph and SVG.
    fig, ax = plt.subplots(figsize=(18, min(180, max(5, 1.15 * attempts))))
    row = 0
    for source, node in graph.nodes(data=True):
        if node.get("kind") != "source_state":
            continue
        edges = list(graph.out_edges(source, keys=True, data=True))
        middle = row + (len(edges) - 1) / 2
        parts = node["parts_present"]
        source_label = f"{len(parts)} parts remaining\n" + textwrap.fill(
            ", ".join(parts), 40
        )
        ax.text(
            0,
            -middle,
            source_label,
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="#d6eaf8"),
        )
        for _, _, _, attrs in edges:
            label = "Reject: " + attrs["label"] + "\nReason: " + attrs["reason"]
            detail = attrs.get("details", {})
            if "parts" in detail:
                label += "\nConflicting parts: " + ", ".join(detail["parts"])
            ax.annotate(
                "",
                xy=(1.5, -row),
                xytext=(0.5, -middle),
                arrowprops=dict(arrowstyle="-|>", color="#ba3030", linestyle="--"),
            )
            ax.text(
                1.55,
                -row,
                textwrap.fill(label.split("\n")[0], 65)
                + "\n"
                + "\n".join(label.split("\n")[1:]),
                fontsize=7,
                va="center",
                color="#9c2020",
            )
            row += 1
    if not attempts:
        ax.text(
            0.5,
            0.5,
            "No rejected parallel attempts",
            transform=ax.transAxes,
            ha="center",
        )
    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(-max(row, 1), 1)
    ax.set_title(
        f"Rejected parallel attempts ({attempts})\nDashed branches are failures, not reachable transitions"
    )
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
