"""Analysis views and exports for the Product explorer, without planner reversal."""

from pathlib import Path

import networkx as nx

from disassembly_pomdp.graph_extraction import (
    plot_graph,
    plot_parallel_graph,
    save_graph,
)
from disassembly_pomdp.parallel_diagnostics import plot_rejected_parallel_graph
from disassembly_pomdp.shortest_path import plot_shortest_path, shortest_path_graph


def analysis_view(graph, agent, node_ids, edges):
    try:
        selected = shortest_path_graph(graph)
        inverse = {edge: key for key, edge in edges.items()}
        shortest = {
            "edges": [inverse[edge] for edge in selected.edges(keys=True)],
            "total_time": selected.graph["total_action_time"],
            "error": None,
        }
    except (nx.NetworkXNoPath, ValueError) as error:
        shortest = {"edges": [], "total_time": None, "error": str(error)}
    rejected = []
    diagnostic = getattr(agent, "rejected_parallel_graph", nx.MultiDiGraph())
    for source, _, attrs in diagnostic.edges(data=True):
        state = tuple(diagnostic.nodes[source]["parts_present"])
        rejected.append(
            {"source": node_ids.get(state), "parts_present": list(state), **attrs}
        )
    return {
        "parallel": bool(graph.graph.get("parallel")),
        "n_robots": graph.graph.get("n_robots", 1),
        "collapse_parallel_screws": graph.graph.get("collapse_parallel_screws", False),
        "min_separation": graph.graph.get("min_separation"),
        "shortest": shortest,
        "rejected_attempts": rejected,
    }


def export_analysis(graph, agent, out_dir):
    """Export raw analysis graphs; these do not claim to be robot-planner files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("json", "pkl"):
        save_graph(graph, out / f"graph.{extension}")
    rejected = getattr(agent, "rejected_parallel_graph", nx.MultiDiGraph())
    for extension in ("json", "pkl"):
        save_graph(rejected, out / f"rejected_parallel_graph.{extension}")
    figure = (plot_parallel_graph if graph.graph.get("parallel") else plot_graph)(
        graph, out / "graph.png"
    )
    plt.close(figure)
    for extension in ("png", "svg"):
        figure = plot_rejected_parallel_graph(
            rejected, out / f"rejected_parallel_graph.{extension}"
        )
        plt.close(figure)
    try:
        shortest = shortest_path_graph(graph)
    except (nx.NetworkXNoPath, ValueError):
        # A repeated export must not leave an old successful path behind.
        for extension in ("png", "json", "pkl"):
            (out / f"shortest_path.{extension}").unlink(missing_ok=True)
    else:
        for extension in ("json", "pkl"):
            save_graph(shortest, out / f"shortest_path.{extension}")
        figure = plot_shortest_path(shortest, out / "shortest_path.png")
        plt.close(figure)
    return out
