"""Select and plot the fastest synchronized-stage disassembly sequence."""

import copy
import math
from numbers import Real
from pathlib import Path

import networkx as nx


def shortest_path_graph(
    graph: nx.MultiDiGraph, root=None, goal=None
) -> nx.MultiDiGraph:
    """Return only selected edges of the cheapest path to any reachable goal.

    Dijkstra minimizes summed action_time. For multiedges the exact cheapest
    edge key is retained. Invalid costs are rejected rather than defaulting to
    unit weights. An explicit goal may be supplied; otherwise is_goal is used.
    """
    if root is None:
        roots = [n for n, d in graph.nodes(data=True) if d.get("depth") == 0]
        if len(roots) != 1:
            raise ValueError("Expected one depth-zero root; pass root explicitly")
        root = roots[0]
    if root not in graph:
        raise nx.NodeNotFound(f"Unknown root: {root!r}")
    goals = (
        [goal]
        if goal is not None
        else [n for n, d in graph.nodes(data=True) if d.get("is_goal")]
    )
    if not goals:
        raise nx.NetworkXNoPath("No goal states are marked in the graph")
    if any(n not in graph for n in goals):
        raise nx.NodeNotFound("Unknown goal")
    for _, _, attrs in graph.edges(data=True):
        cost = attrs.get("action_time")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, Real)
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            raise ValueError("Every edge needs a finite nonnegative action_time")
    distances, paths = nx.single_source_dijkstra(graph, root, weight="action_time")
    reachable = [n for n in goals if n in distances]
    if not reachable:
        raise nx.NetworkXNoPath("No goal is reachable from the initial state")
    best_goal = min(reachable, key=lambda n: (distances[n], repr(n)))
    nodes = paths[best_goal]
    selected = nx.MultiDiGraph()
    selected.graph.update(copy.deepcopy(graph.graph))
    selected.graph.update(
        total_action_time=float(distances[best_goal]),
        selected_goal=copy.deepcopy(best_goal),
        path_nodes=copy.deepcopy(nodes),
        weight="action_time",
    )
    for n in nodes:
        selected.add_node(n, **copy.deepcopy(graph.nodes[n]))
    for stage, (u, v) in enumerate(zip(nodes, nodes[1:]), 1):
        key, attrs = min(
            graph[u][v].items(), key=lambda item: float(item[1]["action_time"])
        )
        selected.add_edge(u, v, key=key, **copy.deepcopy(attrs))
        selected[u][v][key]["stage"] = stage
    return selected


def plot_shortest_path(graph: nx.MultiDiGraph, path: str | Path | None = None):
    """Plot the fastest path as a vertical chain, listing every stage operation.

    Accepts a full graph or the output of shortest_path_graph. Returns a Figure.
    """
    import matplotlib.pyplot as plt

    selected = shortest_path_graph(graph)
    nodes = list(selected.nodes)
    fig, ax = plt.subplots(figsize=(14, max(5, 2.8 * len(nodes))))
    for i, node in enumerate(nodes):
        attrs = selected.nodes[node]
        remaining = attrs.get("n_parts_present", len(node))
        prefix = "START" if i == 0 else "GOAL" if i == len(nodes) - 1 else f"State {i}"
        if len(nodes) == 1:
            prefix = "START / GOAL"
        ax.text(
            0,
            -i * 3,
            f"{prefix}\n{remaining} parts remaining",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#90be6d" if i == len(nodes) - 1 else "#8ecae6",
            ),
            fontsize=11,
        )
        if i == len(nodes) - 1:
            continue
        attrs = next(iter(selected[node][nodes[i + 1]].values()))
        parallel = attrs.get("parallel", False)
        color = "#9b3a9e" if parallel else "#526778"
        operations = attrs.get("operations", [attrs])
        lines = [f"Stage {i + 1}" + (" — PARALLEL" if parallel else "")]
        for operation in operations:
            name = operation.get("target_part", operation.get("label", "Removal"))
            tool = operation.get("tool")
            lines.append(f"Remove {name}" + (f" [{tool}]" if tool else ""))
        lines.append(f"Duration: {attrs['action_time']:.2f} s")
        ax.annotate(
            "",
            xy=(0, -(i + 1) * 3 + 0.45),
            xytext=(0, -i * 3 - 0.45),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
        )
        ax.text(
            0.2,
            -i * 3 - 1.5,
            "\n".join(lines),
            ha="left",
            va="center",
            color=color,
            fontsize=9,
        )
    ax.set_xlim(-1.2, 3.5)
    ax.set_ylim(-(len(nodes) - 1) * 3 - 1, 1)
    ax.set_title(
        f"Shortest disassembly sequence\nTotal synchronized-stage time: {selected.graph['total_action_time']:.2f} s"
    )
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig
