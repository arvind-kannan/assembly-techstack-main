# examples/extract_disassembly_graph.py
"""
Apply the DisassemblyGraphAgent to a built environment and store the
resulting disassembly graph.

Usage:
    python examples/extract_disassembly_graph.py
    python examples/extract_disassembly_graph.py --collapse-screws   # as the Product explorer extracts it
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "examples"))

import networkx as nx
from build_env import build_env

from disassembly_pomdp.graph_extraction import (
    DisassemblyGraphAgent,
    plot_graph,
    save_graph,
)


def main() -> None:
    out_dir = ROOT / "examples" / "runs" / "disassembly_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()

    agent = DisassemblyGraphAgent(env)
    graph = agent.build_graph(
        verbose=True, collapse_screw_groups="--collapse-screws" in sys.argv[1:]
    )

    save_graph(graph, out_dir / "graph.pkl")
    save_graph(graph, out_dir / "graph.json")
    print(
        f"Saved graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) to {out_dir}"
    )

    try:
        plot_graph(graph, path=out_dir / "graph.png")
        print(f"Saved plot to {out_dir / 'graph.png'}")
    except Exception as e:
        print(f"Skipping plot (matplotlib/graphviz issue): {e!r}")

    # Print a shortest disassembly sequence (by total action_time) to each goal state.
    root = next(n for n, d in graph.nodes(data=True) if d.get("depth") == 0)
    goal_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_goal")]

    if goal_nodes:
        for goal in goal_nodes[:3]:
            try:
                path_nodes = nx.dijkstra_path(graph, root, goal, weight="action_time")
            except nx.NetworkXNoPath:
                continue
            print(f"\nShortest sequence to goal state ({len(goal)} parts remaining):")
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                # MultiDiGraph: pick the cheapest parallel edge for display
                edges = graph.get_edge_data(u, v)
                best = min(
                    edges.values(), key=lambda a: a.get("action_time", float("inf"))
                )
                print(f"  {best['label']:<40} time={best['action_time']:.2f}s")
    else:
        print(
            "\nNo goal states found (no target_part_names set on the env, or unreachable)."
        )


if __name__ == "__main__":
    main()
