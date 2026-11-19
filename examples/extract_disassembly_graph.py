# examples/extract_disassembly_graph.py
"""
Apply the DisassemblyGraphAgent to a built environment and store the
resulting disassembly graph.

Usage:
    python examples/extract_disassembly_graph.py
    python examples/extract_disassembly_graph.py --collapse-screws   # as the Product explorer extracts it
    python examples/extract_disassembly_graph.py --parallel --robots 2
    python examples/extract_disassembly_graph.py --product cases/angle_grinder/model --parallel --robots 2 --output examples/runs/angle_grinder_parallel

Parallel mode screens proximity, finite centre rays and sampled mesh collisions,
and reports synchronized-stage makespan. It cannot be combined with screw collapsing.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "examples"))

import networkx as nx
from build_env import build_env

from disassembly_pomdp.env_builder import build_env as build_product_env
from disassembly_pomdp.graph_extraction import (
    DisassemblyGraphAgent,
    plot_graph,
    plot_parallel_graph,
    save_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product",
        type=Path,
        help="Product folder containing assembly XML and product_setup.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder for graph.png, graph.json and graph.pkl",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--collapse-screws", action="store_true")
    args = parser.parse_args()
    if args.robots < 1:
        parser.error("--robots must be positive")
    if args.parallel and args.collapse_screws:
        parser.error("parallel mode cannot be combined with --collapse-screws")
    out_dir = args.output or (
        ROOT
        / "examples"
        / "runs"
        / ("parallel_disassembly_graph" if args.parallel else "disassembly_graph")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_product_env(args.product) if args.product is not None else build_env()

    agent = DisassemblyGraphAgent(env)
    graph = agent.build_graph(
        verbose=True,
        collapse_screw_groups=args.collapse_screws,
        parallel=args.parallel,
        n_robots=args.robots,
    )

    save_graph(graph, out_dir / "graph.pkl")
    save_graph(graph, out_dir / "graph.json")
    print(
        f"Saved graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) to {out_dir}"
    )

    try:
        plotter = plot_parallel_graph if args.parallel else plot_graph
        plotter(graph, path=out_dir / "graph.png")
        print(f"Saved plot to {out_dir / 'graph.png'}")
    except Exception as e:
        print(f"Skipping plot (matplotlib/graphviz issue): {e!r}")

    print(
        f"Parallel edges: {sum(bool(d.get('parallel')) for _, _, d in graph.edges(data=True))}"
    )
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
            total = 0.0
            for stage, (u, v) in enumerate(zip(path_nodes[:-1], path_nodes[1:]), 1):
                # MultiDiGraph: pick the cheapest parallel edge for display
                edges = graph.get_edge_data(u, v)
                best = min(
                    edges.values(), key=lambda a: a.get("action_time", float("inf"))
                )
                total += best["action_time"]
                print(f"Stage {stage}:")
                for operation in best.get("operations", [best]):
                    print(
                        f"    Remove {operation['target_part']} [{operation.get('tool')}]"
                    )
                print(f"    Duration: {best['action_time']:.2f} s")
            title = (
                "Total makespan (synchronized stages)"
                if args.parallel
                else "Total sequential time"
            )
            print(f"{title}: {total:.2f} s")
    else:
        print(
            "\nNo goal states found (no target_part_names set on the env, or unreachable)."
        )


if __name__ == "__main__":
    main()
