# examples/line_balance_disassembly.py
"""
Disassembly line balancing.

Turns the graph produced by `DisassemblyGraphAgent` (see graph_extraction.py /
extract_disassembly_graph.py) into a station assignment for a disassembly
*line* with a fixed number of robots/stations.

Problem setup
-------------
- Input: number of stations K (= number of robots), one per --stations.
- Each station is dedicated to exactly one process/capability type (e.g. a
  station only ever screws, or only ever grips/manipulates) -- mirroring a
  real cell where a robot is fitted with one class of end-effector.
- Tasks are the individual part-removal actions (graph edges), each carrying
  a capability type and an execution time (`action_time`).
- Objective: choose (a) how many of the K stations to dedicate to each
  process type and (b) which tasks go on which station, to minimize cycle
  time (= maximize throughput) while keeping resource utilization high.

Simplifications (stated explicitly, since they matter for interpreting the
result):
- Precedence is taken from a single representative disassembly sequence --
  the cheapest (by total action_time) path through the graph to a chosen
  goal state -- rather than the full multi-path AND/OR precedence graph.
  This matches how extract_disassembly_graph.py already reports "the"
  disassembly sequence.
- Within a process type, tasks have no further ordering constraints on each
  other (only cross-type ordering, implied by the sequence, matters), so
  they can be freely load-balanced across that type's parallel stations.
  This holds for the reference assembly (all screws are removable in any
  order; only downstream gripping and removing tasks depend on them).
- Tool-change overhead between tasks of the same capability (e.g. switching
  screwdriver bits) is not modeled -- `action_time` is each task's own
  execution time (see graph_extraction.toolchange_time's docstring); a
  station is assumed to have whatever tooling its process type needs on hand.
- Cycle time = the slowest station's total assigned time, and throughput =
  1 / cycle_time, the standard assembly-line-balancing assumption of enough
  inter-station buffering that stations don't block each other.

Usage:
    python examples/line_balance_disassembly.py --stations 3
    python examples/line_balance_disassembly.py --stations 4 --goal target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "examples"))

import networkx as nx
from build_env import build_env

from disassembly_pomdp.graph_extraction import (
    DISASSEMBLED_STATES,
    FULL_TEARDOWN,
    DisassemblyGraphAgent,
    save_graph,
)

# Fixed categorical hue order (see dataviz skill's palette.md) -- assigned to
# process types in order of first appearance, never cycled/reused per-task.
_PALETTE_HEX = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
]


def _find_root(graph: nx.MultiDiGraph):
    for n, d in graph.nodes(data=True):
        if d.get("depth") == 0:
            return n
    return next(iter(graph.nodes()))


def _human_type(capability: str) -> str:
    return capability.replace("Capability", "")


def choose_goal_node(graph: nx.MultiDiGraph, mode: str = "full"):
    """
    "full"   -> the cheapest-to-reach, most-fully-disassembled state (fewest
                parts_present) -- balances the whole teardown line.
    "target" -> the cheapest-to-reach `is_goal` state, i.e. the env's own
                target_part_names goal (as extract_disassembly_graph.py uses).
    """
    root = _find_root(graph)
    dist = nx.single_source_dijkstra_path_length(graph, root, weight="action_time")

    if mode == "target":
        goal_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("is_goal") and n in dist
        ]
        if not goal_nodes:
            raise ValueError(
                "No reachable is_goal nodes in graph (env has no target_part_names?); "
                "use --goal full instead."
            )
        return min(goal_nodes, key=lambda n: dist[n])

    if mode == "full":
        min_parts = min(graph.nodes[n]["n_parts_present"] for n in dist)
        candidates = [n for n in dist if graph.nodes[n]["n_parts_present"] == min_parts]
        return min(candidates, key=lambda n: dist[n])

    raise ValueError(f"Unknown goal mode: {mode!r} (expected 'full' or 'target')")


def build_task_sequence(graph: nx.MultiDiGraph, root, goal) -> list[dict]:
    """Cheapest root->goal path, unrolled into an ordered list of task dicts."""
    path_nodes = nx.dijkstra_path(graph, root, goal, weight="action_time")
    tasks = []
    for order_idx, (u, v) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
        edges = graph.get_edge_data(u, v)
        best = min(edges.values(), key=lambda a: a.get("action_time", float("inf")))
        tasks.append(
            {
                "order_idx": order_idx,
                "target_part": best["target_part"],
                "capability": best["capability"],
                "tool": best.get("tool"),
                "action_time": float(best["action_time"]),
            }
        )
    return tasks


# ----------------------------------------------------------------------
# Station allocation
# ----------------------------------------------------------------------


def _lpt_balance(
    times: list[float], n_bins: int
) -> tuple[list[list[int]], list[float]]:
    """
    Longest-processing-time-first greedy multiprocessor scheduling: a
    standard, cheap-to-compute heuristic for balancing loads across
    identical parallel resources (exact balancing is NP-hard; LPT is
    within 4/3 of optimal makespan).
    """
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    for i in sorted(range(len(times)), key=lambda i: -times[i]):
        b = min(range(n_bins), key=lambda k: loads[k])
        bins[b].append(i)
        loads[b] += times[i]
    return bins, loads


def _compositions(total: int, parts: int):
    """All tuples of `parts` positive ints summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def allocate_stations(tasks: list[dict], n_stations: int) -> dict:
    """
    Decide how many of `n_stations` to dedicate to each process type present
    in `tasks`, and which tasks go on which station within a type, so as to
    minimize the resulting cycle time (bottleneck station load).
    """
    type_groups: dict[str, list[dict]] = {}
    for t in tasks:
        type_groups.setdefault(t["capability"], []).append(t)
    types = list(type_groups.keys())  # first-appearance order
    n_types = len(types)

    if n_stations < n_types:
        raise ValueError(
            f"Need at least {n_types} stations (one per process type "
            f"{[_human_type(t) for t in types]}), got {n_stations}."
        )

    times_by_type = {t: [tk["action_time"] for tk in type_groups[t]] for t in types}

    best = None
    for comp in _compositions(n_stations, n_types):
        per_type = {}
        loads_flat: list[float] = []
        for t, n_p in zip(types, comp):
            bins, loads = _lpt_balance(times_by_type[t], n_p)
            per_type[t] = (bins, loads)
            loads_flat.extend(loads)
        max_load = max(loads_flat)
        variance = sum(
            (l - max_load) ** 2 for l in loads_flat
        )  # tie-break: prefer evener spread
        key = (max_load, variance)
        if best is None or key < best[0]:
            best = (key, comp, per_type)

    _, comp, per_type = best

    stations = []
    station_id = 1
    for t in types:
        bins, loads = per_type[t]
        group_tasks = type_groups[t]
        for indices, load in zip(bins, loads):
            assigned = sorted(
                (group_tasks[i] for i in indices), key=lambda tk: tk["order_idx"]
            )
            stations.append(
                {
                    "station_id": station_id,
                    "capability": t,
                    "process": _human_type(t),
                    "tasks": assigned,
                    "total_time": load,
                }
            )
            station_id += 1

    cycle_time = max((s["total_time"] for s in stations), default=0.0)
    total_work = sum(tk["action_time"] for tk in tasks)
    for s in stations:
        s["utilization"] = (s["total_time"] / cycle_time) if cycle_time > 0 else 0.0
        s["idle_time"] = cycle_time - s["total_time"]

    return {
        "n_stations": n_stations,
        "stations_per_type": dict(zip(types, comp)),
        "stations": stations,
        "cycle_time": cycle_time,
        "throughput_per_second": (1.0 / cycle_time) if cycle_time > 0 else float("inf"),
        "total_work": total_work,
        "resource_utilization": (total_work / (n_stations * cycle_time))
        if cycle_time > 0
        else 0.0,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_plan(plan: dict) -> None:
    print(f"\n=== Disassembly line balance ({plan['n_stations']} stations) ===")
    per_type = ", ".join(
        f"{_human_type(t)}={n}" for t, n in plan["stations_per_type"].items()
    )
    print(f"Stations per process type: {per_type}")
    print(f"Cycle time:          {plan['cycle_time']:.2f} s")
    print(
        f"Throughput:          {plan['throughput_per_second'] * 3600:.1f} units/hour "
        f"({plan['throughput_per_second']:.4f} units/s)"
    )
    print(f"Total work content:  {plan['total_work']:.2f} s")
    print(f"Resource utilization: {plan['resource_utilization'] * 100:.1f}%")

    for s in plan["stations"]:
        print(
            f"\nStation {s['station_id']} [{s['process']}] "
            f"-- {s['total_time']:.2f}s busy, {s['idle_time']:.2f}s idle "
            f"({s['utilization'] * 100:.0f}% utilized)"
        )
        for tk in s["tasks"]:
            print(
                f"    #{tk['order_idx']:>2}  {tk['target_part']:<20} "
                f"tool={tk['tool']!s:<6} time={tk['action_time']:.2f}s"
            )


def save_plan(plan: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


# ----------------------------------------------------------------------
# Gantt-style diagram: x = process time, y = station/robot
# ----------------------------------------------------------------------


def plot_line_balance(plan: dict, path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    stations = plan["stations"]
    types = list(
        dict.fromkeys(s["capability"] for s in stations)
    )  # first-appearance order
    color_by_type = {
        t: _PALETTE_HEX[i % len(_PALETTE_HEX)] for i, t in enumerate(types)
    }

    n = len(stations)
    fig_height = max(2.8, 0.6 * n + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    bar_height = 0.6
    y_positions = list(range(n))[::-1]  # station 1 at top

    for y, s in zip(y_positions, stations):
        color = color_by_type[s["capability"]]
        offset = 0.0
        for tk in s["tasks"]:
            width = tk["action_time"]
            ax.barh(
                y,
                width,
                left=offset,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            # Label only if the segment is wide enough to hold the text
            # legibly -- narrow segments skip the inline label rather than
            # clipping it (legend + saved JSON plan carry the rest).
            if width >= plan["cycle_time"] * 0.06:
                ax.text(
                    offset + width / 2,
                    y,
                    tk["target_part"],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    zorder=4,
                )
            offset += width

        # Idle tail + utilization annotation to the right of the bar.
        ax.text(
            plan["cycle_time"] * 1.02,
            y,
            f"{s['utilization'] * 100:.0f}% util",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#52514e",
        )

    ax.axvline(
        plan["cycle_time"], color="#0b0b0b", linestyle="--", linewidth=1.5, zorder=2
    )
    ax.text(
        plan["cycle_time"],
        n - 0.3,
        f"  cycle time = {plan['cycle_time']:.1f}s",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#0b0b0b",
        fontweight="bold",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"Station {s['station_id']} ({s['process']})" for s in stations]
    )
    ax.set_xlabel("Process time (s)", labelpad=8)
    ax.set_xlim(0, plan["cycle_time"] * 1.22)
    ax.set_ylim(-0.7, n - 1 + 0.7)

    ax.grid(axis="x", color="#e1e0d9", linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_by_type[t]) for t in types
    ]
    fig.legend(
        legend_handles,
        [_human_type(t) for t in types],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(types),
        frameon=False,
    )

    subtitle = (
        f"cycle time {plan['cycle_time']:.1f}s  •  "
        f"throughput {plan['throughput_per_second'] * 3600:.1f} units/h  •  "
        f"resource utilization {plan['resource_utilization'] * 100:.0f}%"
    )
    fig.suptitle(
        f"Disassembly line balance -- {plan['n_stations']} stations",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(subtitle, fontsize=9.5, color="#52514e", loc="left")

    fig.subplots_adjust(top=0.86, bottom=0.34, left=0.16, right=0.92)

    if path is not None:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stations",
        "-k",
        type=int,
        default=4,
        help="Number of robots/stations (default: 3)",
    )
    ap.add_argument(
        "--goal",
        choices=["full", "target"],
        default="full",
        help="'full' = balance the whole teardown to minimal remaining parts; "
        "'target' = balance up to the env's target_part_names goal (default: full)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: examples/runs/disassembly_graph)",
    )
    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "examples" / "runs" / "disassembly_graph"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    agent = DisassemblyGraphAgent(env)
    # "full" balances the whole teardown line, which only a full-teardown graph holds
    graph = agent.build_graph(
        verbose=True,
        extraction_depth=FULL_TEARDOWN if args.goal == "full" else DISASSEMBLED_STATES,
    )
    save_graph(graph, out_dir / "graph.pkl")

    root = _find_root(graph)
    goal = choose_goal_node(graph, mode=args.goal)
    tasks = build_task_sequence(graph, root, goal)
    print(
        f"\nDisassembly sequence to goal ({graph.nodes[goal]['n_parts_present']} parts remaining): "
        f"{len(tasks)} tasks, {sum(t['action_time'] for t in tasks):.2f}s total work"
    )

    plan = allocate_stations(tasks, args.stations)
    print_plan(plan)

    save_plan(plan, out_dir / "line_balance_plan.json")
    print(f"\nSaved plan to {out_dir / 'line_balance_plan.json'}")

    plot_line_balance(plan, path=out_dir / "line_balance_gantt.png")
    print(f"Saved diagram to {out_dir / 'line_balance_gantt.png'}")


if __name__ == "__main__":
    main()
