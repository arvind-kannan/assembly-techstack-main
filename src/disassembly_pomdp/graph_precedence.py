"""The precedence relation of a sequence graph: for a goal, the steps to be done and which must come first.

Ported from Jan's line balancing (`extract_tasks_and_precedence`, `choose_goal_node`, #133) onto the
exported sequence graph. Part p precedes part q if q's step is never taken while p is still present,
on any path from the start state to the goal; pairs with no forced order commute. Line balancing and
MRTAMP planners may use it to order steps; it chooses no path. networkx only, and the input is a graph
object (no pickle loading).

Algorithm by Jan (`line_balance_disassembly.py`, received as a zip in 2026-10); the licence is open, ask Jan
before this code is made public outside the KIT GitLab.

Differences from Jan's code, because our graph is the complete one:

- Only the disassembly direction is read: steps marked `inverse` (added by `complete_by_reversal`) are
  dropped first, so the graph has no cycles and parts do not come back. A contradiction still raises.
- Product states are read from the node attribute `parts_present`, not from the node id.
- A task carries the `process` (e.g. unscrewing), not the capability class name, and no toolpath.
- A part removable by several steps gets the cheapest kind of step (process and tool, by
  `action_time`); the other kinds are dropped, e.g. milling the lid off instead of gripping it.
  The precedence relation is derived only over paths that use the chosen kinds, so every order it
  allows is doable with the tasks returned.
- A NaN `action_time` on a step that could be chosen raises: a solver cannot use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import networkx as nx


class PrecedenceError(ValueError):
    """The sequence graph and goal give no consistent precedence relation."""


@dataclass(frozen=True)
class Task:
    """One step to be done: the removal of one part."""

    target_part: str
    process: str  # e.g. "unscrewing"; not the capability class name
    tool: str | None
    action_time: float
    direction: tuple[float, float, float] | None


def choose_goal_state(graph: nx.MultiDiGraph, mode: Literal["full", "target"] = "full"):
    """The goal product state: the cheapest to reach of the states with fewest parts ("full"),
    or of the `is_goal` states, where all target parts are removed ("target")."""
    if mode not in ("full", "target"):
        raise ValueError(f"unknown goal mode {mode!r} (expected 'full' or 'target')")
    disassembly = _disassembly_direction(graph)
    dist = nx.single_source_dijkstra_path_length(
        disassembly,
        _find_root(disassembly),
        weight=lambda u, _, steps: min(
            _action_time(disassembly, u, step) for step in steps.values()
        ),
    )
    if mode == "target":
        candidates = [n for n in dist if disassembly.nodes[n].get("is_goal")]
        if not candidates:
            raise PrecedenceError(
                "no reachable product state has is_goal set (no target parts?); use goal mode 'full'"
            )
    else:
        fewest = min(len(_parts(disassembly, n)) for n in dist)
        candidates = [n for n in dist if len(_parts(disassembly, n)) == fewest]
    return min(candidates, key=lambda n: dist[n])


def extract_precedence(
    graph: nx.MultiDiGraph, goal
) -> tuple[dict[str, Task], set[tuple[str, str]]]:
    """The tasks (one per part removed between the start state and `goal`) and the precedence relation:
    pairs (p, q) meaning p's step comes before q's step on every path to `goal` that uses the tasks' steps."""
    disassembly = _disassembly_direction(graph)
    root = _find_root(disassembly)
    if goal not in disassembly or not nx.has_path(disassembly, root, goal):
        raise PrecedenceError(
            f"goal state {_parts_list(disassembly, goal)} is not reachable from the start state "
            f"{_parts_list(disassembly, root)} along disassembly steps"
        )
    needed = _parts(disassembly, root) - _parts(disassembly, goal)
    if not needed:
        return {}, set()

    chosen_kind = {
        part: _process_and_tool(_cheapest(disassembly, part_steps))
        for part, part_steps in _steps_by_part(disassembly, root, goal, needed).items()
    }
    chosen = disassembly.edge_subgraph(
        (u, v, k)
        for u, v, k, step in disassembly.edges(keys=True, data=True)
        if _process_and_tool(step) == chosen_kind.get(step["target_part"])
    )
    if root not in chosen or goal not in chosen or not nx.has_path(chosen, root, goal):
        raise PrecedenceError(
            f"the cheapest steps for parts {sorted(needed)} do not reach goal state "
            f"{_parts_list(disassembly, goal)} together"
        )
    steps = _steps_by_part(chosen, root, goal, needed)
    tasks = {
        part: _task(_cheapest(chosen, part_steps)) for part, part_steps in steps.items()
    }

    precedence = {
        (p, q)
        for p in needed
        for q in needed
        if p != q and not any(p in _parts(chosen, source) for source, _ in steps[q])
    }
    contradictions = sorted((p, q) for p, q in precedence if (q, p) in precedence)
    if contradictions:
        raise PrecedenceError(
            f"contradictory precedence pairs {contradictions}: some step brings a part back without being "
            f"marked inverse; check the sequence graph"
        )
    return tasks, precedence


def _disassembly_direction(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    disassembly = graph.copy()
    disassembly.remove_edges_from(
        [
            (u, v, k)
            for u, v, k, inverse in graph.edges(keys=True, data="inverse")
            if inverse
        ]
    )
    return disassembly


def _find_root(graph: nx.MultiDiGraph):
    """The start state: depth 0, or else the state with most parts present."""
    for node, depth in graph.nodes(data="depth"):
        if depth == 0:
            return node
    return max(graph.nodes, key=lambda n: len(_parts(graph, n)))


def _parts(graph: nx.MultiDiGraph, node) -> frozenset[str]:
    try:
        return frozenset(graph.nodes[node]["parts_present"])
    except KeyError:
        raise PrecedenceError(f"product state {node!r} has no parts_present") from None


def _parts_list(graph: nx.MultiDiGraph, node):
    """The parts present in a product state, for messages; the node id itself if it is not in the graph."""
    return sorted(_parts(graph, node)) if node in graph else node


def _steps_by_part(
    graph: nx.MultiDiGraph, root, goal, needed
) -> dict[str, list[tuple[object, dict]]]:
    """For each needed part, its steps on some path from `root` to `goal`, with their source states."""
    on_a_path = (nx.descendants(graph, root) | {root}) & (
        nx.ancestors(graph, goal) | {goal}
    )
    steps: dict[str, list[tuple[object, dict]]] = {p: [] for p in needed}
    for u, v, step in graph.edges(data=True):
        if u in on_a_path and v in on_a_path and step["target_part"] in steps:
            steps[step["target_part"]].append((u, step))
    missing = sorted(p for p, part_steps in steps.items() if not part_steps)
    if missing:
        raise PrecedenceError(
            f"parts {missing} must be removed to reach the goal, but no step on a path removes them"
        )
    return steps


def _cheapest(graph: nx.MultiDiGraph, part_steps: list[tuple[object, dict]]) -> dict:
    """The step with the lowest action_time (the first of equals); every step is checked for NaN."""
    times = [_action_time(graph, source, step) for source, step in part_steps]
    return part_steps[times.index(min(times))][1]


def _action_time(graph: nx.MultiDiGraph, source, step: dict) -> float:
    time = float(step["action_time"])
    if math.isnan(time):
        raise PrecedenceError(
            f"action_time is NaN for the step removing {step['target_part']!r} from product state "
            f"{_parts_list(graph, source)} ({step.get('label', step.get('process'))})"
        )
    return time


def _process_and_tool(step: dict) -> tuple[str, str | None]:
    return step["process"], step.get("tool")


def _task(step: dict) -> Task:
    direction = step.get("direction")
    return Task(
        step["target_part"],
        step["process"],
        step.get("tool"),
        float(step["action_time"]),
        None if direction is None else tuple(float(x) for x in direction),
    )
