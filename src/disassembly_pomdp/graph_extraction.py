"""
Disassembly graph extraction agent.

Given a built `DisassemblyPOMDPEnv`, `DisassemblyGraphAgent` performs an
exhaustive breadth-first search over every assembly state reachable via the
env's capabilities, assuming every physically-feasible action succeeds
(EOL stuck/missing sampling and stochastic execution noise are disabled for
the duration of the search -- those model *runtime* uncertainty, not the
structural reachability of the disassembly process).

The extraction depth decides where the search ends: by default
(`DISASSEMBLED_STATES`) a disassembled state, one with every target part
removed (`is_goal`), is not expanded; a `FULL_TEARDOWN` expands every state
until no step removes a part. The graph records it in
`graph.graph["extraction_depth"]`.

The result is an `nx.MultiDiGraph`:
    - Nodes are assembly states (identified by the sorted tuple of part
      names still present) with attributes describing what remains.
    - Edges are individual disassembly actions, carrying the endeffector
      (tool) required, the process (core `Process` value), the action's
      own execution time (toolchange time is *not* baked in, since it
      depends on what tool happened to be mounted before -- see
      `toolchange_time()` below), the capability involved, and the target
      removed.

The agent is applied to an already-constructed env (e.g. `build_env()` in
examples/build_env.py) and only relies on that env's public composition
(capabilities, cell_settings, base_assembly) plus its transition-orchestration
routine, so it works for any assembly/capability configuration built on top
of `DisassemblyPOMDPEnv`.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np
from diss_assembly_dataclasses.process import Process

# Extraction depths: stop at the disassembled states (the default), or tear the product down completely.
DISASSEMBLED_STATES = "disassembled_states"
FULL_TEARDOWN = "full_teardown"
EXTRACTION_DEPTHS = (DISASSEMBLED_STATES, FULL_TEARDOWN)


class ScrewGroupError(RuntimeError):
    """A collapsed screw group is not interchangeable: a screw is blocked where another is free, or another step
    sits between its screws in the sequence. The graph would then be wrong, so the extraction stops."""


def check_extraction_depth(extraction_depth: str) -> str:
    """`extraction_depth` if it is one of `EXTRACTION_DEPTHS`, else ValueError."""
    if extraction_depth not in EXTRACTION_DEPTHS:
        raise ValueError(
            f"unknown extraction depth {extraction_depth!r}; one of {', '.join(EXTRACTION_DEPTHS)}"
        )
    return extraction_depth


class _ForceSuccessRNG:
    """
    Wraps a real np.random.Generator but forces `.random()` to always
    return 0.0, so that `sampled_success = rng.random() < p_succ` is
    guaranteed True whenever p_succ > 0. All other calls (e.g. a planner
    sampling candidate motions) are delegated to the wrapped generator.
    """

    def __init__(self, real_rng: np.random.Generator):
        self._real = real_rng

    def random(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, item):
        return getattr(self._real, item)


def _state_key(assembly) -> tuple[str, ...]:
    return tuple(sorted(p.name for p in assembly.parts.values()))


def _normalize_dirs(dirs) -> list:
    out = []
    for d in dirs:
        d = np.asarray(d, dtype=float)
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            out.append(d / n)
    return out


def _quiet_check_visibility(assembly, part, dirs) -> bool:
    """Assembly.check_visibility() prints a debug line per direction; silence it."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return bool(assembly.check_visibility(part, dirs))


def toolchange_time(
    cell_settings, tool_before: Optional[str], tool_after: Optional[str]
) -> float:
    """
    Cost of switching the mounted tool from `tool_before` to `tool_after`,
    starting and ending at the assembly area (mirrors
    DisassemblyPOMDPEnv._toolchange_prefix's assembly-start case).

    Returns 0.0 if no change is required. Not stored per-edge (it depends
    on path history), but exposed here so consumers of the graph can add
    it back in given whatever tool sequencing they plan on top.
    """
    if tool_after is None or tool_after == tool_before:
        return 0.0
    t = float(getattr(cell_settings, "toolchange_time_assembly_to_storage", 0.0))
    dt = getattr(cell_settings, "toolchange_time_at_toolstorage", 0.0)
    t += float(dt(tool_before, tool_after)) if callable(dt) else float(dt)
    t += float(getattr(cell_settings, "toolchange_time_storage_to_assembly", 0.0))
    return t


class DisassemblyGraphAgent:
    """
    Explores a `DisassemblyPOMDPEnv` and builds the full disassembly graph.

    Usage:
        env = build_env()
        agent = DisassemblyGraphAgent(env)
        graph = agent.build_graph()
        save_graph(graph, "disassembly_graph.pkl")
    """

    def __init__(self, env, seed: int = 0):
        self.env = env
        self.capabilities = list(env.capabilities)
        self.cell_settings = env.cell_settings
        self.base_assembly = copy.deepcopy(env.base_assembly)
        self.target_part_names = list(getattr(env, "target_part_names", []) or [])
        self._all_part_names = [p.name for p in self.base_assembly.parts.values()]

        self.local_dims: list[tuple[int, int]] = [
            (int(sp.nvec[0]), int(sp.nvec[1])) for sp in env._cap_action_spaces
        ]

        self._rng = _ForceSuccessRNG(np.random.default_rng(seed))

        self.graph: Optional[nx.MultiDiGraph] = None
        self.skip_counts: dict[str, int] = {}
        # Populated by build_graph(): node key -> Assembly, for callers (e.g.
        # a rendering/UI step) that need the actual assembly object per state.
        self.assemblies: dict[tuple[str, ...], Any] = {}
        # Screw groups collapsed by the last build_graph(collapse_screw_groups=True).
        self.collapsed_screw_groups: set[frozenset[str]] = set()
        # Why the last _try_action returned None: (reason, details).
        self._last_skip: tuple[str, dict] = ("", {})

    # ------------------------------------------------------------------
    def _bump(self, reason: str, details: Optional[dict] = None) -> None:
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1
        self._last_skip = (reason, dict(details or {}))

    def _node_attrs(self, assembly, depth: Optional[int] = None) -> dict[str, Any]:
        present = sorted(p.name for p in assembly.parts.values())
        removed = sorted(set(self._all_part_names) - set(present))
        attrs: dict[str, Any] = {
            "parts_present": present,
            "parts_removed": removed,
            "n_parts_present": len(present),
        }
        if depth is not None:
            attrs["depth"] = int(depth)
        if self.target_part_names:
            attrs["is_goal"] = all(t not in present for t in self.target_part_names)
        return attrs

    def _try_action(self, assembly, cap_idx: int, cap, sub0: int, sub1: int):
        """
        Attempt one raw local action [sub0, sub1] for capability `cap` from
        `assembly`. Returns (next_assembly, edge_attrs) if it deterministically
        removes a part, else None.
        """
        raw = np.array([sub0, sub1], dtype=np.int64)

        try:
            tref = cap.get_target_ref(assembly, raw)
        except Exception:
            self._bump("get_target_ref_error")
            return None
        if tref is None:
            self._bump("no_target_ref")
            return None
        if getattr(tref, "kind", None) in ("buffer_slot", "part_never_existed"):
            self._bump("buffer_or_never_existed")
            return None

        try:
            if not cap.is_target_present(assembly, tref):
                self._bump("target_missing")
                return None
        except Exception:
            self._bump("is_target_present_error")
            return None

        try:
            semantic_action = cap.decode_action(assembly, raw)
        except Exception:
            self._bump("decode_action_error")
            return None

        tool_name = getattr(semantic_action, "tool_name", None)

        try:
            masked, reason, _details = cap.predecode_mask_reason(
                assembly, raw, current_tool_name=tool_name
            )
        except Exception:
            self._bump("predecode_mask_error")
            return None
        if masked:
            self._bump(f"masked:{reason}", _details)
            return None

        try:
            next_assembly, success, _extra, semantic_action = (
                self.env._transition_orchestrate(cap, assembly, semantic_action)
            )
        except Exception:
            self._bump("transition_orchestrate_error")
            return None

        if not success:
            gate = cap.get_last_exec_gate() or {}
            self._bump(
                "exec_gate_or_infeasible",
                {
                    "reason": gate.get("reason"),
                    **(gate.get("details") or {}),
                },
            )
            return None

        target_name = getattr(semantic_action, "part_name", None) or getattr(
            semantic_action, "screw_part_name", None
        )
        if target_name is None:
            self._bump("no_target_name")
            return None
        if next_assembly.get_part_by_name(target_name) is not None:
            self._bump("target_not_actually_removed")
            return None

        try:
            action_time = float(
                cap.estimate_action_time(
                    assembly, semantic_action, self.cell_settings, tool_before=tool_name
                )
            )
        except Exception:
            action_time = float("nan")

        edge_attrs: dict[str, Any] = {
            "process": cap.process.value,
            "capability": type(cap).__name__,
            "cap_idx": int(cap_idx),
            "raw_action": [int(cap_idx), int(sub0), int(sub1)],
            "tool": tool_name,
            # NB: not "target" -- that key is reserved by nx.node_link_data
            # for the destination node id and would silently get overwritten.
            "target_part": target_name,
            "action_time": action_time,
            "label": f"{type(cap).__name__}:{target_name} [{tool_name}]",
        }
        connection = getattr(semantic_action, "connection", None)
        if connection is not None:
            edge_attrs["screw_head"] = getattr(connection, "screw_head", None)

        # Removal direction (unit vector), normalized by env._transition_orchestrate
        # into operation.data -- used by UI builders to animate the part leaving
        # the assembly along its actual DOF axis.
        op = getattr(semantic_action, "operation", None)
        direction = None
        if op is not None and isinstance(getattr(op, "data", None), dict):
            direction = op.data.get("direction")
        if direction is not None:
            edge_attrs["direction"] = [float(x) for x in direction]

        return next_assembly, edge_attrs

    @staticmethod
    def _decode(
        assembly, cap, sub0: int, sub1: int
    ) -> tuple[Optional[str], Optional[str]]:
        """(target part, tool) of the raw local action [sub0, sub1]; (None, None) if it does not decode."""
        try:
            action = cap.decode_action(assembly, np.array([sub0, sub1], dtype=np.int64))
        except Exception:
            return None, None
        target = getattr(action, "part_name", None) or getattr(
            action, "screw_part_name", None
        )
        return target, getattr(action, "tool_name", None)

    def _screw_group_key(
        self, assembly, cap_idx: int, cap, sub0: int, target: Optional[str]
    ):
        """Unscrewing steps with the same key are interchangeable: same capability and tool, and the screw
        holds the same parts with the same screw head. None for anything else."""
        if target is None or cap.process is not Process.UNSCREWING:
            return None
        holds = sorted(
            (c.screw_head, c.part_name2 if c.part_name1 == target else c.part_name1)
            for c in assembly.get_connections_by_part_name(target)
            if getattr(c, "connection_type", None) == "screw"
        )
        return (cap_idx, sub0, tuple(holds)) if holds else None

    def _local_actions(self):
        """Every raw local action as (cap_idx, cap, sub0, sub1)."""
        for cap_idx, cap in enumerate(self.capabilities):
            n0, n1 = self.local_dims[cap_idx]
            for sub0 in range(n0):
                for sub1 in range(n1):
                    yield cap_idx, cap, sub0, sub1

    def available_steps(
        self, assembly, collapse_screw_groups: bool = False
    ) -> list[tuple[Any, dict]]:
        """Every action that removes a part from `assembly`, as (next_assembly, edge_attrs).

        With `collapse_screw_groups`, interchangeable screws (see `_screw_group_key`) come out in one order
        only: of each group just the first screw by name that can be removed is kept. This assumes the screws
        of a group are removable together, and keeps a product with many screws (41 lid screws: 2^41
        product states) small. Every remaining screw of a group is tried: one that cannot be removed where
        another can raises `ScrewGroupError` (`build_graph` also checks the screws are contiguous).
        """
        steps, groups = [], {}
        for cap_idx, cap, sub0, sub1 in self._local_actions():
            if collapse_screw_groups:
                target, _tool = self._decode(assembly, cap, sub0, sub1)
                key = self._screw_group_key(assembly, cap_idx, cap, sub0, target)
                if key is not None:
                    groups.setdefault(key, []).append((target, sub1))
                    continue
            result = self._try_action(assembly, cap_idx, cap, sub0, sub1)
            if result is not None:
                steps.append(result)
        for (cap_idx, sub0, _holds), members in groups.items():
            first, blocked = None, []
            for target, sub1 in sorted(members):
                result = self._try_action(
                    assembly, cap_idx, self.capabilities[cap_idx], sub0, sub1
                )
                if result is None:
                    blocked.append(target)
                elif first is None:
                    first = result
            if first is None:
                continue
            if blocked:
                raise ScrewGroupError(
                    f"screw {blocked[0]} of the group {sorted(t for t, _ in members)} cannot be removed in "
                    f"the state {_state_key(assembly)} although another screw of it can: they are not "
                    "interchangeable there, so collapsing the group would drop sequences"
                )
            steps.append(first)
            if len(members) > 1:
                self.collapsed_screw_groups.add(frozenset(t for t, _ in members))
        return steps

    @staticmethod
    def _check_screw_groups_contiguous(G: nx.MultiDiGraph, groups) -> None:
        """Raise `ScrewGroupError` if a step that is not one of a group's screws leaves a state where some but
        not all of the group's screws are out: a path would then interleave it with the group's screws."""
        for group in groups:
            for state in G.nodes:
                out = group - set(state)
                if not out or out == group:
                    continue
                for _, _, step in G.out_edges(state, data=True):
                    if step["target_part"] not in group:
                        raise ScrewGroupError(
                            f"{step['target_part']} can be removed in the state {state} between the screws "
                            f"of the group {sorted(group)} (out: {sorted(out)}): the screws of a group must "
                            "be a contiguous block of the sequence"
                        )

    def rejection_reasons(self, assembly, part_name: str) -> list[dict]:
        """Why each action on `part_name` in `assembly` is rejected: capability, tool, reason and details.
        Empty if the part is not present or no action acts on it; an accepted action gives reason None."""
        reasons = []
        with self._deterministic():
            for cap_idx, cap, sub0, sub1 in self._local_actions():
                target, tool = self._decode(assembly, cap, sub0, sub1)
                if target != part_name:
                    continue
                accepted = (
                    self._try_action(assembly, cap_idx, cap, sub0, sub1) is not None
                )
                reason, details = (None, {}) if accepted else self._last_skip
                reasons.append(
                    {
                        "capability": type(cap).__name__,
                        "tool": tool,
                        "reason": reason,
                        "details": _jsonify(details),
                    }
                )
        return reasons

    def replay(self, removed: list[str]):
        """The assembly after removing `removed` from the full product in this order, or None if one of the
        removals is not an available action there. Builds no graph (re-check after an edit)."""
        with self._deterministic():
            assembly = copy.deepcopy(self.base_assembly)
            for name in removed:
                for cap_idx, cap, sub0, sub1 in self._local_actions():
                    if self._decode(assembly, cap, sub0, sub1)[0] != name:
                        continue
                    result = self._try_action(assembly, cap_idx, cap, sub0, sub1)
                    if result is not None:
                        assembly = result[0]
                        break
                else:
                    return None
        return assembly

    def available_actions(self, assembly) -> list[dict]:
        """The actions that remove a part from `assembly`: part, capability, tool and action time."""
        with self._deterministic():
            steps = self.available_steps(assembly)
        return [
            {
                key: attrs[key]
                for key in ("target_part", "capability", "tool", "action_time")
            }
            for _next, attrs in steps
        ]

    @contextmanager
    def _deterministic(self):
        """Disable EOL sampling / stochastic effects while trying actions: every feasible action succeeds."""
        old_eol_op = getattr(self.env, "_eol_op", None)
        old_rng = getattr(self.env, "rng", None)
        self.env._eol_op = None
        self.env.rng = self._rng
        for cap in self.capabilities:
            if hasattr(cap, "_ever_existed"):
                cap._ever_existed = set(self._all_part_names)
        try:
            yield
        finally:
            self.env._eol_op = old_eol_op
            self.env.rng = old_rng

    # ------------------------------------------------------------------
    def build_graph(
        self,
        max_states: Optional[int] = None,
        verbose: bool = True,
        collapse_screw_groups: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        extraction_depth: str = DISASSEMBLED_STATES,
    ) -> nx.MultiDiGraph:
        """
        BFS over reachable assembly states. Returns and stores `self.graph`.

        `collapse_screw_groups`: see `available_steps`; the collapsed groups (largest first seen, as sorted
        part name lists) go to `graph.graph["collapsed_screw_groups"]`, where the order inside a group is free.
        Raises `ScrewGroupError` if a collapsed group's screws are not all removable at a visited state, or if
        another step comes between them (the graph's screws of a group must be one contiguous block).
        `progress(expanded, discovered)` is called after each expanded state.
        `extraction_depth`: `DISASSEMBLED_STATES` does not expand a disassembled state (all target parts
        removed); `FULL_TEARDOWN` expands every state. Without target parts no state is marked, so both give
        the same graph.
        """
        check_extraction_depth(extraction_depth)
        self.collapsed_screw_groups = set()
        with self._deterministic():
            G = nx.MultiDiGraph()
            self.skip_counts = {}

            root_assembly = copy.deepcopy(self.base_assembly)
            root_key = _state_key(root_assembly)
            G.add_node(root_key, **self._node_attrs(root_assembly, depth=0))

            visited: dict[tuple[str, ...], Any] = {root_key: root_assembly}
            frontier: deque = deque([root_key])
            n_expanded = 0

            while frontier:
                if max_states is not None and len(visited) >= max_states:
                    if verbose:
                        print(
                            f"[graph_extraction] hit max_states={max_states}, stopping BFS"
                        )
                    break

                key = frontier.popleft()
                assembly = visited[key]
                if extraction_depth == DISASSEMBLED_STATES and G.nodes[key].get(
                    "is_goal"
                ):
                    continue  # a disassembled state: the disassembly ends here
                depth = G.nodes[key].get("depth", 0)
                n_expanded += 1

                for next_assembly, edge_attrs in self.available_steps(
                    assembly, collapse_screw_groups
                ):
                    next_key = _state_key(next_assembly)
                    if next_key == key:
                        continue
                    if next_key not in visited:
                        visited[next_key] = next_assembly
                        frontier.append(next_key)
                        G.add_node(
                            next_key, **self._node_attrs(next_assembly, depth=depth + 1)
                        )
                    G.add_edge(key, next_key, **edge_attrs)

                if progress is not None:
                    progress(n_expanded, len(visited))
                if verbose and n_expanded % 25 == 0:
                    print(
                        f"[graph_extraction] expanded {n_expanded} states, "
                        f"{len(visited)} discovered, {G.number_of_edges()} edges"
                    )

            if verbose:
                n_goal = sum(1 for _, d in G.nodes(data=True) if d.get("is_goal"))
                print(
                    f"[graph_extraction] done: {len(visited)} states, "
                    f"{G.number_of_edges()} transitions"
                    + (f", {n_goal} goal states" if self.target_part_names else "")
                )
                if self.skip_counts:
                    print(
                        f"[graph_extraction] skipped action attempts by reason: {self.skip_counts}"
                    )

            G.graph["extraction_depth"] = extraction_depth
            if collapse_screw_groups:
                groups = self.collapsed_screw_groups
                largest = [g for g in groups if not any(g < other for other in groups)]
                self._check_screw_groups_contiguous(G, largest)
                G.graph["collapsed_screw_groups"] = sorted(sorted(g) for g in largest)
            self.graph = G
            self.assemblies = visited
            return G

    # ------------------------------------------------------------------
    def compute_inspectable_parts(self, verbose: bool = True) -> None:
        """
        For every discovered node, determine which currently-present parts
        are 'inspectable' -- visible (unoccluded) along at least one of
        their own feasible DOF directions, per `Assembly.check_visibility`.

        This is the same geometric visibility gate the env's capabilities
        already use to mask infeasible actions (the 'geometrically_infeasible'
        pre-decode reason); here it's applied for observation rather than
        action-masking, i.e. "could this part currently be seen/inspected",
        not "can it currently be acted on".

        Must be called after build_graph(). Adds two node attributes:
            inspectable_parts     : sorted list of visible, present part names
            non_inspectable_parts : sorted list of occluded, present part names
        """
        if self.graph is None or not self.assemblies:
            raise RuntimeError("Call build_graph() before compute_inspectable_parts().")

        n = len(self.assemblies)
        for i, (key, assembly) in enumerate(self.assemblies.items()):
            inspectable: list[str] = []
            occluded: list[str] = []
            for part in assembly.parts.values():
                try:
                    dirs = _normalize_dirs(assembly.get_part_dof(part))
                    visible = bool(dirs) and bool(
                        _quiet_check_visibility(assembly, part, dirs)
                    )
                except Exception:
                    visible = False
                (inspectable if visible else occluded).append(part.name)

            self.graph.nodes[key]["inspectable_parts"] = sorted(inspectable)
            self.graph.nodes[key]["non_inspectable_parts"] = sorted(occluded)

            if verbose and (i + 1) % 5 == 0:
                print(f"[graph_extraction] inspectability: {i + 1}/{n} states")

        if verbose:
            print(f"[graph_extraction] inspectability computed for all {n} states")


# ----------------------------------------------------------------------
# Persistence / plotting helpers
# ----------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


def save_graph(graph: nx.MultiDiGraph, path: str | Path) -> None:
    """
    Save a graph built by `DisassemblyGraphAgent.build_graph()`.

    Format is picked from the file extension:
        .pkl / .gpickle -> pickle (preserves node/edge attrs exactly)
        .json           -> node-link JSON (portable, numpy values coerced to plain python)
        .graphml        -> GraphML (portable, node ids and attrs coerced to strings)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix in (".pkl", ".gpickle", ".pickle"):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(graph, f)
        return

    if path.suffix == ".json":
        data = nx.node_link_data(graph, edges="edges")
        data = _jsonify(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    if path.suffix == ".graphml":
        H = nx.MultiDiGraph()
        for node, attrs in graph.nodes(data=True):
            H.add_node(
                "|".join(node) if isinstance(node, tuple) else str(node),
                **{
                    k: json.dumps(_jsonify(v)) if isinstance(v, (list, dict)) else v
                    for k, v in attrs.items()
                },
            )
        for u, v, attrs in graph.edges(data=True):
            u_s = "|".join(u) if isinstance(u, tuple) else str(u)
            v_s = "|".join(v) if isinstance(v, tuple) else str(v)
            H.add_edge(
                u_s,
                v_s,
                **{
                    k: json.dumps(_jsonify(val))
                    if isinstance(val, (list, dict))
                    else val
                    for k, val in attrs.items()
                },
            )
        nx.write_graphml(H, path)
        return

    raise ValueError(f"Unsupported graph file extension: {path.suffix}")


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    path = Path(path)
    if path.suffix in (".pkl", ".gpickle", ".pickle"):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"load_graph currently only supports .pkl (got {path.suffix})")


def plot_graph(graph: nx.MultiDiGraph, path: Optional[str | Path] = None):
    """
    Plot a disassembly graph as a top-down hierarchical DAG.

    - Fully assembled states at the top.
    - Disassembled states at the bottom.
    - Nodes grouped by number of remaining parts.
    - Different removal sequences shown as branches.
    - Goal states highlighted in green.
    - Edges labeled with the removed component.
    """

    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import defaultdict

    # --------------------------------------------------
    # 1. Determine the hierarchy
    # --------------------------------------------------

    # Fully assembled state has the most parts.
    max_parts = max(len(n) for n in graph.nodes())

    # Assign each node to a layer.
    # Layer 0 = fully assembled.
    layers = {}

    for node in graph.nodes():
        layers[node] = max_parts - len(node)

    # --------------------------------------------------
    # 2. Generate hierarchical positions
    # --------------------------------------------------

    try:
        # Graphviz produces a proper layered DAG.
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(
            graph,
            prog="dot",
            args="-Grankdir=TB -Gnodesep=0.8 -Granksep=1.2"
        )

    except Exception:

        # Fallback: manually position nodes by layer.
        grouped = defaultdict(list)

        for node, layer in layers.items():
            grouped[layer].append(node)

        pos = {}

        for layer, nodes in grouped.items():

            n = len(nodes)

            for i, node in enumerate(nodes):

                x = i - (n - 1) / 2
                y = -layer

                pos[node] = (x, y)

    # --------------------------------------------------
    # 3. Node labels
    # --------------------------------------------------

    labels = {}

    for node, data in graph.nodes(data=True):

        n_parts = len(node)

        if layers[node] == 0:
            labels[node] = f"START\n{n_parts} parts"

        elif data.get("is_goal"):
            labels[node] = f"GOAL\n{n_parts} parts"

        else:
            labels[node] = f"{n_parts} parts"

    # --------------------------------------------------
    # 4. Node colors
    # --------------------------------------------------

    node_colors = []

    for node, data in graph.nodes(data=True):

        if layers[node] == 0:
            node_colors.append("#8ecae6")  # Blue: start

        elif data.get("is_goal"):
            node_colors.append("#90be6d")  # Green: goal

        else:
            node_colors.append("#ffffff")  # White: intermediate

    # --------------------------------------------------
    # 5. Draw
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 12))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1400,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.5,
        ax=ax
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        edge_color="#888888",
        width=1.3,
        node_size=1400,
        connectionstyle="arc3,rad=0.05",
        ax=ax
    )

    # --------------------------------------------------
    # 6. Edge labels
    # --------------------------------------------------

    edge_labels = {}

    for u, v, key, data in graph.edges(keys=True, data=True):

        target = data.get("target_part", "")

        if target:
            edge_labels[(u, v)] = target

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#555555",
        rotate=False,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8
        },
        ax=ax
    )

    # --------------------------------------------------
    # 7. Final formatting
    # --------------------------------------------------

    ax.set_title(
        "Disassembly Sequence Graph",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    ax.axis("off")

    plt.tight_layout()

    if path is not None:
        fig.savefig(
            str(path),
            dpi=200,
            bbox_inches="tight"
        )

    return fig
