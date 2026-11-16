"""Interchangeable screws as order-free steps: a group-aware view of the sequence graph, read by the MRTAMP planners.

The export holds each group of interchangeable screws as one chain in one order (by name) and lists the group in
the graph attribute `collapsed_screw_groups`; the export and its format stay as they are (#143, #264). `GroupView`
reads such a group as order-free: within a group a product state is the chain's state with the same number of
group screws plus the set of group screws that are done, so states off the chain are allowed. Once the group's
preconditions hold (the chain's state has a step on a screw of the group), every remaining screw of the group is
a next step, with that screw's own step attributes. All other steps are the chain state's steps, applied to the
state. A state that maps to no chain state, or to two of them, is not in the view.

`GroupView` answers `in` and `out_edges(state, data=True)` as the graph does, so the shared commute test and the
inspection placement (`mrtamp.inspections`) read it in place of the graph; same-group screws then commute. Without
groups it is the graph itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState

from motion_planning.common import PRODUCT_MODEL_UNIT, assembly_of

MIN_SCREW_DISTANCE = (
    0.2  # metres: B1's default for two same-group screws to run at the same time
)


class GroupView:
    """The sequence graph with the screws of its `collapsed_screw_groups` order-free."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.groups = tuple(
            frozenset(g) for g in graph.graph.get("collapsed_screw_groups", ())
        )
        self._group_of = {part: g for g in self.groups for part in g}
        self._chain_state: dict[tuple, tuple[str, ...] | None] = {}
        self._screw_steps: dict[tuple[str, str], dict] = {}
        if not self.groups:
            return
        for node in graph.nodes:
            key = self._key(node)
            self._chain_state[key] = (
                None if key in self._chain_state else node
            )  # two chain states for one key: neither is taken
        for _, _, attrs in graph.edges(data=True):
            if attrs["target_part"] in self._group_of:
                self._screw_steps.setdefault(
                    (attrs["target_part"], attrs["process"]), attrs
                )

    def group_of(self, part: str) -> frozenset[str] | None:
        return self._group_of.get(part)

    def chain_state(self, state: tuple[str, ...]) -> tuple[str, ...] | None:
        """The product state of the graph that `state` stands for: itself if it is a node, else the chain's state
        with the same parts outside the groups and the same number of screws of each group; None if there is none."""
        if state in self.graph:
            return state
        if not self.groups:
            return None
        return self._chain_state.get(self._key(state))

    def __contains__(self, state) -> bool:
        return self.chain_state(tuple(state)) is not None

    def out_edges(self, state, data=True):
        """The steps out of `state` as (state, after, edge attributes), like the graph's `out_edges(..., data=True)`;
        always with the attributes (`data` is there for the graph's signature)."""
        state = tuple(state)
        if not self.groups:
            return list(self.graph.out_edges(state, data=True))
        node = self.chain_state(state)
        if node is None:
            return []
        steps, offered = [], set()
        for _, after, attrs in self.graph.out_edges(node, data=True):
            effect = set(node) ^ set(after)
            group = self._group_of.get(attrs["target_part"])
            if group is not None and effect == {attrs["target_part"]}:
                removing = attrs["target_part"] in node
                for screw in sorted(
                    group & set(state) if removing else group - set(state)
                ):
                    key = (screw, attrs["process"])
                    if key in offered or key not in self._screw_steps:
                        continue
                    offered.add(key)
                    steps.append(
                        (state, _sorted(set(state) ^ {screw}), self._screw_steps[key])
                    )
            elif node == state or not effect & self._group_of.keys():
                steps.append((state, _sorted(set(state) ^ effect), attrs))
            # a step that changes group screws other than one at a time is the chain's own: off the chain it is not offered
        return steps

    def _key(self, state) -> tuple:
        present = set(state)
        return (
            _sorted(p for p in present if p not in self._group_of),
            tuple(len(present & g) for g in self.groups),
        )


def apart(
    first: str,
    second: str,
    positions: Mapping[str, Sequence[float]],
    min_distance: float,
) -> bool:
    """Whether two parts are at least `min_distance` apart; a part without a position (absent, or not a finite
    number) counts as close (B1 also logs such a group screw as an error)."""
    if not all(p in positions and has_position(positions[p]) for p in (first, second)):
        return False
    return math.dist(positions[first], positions[second]) >= min_distance


def mesh_position(mesh) -> tuple[float, ...] | None:
    """The one rule for a part's position, in metres: the mean of its mesh vertices. A part has a position only
    with a non-empty mesh whose vertex mean is a finite number; otherwise None. The Product explorer's check
    and B1's plan-start check both read it (#266)."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return None
    position = vertices.mean(axis=0) * PRODUCT_MODEL_UNIT
    if not np.all(np.isfinite(position)):
        return None
    return tuple(float(x) for x in position)


def has_position(position: Sequence[float]) -> bool:
    """Whether `position` is a position by the rule: finite numbers only."""
    return all(math.isfinite(x) for x in position)


def parts_without_position(assembly) -> list[str]:
    """The parts of `assembly` that have no position by `mesh_position`; a mesh that cannot be read counts too."""
    missing = []
    for name, part in assembly.parts.items():
        try:
            position = mesh_position(part.mesh)
        except (ValueError, OSError):
            position = None
        if position is None:
            missing.append(name)
    return missing


def groups_naming_unknown_parts(
    groups: Sequence[Sequence[str]], part_names: Sequence[str]
) -> list[str]:
    """The screws of the screw groups that are not parts of the product model, sorted."""
    known = set(part_names)
    return sorted({part for group in groups for part in group} - known)


def part_positions(
    product_model_xml: str, model_folder: str | Path
) -> dict[str, tuple[float, ...]]:
    """Each part's position in the product model's frame, in metres (`mesh_position`); a part without one is
    left out. Pass the full product model (an assembly run's start snapshot lacks the parts it inserts)."""
    assembly = assembly_of(
        Cell_Assembly_Snapshot(product_model_xml, CellState()), Path(model_folder)
    )
    positions = {
        name: mesh_position(part.mesh) for name, part in assembly.parts.items()
    }
    return {name: p for name, p in positions.items() if p is not None}


def _sorted(parts) -> tuple[str, ...]:
    return tuple(sorted(parts))
