"""Inspection placement: a shared helper that v1 (and later B1) use to put inspections where an outcome must be known.

An inspection goes (1) at the start, when a reference part's pose is needed (a run that targets the
cell), (2) before any step that depends on an uninspected step of a risky process, and (3) at the end.
Steps that do not depend on each other (they commute in the sequence graph) are batched under one
inspection. The risky processes are a run setting; by default all processes that change the product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.graph_steps import Step
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_simulation import INSPECTION_TIME
from mrtamp.planning import RunSettings

PRODUCT_CHANGING = (
    Process.REMOVING,
    Process.GRIPPING_AND_REMOVING,
    Process.UNSCREWING,
    Process.MILLING,
    Process.INSERTING,
    Process.SCREWING_IN,
)


@dataclass(frozen=True)
class PlannedStep:
    """One step of the sequence graph with the operations that carry it out, in plan order."""

    before: tuple[str, ...]
    after: tuple[str, ...]
    step: Step
    operations: tuple[Operation, ...]

    @property
    def process(self) -> Process:
        return self.step.process


def place_inspections(
    graph: nx.MultiDiGraph,
    steps: Sequence[PlannedStep],
    settings: RunSettings,
    cell: Cell,
    start_state: tuple[str, ...],
    first_inspection: int = 1,
    *,
    travel_times: TravelTimeTable,
) -> tuple[Operation, ...]:
    """The operations of `steps` with inspections between them, numbered from `first_inspection`; an
    inspection lasts the table's inspection time."""
    risky = (
        PRODUCT_CHANGING
        if settings.risky_processes is None
        else settings.risky_processes
    )
    operations: list[Operation] = []
    counter = iter(range(first_inspection, 10**6))

    def inspect(state):
        operations.append(
            _inspection(f"inspection_{next(counter)}", state, cell, travel_times)
        )

    if settings.target == "cell":
        inspect(
            start_state
        )  # measures the reference part's pose that places the following operations
    uninspected: list[PlannedStep] = []
    for step in steps:
        if any(not _independent(graph, earlier, step) for earlier in uninspected):
            inspect(step.before)
            uninspected = []
        operations.extend(step.operations)
        if step.process in risky:
            uninspected.append(step)
    inspect(steps[-1].after if steps else start_state)
    return tuple(operations)


def _independent(
    graph: nx.MultiDiGraph, first: PlannedStep, second: PlannedStep
) -> bool:
    """`second` does not depend on `first` when the two steps commute in the sequence graph."""
    return commute(
        graph,
        second.before,
        first.step,
        set(first.before) ^ set(first.after),
        second.step,
        set(second.before) ^ set(second.after),
    )


def commute(
    graph: nx.MultiDiGraph,
    state_before_second: tuple[str, ...],
    first: Step,
    first_effect: set[str],
    second: Step,
    second_effect: set[str],
) -> bool:
    """Whether step `first` (changing the parts in `first_effect`) and step `second` could also have run the
    other way round. From the state as if `first` had not happened, `second` and then `first` must be steps of the
    graph; they end in the same product state as `first` and then `second`. A step is named by target and process."""
    without_first = tuple(sorted(set(state_before_second) ^ first_effect))
    without_first_and_second = tuple(sorted(set(without_first) ^ second_effect))
    with_second_only = tuple(sorted(set(state_before_second) ^ second_effect))
    return _is_step(
        graph, without_first, without_first_and_second, second
    ) and _is_step(graph, without_first_and_second, with_second_only, first)


def _is_step(graph: nx.MultiDiGraph, source: tuple, target: tuple, step: Step) -> bool:
    return (
        any(
            Step.from_edge(attrs) == step
            for _, v, attrs in graph.out_edges(source, data=True)
            if v == target
        )
        if source in graph
        else False
    )


def _inspection(
    operation_id: str,
    state: tuple[str, ...],
    cell: Cell,
    travel_times: TravelTimeTable,
) -> Operation:
    time = travel_times.get(INSPECTION_TIME)
    return Operation(
        id=operation_id,
        process=Process.INSPECTING,
        target=None,
        shares=tuple(Share(r.id, None, path_duration=time) for r in cell.robots),
        expected_product_state=tuple(state),
    )
