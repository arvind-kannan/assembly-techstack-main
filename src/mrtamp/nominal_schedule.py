"""Nominal timing of an operation sequence: share path durations plus the tool change times of the cell's
travel-time table.

Shared by the operation-plan builder (robot clocks, step spans and plan metrics while a planner commits its
steps) and the model-level stand-in (to recompute the metrics).
"""

from __future__ import annotations

from types import SimpleNamespace

from diss_assembly_dataclasses.operation import Operation
from diss_assembly_dataclasses.operation_plan import (
    OperationPlan,
    PlanMetrics,
    SyncPoint,
)
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.graph_extraction import toolchange_time
from disassembly_pomdp.travel_time_table import TravelTimeTable


def tool_change_time(
    travel_times: TravelTimeTable, tool_before: str | None, tool_after: str | None
) -> float:
    return toolchange_time(
        SimpleNamespace(**travel_times.times), tool_before, tool_after
    )


class NominalSchedule:
    """The nominal schedule of an operation sequence, built one operation at a time: an operation starts when all
    its robots are ready. Adding an operation never moves the ones before it.

    A sync point makes its robot wait, before the operation it names, until another operation has ended
    (the robot may change its tool meanwhile); it must be known before that operation is added. A sync point
    whose operation is never added (a plan cut at an infeasible operation) is ignored.
    """

    def __init__(
        self,
        cell: Cell,
        mounted_tools: dict[str, str | None],
        sync_points: tuple[SyncPoint, ...] = (),
        *,
        travel_times: TravelTimeTable,
    ):
        self.cell = cell
        self.travel_times = travel_times
        self.mounted = dict(mounted_tools)
        self.clock = {r.id: 0.0 for r in cell.robots}
        self.busy = {r.id: 0.0 for r in cell.robots}
        self.timeline: list[dict] = []
        self.spans: dict[str, tuple[float, float]] = {}
        self.tool_changes = 0
        self.inspections = 0
        self._waits: dict[tuple[str, str], list[str]] = {}
        for s in sync_points:
            self.wait(s)

    def wait(self, sync_point: SyncPoint) -> None:
        """Add a sync point; a robot may wait for several operations before one."""
        self._waits.setdefault((sync_point.robot, sync_point.before), []).append(
            sync_point.until
        )

    def add(self, op: Operation) -> tuple[float, float]:
        """Schedule `op` after the operations added so far; its start and end."""
        ready = {}
        for share in op.shares:
            t = self.clock[share.robot]
            mounted = self.mounted.get(share.robot)
            if share.tool is not None and share.tool != mounted:
                dt = tool_change_time(self.travel_times, mounted, share.tool)
                self.timeline.append(
                    {
                        "kind": "tool_change",
                        "robot": share.robot,
                        "tool": share.tool,
                        "start": t,
                        "end": t + dt,
                    }
                )
                self.busy[share.robot] += dt
                self.mounted[share.robot] = share.tool
                self.tool_changes += 1
                t += dt
            for until in self._waits.get((share.robot, op.id), ()):
                if until not in self.spans:
                    raise ValueError(
                        f"{share.robot} waits before {op.id} for {until}, which does not end before it"
                    )
                t = max(t, self.spans[until][1])
            ready[share.robot] = t
        start = max(ready.values())
        self.spans[op.id] = (start, start + max(s.path_duration for s in op.shares))
        for share in op.shares:
            end = start + share.path_duration
            self.timeline.append(
                {
                    "kind": "operation",
                    "operation_id": op.id,
                    "robot": share.robot,
                    "start": start,
                    "end": end,
                }
            )
            self.busy[share.robot] += share.path_duration
            self.clock[share.robot] = end
        self.inspections += op.process is Process.INSPECTING
        return self.spans[op.id]

    @property
    def metrics(self) -> PlanMetrics:
        """The plan metrics of the operations added so far."""
        return PlanMetrics(
            makespan=max(self.clock.values(), default=0.0),
            busy_time=dict(self.busy),
            tool_changes=self.tool_changes,
            inspections=self.inspections,
        )


def nominal_schedule(
    operations: tuple[Operation, ...],
    cell: Cell,
    mounted_tools: dict[str, str | None],
    sync_points: tuple[SyncPoint, ...] = (),
    *,
    travel_times: TravelTimeTable,
) -> tuple[list[dict], PlanMetrics, dict[str, str | None]]:
    """Run the operations in order (see `NominalSchedule`).

    Returns the timeline (start/end per robot of each tool change and operation; gaps are waits),
    the plan metrics and the mounted tools at the end.
    """
    schedule = _scheduled(operations, cell, mounted_tools, sync_points, travel_times)
    return schedule.timeline, schedule.metrics, schedule.mounted


def operation_spans(
    plan: OperationPlan, cell: Cell, *, travel_times: TravelTimeTable
) -> dict[str, tuple[float, float]]:
    """The nominal start and end of every operation of `plan` (by id), from its start snapshot with its sync points."""
    return _scheduled(
        plan.operations,
        cell,
        plan.start_snapshot.cell_state.mounted_tools,
        plan.sync_points,
        travel_times,
    ).spans


def _scheduled(
    operations, cell, mounted_tools, sync_points, travel_times
) -> NominalSchedule:
    schedule = NominalSchedule(
        cell, mounted_tools, sync_points, travel_times=travel_times
    )
    for op in operations:
        schedule.add(op)
    return schedule
