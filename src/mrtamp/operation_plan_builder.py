"""The operation-plan builder: records the steps an MRTAMP planner chose and finishes them into an operation plan.

The step catalogue answers "what can I do next", the builder records "what I chose". A planner commits a step to a
robot, optionally waiting for a step it committed earlier (an order sync point); the builder turns the step into
operations (step expansion, shares planned by the plan functions from the cell state the steps before it leave),
numbers them in commit order and keeps their nominal schedule, so the planner can ask for robot clocks, the span of a
committed step and the mounted tools. `finish` places the inspections and reports the plan metrics of the finished
operations. A helper next to the MRTAMP seam (ADR-0007): a planner may use it or build its plan on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from diss_assembly_dataclasses.operation_plan import OperationPlan, SyncPoint
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.graph_steps import Step
from mrtamp.inspections import PlannedStep, place_inspections
from mrtamp.nominal_schedule import NominalSchedule, nominal_schedule
from mrtamp.plan_functions import PlanFunction
from mrtamp.planning import PlanningRequest
from mrtamp.step_expansion import ExpandedOperation, expand_step
from mrtamp.step_operations import RobotPlanning, cell_state_after, plan_step


@dataclass(frozen=True)
class CommittedStep:
    """A step committed to a robot, with its nominal start and end (of its first and last operation). Committing
    later steps never moves it."""

    planned: PlannedStep
    robot: str
    start: float
    end: float

    @property
    def effect(self) -> set[str]:
        """The parts the step removes or adds."""
        return set(self.planned.before) ^ set(self.planned.after)


class OperationPlanBuilder:
    """The operation plan of one planning request, built step by step. `expand(step, supply_slots)` is the step
    expansion (pass the planner's own), `plan_functions` plan the shares (nominal times for a process without one),
    and `graph` is what the inspections are placed on: the request's sequence graph unless a planner reads it
    through a view (such as `mrtamp.screw_groups.GroupView`)."""

    def __init__(
        self,
        request: PlanningRequest,
        expand: Callable[[Step, tuple], list[ExpandedOperation]] = expand_step,
        plan_functions: Mapping[Process, PlanFunction] | None = None,
        graph: Any = None,
    ):
        self.request, self.expand = request, expand
        self.plan_functions = plan_functions or {}
        self.graph = request.sequence_graph if graph is None else graph
        start = request.start_snapshot.cell_state
        self._cell_state = start
        self._schedule = NominalSchedule(
            request.cell, start.mounted_tools, travel_times=request.travel_times
        )
        self._steps: list[CommittedStep] = []
        self._sync_points: list[SyncPoint] = []

    def commit(
        self,
        robot: Robot,
        before: tuple[str, ...],
        after: tuple[str, ...],
        step: Step,
        wait_for: CommittedStep | None = None,
    ) -> CommittedStep:
        """Commit `step` (from product state `before` to `after`) to `robot`, after the steps committed so far. With
        `wait_for`, the robot waits before the step until that step has ended (an order sync point)."""
        planned = plan_step(
            RobotPlanning(self.request, robot, self.expand),
            before,
            after,
            step,
            sum(len(s.planned.operations) for s in self._steps) + 1,
            self.plan_functions,
            self._cell_state,
        )
        if wait_for is not None:
            sync_point = SyncPoint(
                robot.id,
                planned.operations[0].id,
                wait_for.planned.operations[-1].id,
                "planner",
                "order",
            )
            self._sync_points.append(sync_point)
            self._schedule.wait(sync_point)
        spans = [self._schedule.add(op) for op in planned.operations]
        self._cell_state = cell_state_after(self._cell_state, planned.operations)
        committed = CommittedStep(planned, robot.id, spans[0][0], spans[-1][1])
        self._steps.append(committed)
        return committed

    @property
    def committed(self) -> tuple[CommittedStep, ...]:
        """The committed steps, in commit order."""
        return tuple(self._steps)

    @property
    def clock(self) -> dict[str, float]:
        """When each robot is free: the end of its last committed operation (0 for a robot without one)."""
        return dict(self._schedule.clock)

    @property
    def mounted_tools(self) -> dict[str, str | None]:
        """The tool each robot has mounted after its committed steps."""
        return dict(self._schedule.mounted)

    def finish(self) -> OperationPlan:
        """The operation plan of the committed steps: inspections placed, the operations per robot, the order sync
        points, and the plan metrics of the nominal schedule of all its operations."""
        request = self.request
        cell, start = request.cell, request.start_snapshot
        operations = place_inspections(
            self.graph,
            [s.planned for s in self._steps],
            request.settings,
            cell,
            start.product_state_key,
            travel_times=request.travel_times,
        )
        sequences = {
            r.id: tuple(op.id for op in operations if r.id in op.robots)
            for r in cell.robots
        }
        sync_points = tuple(self._sync_points)
        _, metrics, _ = nominal_schedule(
            operations,
            cell,
            start.cell_state.mounted_tools,
            sync_points,
            travel_times=request.travel_times,
        )
        return OperationPlan(operations, sequences, sync_points, metrics, start)
