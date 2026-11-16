"""What the baselines share when they turn a chosen step into operations: whether a robot can carry a step
out, and the operations (with shares planned by the plan functions) that carry it out on one robot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.operation import Operation, Share, StepRef
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.graph_steps import Step
from mrtamp.inspections import PlannedStep
from mrtamp.plan_functions import PlanFunction, fill_shares
from mrtamp.planning import PlanningRequest
from mrtamp.step_expansion import ExpandedOperation, MissingSupplySlot, update_holds


@dataclass(frozen=True)
class RobotPlanning:
    """What planning the steps of one robot needs: the request, the robot, and `expand`, which turns a step into
    operations (`expand_step` unless a planner replaces it)."""

    request: PlanningRequest
    robot: Robot
    expand: Callable[[Step, tuple], list[ExpandedOperation]]


def can_carry_out(
    planning: RobotPlanning, before: tuple, after: tuple, step: Step
) -> bool:
    """The goal decides which steps may run (the graph is never reversed here); the robot needs every operation's process and tool."""
    request = planning.request
    if not request.settings.goal.allows_step(step.inverse, set(before) ^ set(after)):
        return False
    try:
        operations = planning.expand(step, request.supply_slots)
    except MissingSupplySlot:
        return False
    return all(planning.robot.can_carry_out(op.process, step.tool) for op in operations)


def plan_step(
    planning: RobotPlanning,
    before,
    after,
    step: Step,
    first_number: int,
    plan_functions: Mapping[Process, PlanFunction],
    cell_state: CellState,
) -> PlannedStep:
    """The operations of `step` on the robot, numbered `op_<first_number>`..., each share planned from the
    snapshot before the step: the product model at `before` with `cell_state`. Each operation names `step`."""
    request = planning.request
    snapshot = request.product_model.snapshot_at(tuple(before), cell_state)
    robots = {r.id: r for r in request.cell.robots}
    carried_out = StepRef(tuple(before), step.target_part, step.process)
    operations = []
    for number, expanded in enumerate(
        planning.expand(step, request.supply_slots), first_number
    ):
        operation = Operation(
            id=f"op_{number}",
            process=expanded.process,
            target=expanded.target,
            shares=(
                Share(planning.robot.id, step.tool, path_duration=expanded.duration),
            ),
            direction=expanded.direction,
            expected_product_state=tuple(
                after if expanded.reaches_next_state else before
            ),
            parameters=dict(expanded.parameters),
            step=carried_out,
        )
        operations.append(
            fill_shares(
                operation, request.product_model, snapshot, robots, plan_functions
            )
        )
    return PlannedStep(tuple(before), tuple(after), step, tuple(operations))


def cell_state_after(
    cell_state: CellState, operations: Iterable[Operation]
) -> CellState:
    """`cell_state` once `operations` are carried out in order: their tools mounted, their holds updated."""
    holds, mounted = dict(cell_state.holds), dict(cell_state.mounted_tools)
    for op in operations:
        mounted.update({s.robot: s.tool for s in op.shares if s.tool is not None})
        update_holds(op, holds)
    return CellState(holds=holds, mounted_tools=mounted)
