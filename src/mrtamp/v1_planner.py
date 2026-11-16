"""v1 baseline MRTAMP planner: one-robot cheapest path through the sequence graph.

Dijkstra over (product state, mounted tool), weight = the step's cost in the step catalogue
(`mrtamp.step_catalogue`: nominal step time with the penalty cost, plus tool change time); every operation
goes to one robot, and inspections over all robots go where outcomes must be known (see
`mrtamp.inspections`); the operation-plan builder (`mrtamp.operation_plan_builder`) turns the path into the
operation plan. The goal decides the direction: a disassembly follows the graph's removing steps, an
assembly its inverse steps. Ports the idea of the legacy planner search graph, not its code. Stateless:
everything comes from the planning request.
"""

from __future__ import annotations

import functools
import heapq
import itertools
from typing import Mapping

from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from mrtamp.operation_plan_builder import OperationPlanBuilder
from mrtamp.plan_functions import PlanFunction
from mrtamp.planning import (
    GoalUnreachable,
    Planner,
    PlanningRequest,
    PlanResult,
    unreachable_reason,
)
from mrtamp.step_catalogue import StepCatalogue
from mrtamp.step_expansion import expand_step
from mrtamp.step_operations import RobotPlanning


def plan(
    request: PlanningRequest,
    expand=expand_step,
    plan_functions: Mapping[Process, PlanFunction] | None = None,
) -> PlanResult:
    """`expand(step, supply_slots)` turns a step into operations; pass your own to research other expansions.
    Shares get paths from `plan_functions` (nominal times for a process without one)."""
    best, catalogue = None, StepCatalogue(request, expand)
    for robot in request.cell.robots:
        found = _cheapest_path(RobotPlanning(request, robot, expand), catalogue)
        if found is not None and (best is None or found[0] < best[0]):
            best = (found[0], robot, found[1])
    if best is None:
        return GoalUnreachable(unreachable_reason(request))
    _, robot, steps = best
    return _operation_plan(request, robot, steps, expand, plan_functions or {})


def _cheapest_path(planning: RobotPlanning, catalogue: StepCatalogue):
    """Cheapest list of (from, to, step) to a goal state for the robot, with its cost, or None."""
    request, robot = planning.request, planning.robot
    start = request.start_snapshot.product_state_key
    tool = request.start_snapshot.cell_state.mounted_tools.get(robot.id)
    tie = itertools.count()
    queue: list[tuple[float, int, tuple, str | None, list]] = [
        (0.0, next(tie), start, tool, [])
    ]
    settled = set()
    while queue:
        cost, _, state, tool, path = heapq.heappop(queue)
        if request.settings.goal.reached(state):
            return cost, path
        if (state, tool) in settled:
            continue
        settled.add((state, tool))
        for s in catalogue.steps(state, robot, tool):
            heapq.heappush(
                queue,
                (
                    cost + s.cost,
                    next(tie),
                    s.after,
                    s.step.tool,
                    path + [(state, s.after, s.step)],
                ),
            )
    return None


def _operation_plan(
    request: PlanningRequest, robot: Robot, steps, expand, plan_functions
) -> OperationPlan:
    builder = OperationPlanBuilder(request, expand, plan_functions)
    for before, after, step in steps:
        builder.commit(robot, before, after, step)
    return builder.finish()


def make_v1_planner(plan_functions: Mapping[Process, PlanFunction]) -> Planner:
    """The v1 planner with its shares planned by `plan_functions`."""
    return functools.partial(plan, plan_functions=plan_functions)
