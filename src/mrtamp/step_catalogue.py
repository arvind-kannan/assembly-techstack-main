"""The step catalogue: the steps of the sequence graph an MRTAMP planner may choose from a product state for a
robot, each with its cost after the feedback of the run so far.

One catalogue per planning request. It hides the run direction (the goal decides which steps may run), step
expansion with the supply slots, the robot's capability, the penalties of the feedback (retry cost, exclusion of
infeasible steps and of failed steps past the retry limit) and the tool change time. A planner may use it or
choose its steps on its own (ADR-0007).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.graph_steps import Step
from mrtamp.nominal_schedule import tool_change_time
from mrtamp.penalties import Penalties, StepKey, penalties
from mrtamp.planning import PlanningRequest
from mrtamp.step_expansion import ExpandedOperation, expand_step
from mrtamp.step_operations import RobotPlanning, can_carry_out


@dataclass(frozen=True)
class AdmissibleStep:
    """A step a robot may carry out from `before`, and its cost: the nominal time with the penalty cost, plus the
    tool change time from the tool mounted before it."""

    before: tuple[str, ...]
    after: tuple[str, ...]
    step: Step
    cost: float


@dataclass(frozen=True)
class StepCatalogue:
    """The admissible steps of one planning request. `expand(step, supply_slots)` is the step expansion (pass the
    planner's own); `graph` is what the steps are read from, the request's sequence graph unless a planner reads it
    through a view with the graph's `out_edges(state, data=True)` (such as `mrtamp.screw_groups.GroupView`)."""

    request: PlanningRequest
    expand: Callable[[Step, tuple], list[ExpandedOperation]] = expand_step
    graph: Any = None
    penalty: Penalties = field(init=False)  # read off the request's feedback

    def __post_init__(self):
        if self.graph is None:
            object.__setattr__(self, "graph", self.request.sequence_graph)
        object.__setattr__(self, "penalty", penalties(self.request))

    def through(self, graph) -> StepCatalogue:
        """The same catalogue, its steps read from `graph` (a view of the sequence graph)."""
        return replace(self, graph=graph)

    def steps(
        self, state: tuple[str, ...], robot: Robot, tool: str | None = None
    ) -> list[AdmissibleStep]:
        """The admissible steps from `state` for `robot` with `tool` mounted, in the graph's order."""
        if state not in self.graph:
            return []
        planning = RobotPlanning(self.request, robot, self.expand)
        found = []
        for _, after, attrs in self.graph.out_edges(state, data=True):
            step = Step.from_edge(attrs)
            if not can_carry_out(planning, state, after, step):
                continue
            key = StepKey(tuple(state), step.target_part, step.process, (robot.id,))
            if self.penalty.banned(key):
                continue
            cost = self.penalty.cost(step.action_time, key) + tool_change_time(
                self.request.travel_times, tool, step.tool
            )
            found.append(AdmissibleStep(tuple(state), tuple(after), step, cost))
        return found
