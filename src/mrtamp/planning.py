"""The MRTAMP seam: `plan(planning request) -> operation plan | goal unreachable`, one stateless call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.feedback import Feedback
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell, SupplySlot

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_simulation import CellSimulation


@dataclass(frozen=True)
class Goal:
    """The product state a run must reach: `present` parts must end in the product, `absent` parts must end
    out of it. The goal has no direction of its own: a step's direction follows from the parts it changes
    against the goal (see `allows_step`), so one run may remove and insert parts. An empty goal is rejected."""

    present: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.present and not self.absent:
            raise ValueError(
                "a goal needs at least one part that must be present or absent"
            )
        both = set(self.present) & set(self.absent)
        if both:
            raise ValueError(
                f"parts cannot be both present and absent in one goal: {sorted(both)}"
            )

    def reached(self, product_state: tuple[str, ...]) -> bool:
        """None of the absent parts is in the product and all the present parts are."""
        return not set(self.absent) & set(product_state) and set(self.present) <= set(
            product_state
        )

    def allows_step(self, inverse: bool, effect: set[str]) -> bool:
        """Whether a step of the sequence graph may run in this goal: a removing step (not `inverse`) may
        change parts the goal does not keep present, an inserting step only parts the goal keeps present.
        `effect` is the set of parts the step changes (the symmetric difference of its two product states)."""
        if inverse:
            return effect <= set(self.present)
        return not effect & set(self.present)

    def to_dict(self) -> dict:
        return {"present": list(self.present), "absent": list(self.absent)}


@dataclass(frozen=True)
class RunSettings:
    """What the person chooses for one run beside the product.

    goal: the product state the run must reach (required; the app builds it from the product setup's
    target parts when the person sets none). target: "simulation" or "cell". risky_processes: None means
    all processes that change the product. retry_limit: how often a failed step may be tried again.
    """

    goal: Goal
    target: str = "simulation"
    risky_processes: tuple[Process, ...] | None = None
    retry_limit: int = 1

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(),
            "target": self.target,
            "risky_processes": None
            if self.risky_processes is None
            else [p.value for p in self.risky_processes],
            "retry_limit": self.retry_limit,
        }


@dataclass(frozen=True)
class PlanningRequest:
    """The complete sequence graph and its product model, the cell and its travel-time table, the start
    snapshot, the run settings, the feedback of the run so far (with the earlier operation plans its operation ids refer to)
    and, optionally, a cell simulation for feasibility queries. Every snapshot at a product state is built
    from the product model, so an assembly run plans with the parts inserted so far (ADR-0007)."""

    sequence_graph: nx.MultiDiGraph
    product_model: ProductModel
    cell: Cell
    start_snapshot: Cell_Assembly_Snapshot
    settings: RunSettings
    feedback: tuple[Feedback, ...] = ()
    cell_simulation: CellSimulation | None = None
    supply_slots: tuple[
        SupplySlot, ...
    ] = ()  # poses in the cell frame: mrtamp.supply.slots_in_cell_frame
    earlier_plans: tuple[
        OperationPlan, ...
    ] = ()  # operation ids are unique across them, so feedback resolves
    travel_times: TravelTimeTable = field(kw_only=True)  # the cell's times (ADR-0012)


@dataclass(frozen=True)
class GoalUnreachable:
    """The planner's answer when no operation plan reaches the goal."""

    reason: str

    def to_dict(self) -> dict:
        return {"goal_unreachable": self.reason}


PlanResult = OperationPlan | GoalUnreachable
Planner = Callable[[PlanningRequest], PlanResult]


def unreachable_reason(request: PlanningRequest) -> str:
    """Why no operation plan reaches the goal, for `GoalUnreachable`."""
    goal = request.settings.goal
    return (
        f"no robot of the cell can reach a product state with {list(goal.present)} present and "
        f"{list(goal.absent)} absent from {list(request.start_snapshot.product_state_key)} along the steps "
        f"of the sequence graph"
    )
