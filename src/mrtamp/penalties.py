"""Shared penalty helper: turns the raw feedback of a run into bans and costs for planning.

Each operation names the step it carries out (`Operation.step`), so its feedback leads back to that step: an
infeasible operation bans its step with that robot assignment from the product state before the step. A failed
operation (an operation outcome that is not *succeeded*) may be retried up to the retry limit at rising cost, then
it is banned too. Anomalies and succeeded outcomes carry no penalty, and so do operations that name no step
(inspections, or operations a planner builds without `mrtamp.step_operations.plan_step` and leaves unnamed).
Planners may use this helper (or the step catalogue, `mrtamp.step_catalogue`, which applies it) or read
`PlanningRequest.feedback` themselves.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from diss_assembly_dataclasses.feedback import InfeasibleOperation, OperationOutcome
from diss_assembly_dataclasses.process import Process

from mrtamp.planning import PlanningRequest


class StepKey(NamedTuple):
    """A step with its robot assignment, from the product state before it."""

    state: tuple[str, ...]
    target: str
    process: Process
    robots: tuple[str, ...]


@dataclass(frozen=True)
class Penalties:
    retry_limit: int
    infeasible: frozenset[StepKey] = frozenset()
    failures: Counter = field(
        default_factory=Counter
    )  # StepKey -> number of failed attempts

    def banned(self, key: StepKey) -> bool:
        return key in self.infeasible or self.failures[key] > self.retry_limit

    def cost(self, base: float, key: StepKey) -> float:
        """The base cost plus one more base cost per earlier failure of this step."""
        return base * (1 + self.failures[key])


def penalties(request: PlanningRequest) -> Penalties:
    infeasible, failures = set(), Counter()
    for item in request.feedback:
        if isinstance(item, InfeasibleOperation):
            key = _step_key(request, item.operation_id)
            if key is not None:
                infeasible.add(key)
        elif isinstance(item, OperationOutcome) and not item.succeeded:
            key = _step_key(request, item.operation_id)
            if key is not None:
                failures[key] += 1
    return Penalties(request.settings.retry_limit, frozenset(infeasible), failures)


def _step_key(request: PlanningRequest, operation_id: str) -> StepKey | None:
    """The step an earlier operation carried out, with its robots; None for an operation that names no step."""
    for plan in request.earlier_plans:
        for op in plan.operations:
            if op.id == operation_id:
                if op.step is None:
                    return None
                return StepKey(
                    op.step.before, op.step.target, op.step.process, op.robots
                )
    return None
