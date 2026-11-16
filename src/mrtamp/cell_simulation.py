"""The cell-simulation seam: what every cell simulation takes and answers, and the rules all of them share.

Two adapters implement it: the model-level stand-in (`mrtamp.stand_in`), which reads the model, and the client of
the crc cell simulation (`mrtamp.sim_client`), which computes. The run controller and the MRTAMP planners depend
only on this module. Request validation and the cell-target rule (ADR-0008) live here, so both adapters reject
the same requests; the contract suite (`tests/contract/test_cell_simulation_contract.py`) pins the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.feedback import (
    Anomaly,
    InfeasibleOperation,
    OperationOutcome,
    ReasonCode,
)
from diss_assembly_dataclasses.operation import Operation
from diss_assembly_dataclasses.operation_plan import (
    OperationPlan,
    PlanMetrics,
    SyncPoint,
)
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable

INSPECTION_TIME = "inspection_time"  # key in the travel-time table: how long an inspection takes, for every adapter


@dataclass(frozen=True)
class SimulationRequest:
    """target: "simulation" or "cell"; execution_mode: "dry" (default) or "physics". `product_model`: the whole
    product, from the planning request; the start snapshot lacks the parts an assembly run inserts later.
    `travel_times`: the cell's travel-time table, from the planning request; until #235 the simulation reads its
    tool-change and inspection times from it (ADR-0012, interim note)."""

    operation_plan: OperationPlan
    start_snapshot: Cell_Assembly_Snapshot
    cell: Cell
    target: str = "simulation"
    execution_mode: str = "dry"
    product_model: ProductModel = field(kw_only=True)
    travel_times: TravelTimeTable = field(kw_only=True)


@dataclass(frozen=True)
class Rejection:
    """Why a request was not carried out at all (`invalid_request`, `internal_error`)."""

    reason_code: ReasonCode
    text: str = ""

    def to_dict(self) -> dict:
        return {"reason_code": self.reason_code.value, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> Rejection:
        return cls(ReasonCode(d["reason_code"]), d.get("text", ""))


@dataclass(frozen=True)
class FeasibilityAnswer:
    """The answer to a feasibility query: feasible, or the infeasible operation with its reason, plus the
    estimated duration of every operation of the concurrent group (tool change included). A malformed
    group is a rejection, as for `simulate`."""

    feasible: bool
    infeasible: InfeasibleOperation | None = None
    durations: dict[str, float] = field(default_factory=dict)
    rejection: Rejection | None = None

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "infeasible": None
            if self.infeasible is None
            else self.infeasible.to_dict(),
            "durations": dict(self.durations),
            "rejection": None if self.rejection is None else self.rejection.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeasibilityAnswer:
        return cls(
            bool(d["feasible"]),
            None
            if d["infeasible"] is None
            else InfeasibleOperation.from_dict(d["infeasible"]),
            {k: float(v) for k, v in d["durations"].items()},
            None if d["rejection"] is None else Rejection.from_dict(d["rejection"]),
        )


@dataclass(frozen=True)
class SimulationReport:
    """ok is False only for a rejected request; an infeasible operation is a report with ok True."""

    ok: bool
    feasible_operations: tuple[str, ...]
    infeasible: InfeasibleOperation | None
    metrics: PlanMetrics
    timeline: tuple[dict, ...] = ()
    sync_points_added: tuple[SyncPoint, ...] = ()
    observed_snapshots: dict[str, Cell_Assembly_Snapshot] = field(default_factory=dict)
    outcomes: tuple[OperationOutcome, ...] = ()
    anomalies: tuple[Anomaly, ...] = ()
    nc_programs: dict[str, dict] | None = (
        None  # channel -> {"file_name", "text"}; only for the cell
    )
    rejection: Rejection | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "feasible_operations": list(self.feasible_operations),
            "infeasible": None
            if self.infeasible is None
            else self.infeasible.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timeline": [dict(t) for t in self.timeline],
            "sync_points_added": [s.to_dict() for s in self.sync_points_added],
            "observed_snapshots": {
                k: s.to_dict() for k, s in self.observed_snapshots.items()
            },
            "outcomes": [o.to_dict() for o in self.outcomes],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "nc_programs": self.nc_programs,
            "rejection": None if self.rejection is None else self.rejection.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SimulationReport:
        return cls(
            ok=bool(d["ok"]),
            feasible_operations=tuple(d["feasible_operations"]),
            infeasible=None
            if d["infeasible"] is None
            else InfeasibleOperation.from_dict(d["infeasible"]),
            metrics=PlanMetrics.from_dict(d["metrics"]),
            timeline=tuple(d["timeline"]),
            sync_points_added=tuple(
                SyncPoint.from_dict(x) for x in d["sync_points_added"]
            ),
            observed_snapshots={
                k: Cell_Assembly_Snapshot.from_dict(v)
                for k, v in d["observed_snapshots"].items()
            },
            outcomes=tuple(OperationOutcome.from_dict(x) for x in d["outcomes"]),
            anomalies=tuple(Anomaly.from_dict(x) for x in d["anomalies"]),
            nc_programs=d["nc_programs"],
            rejection=None
            if d["rejection"] is None
            else Rejection.from_dict(d["rejection"]),
        )


class CellSimulation(Protocol):
    """A cell simulation: `simulate` an operation plan, answer feasibility queries for a concurrent group.

    Both calls get the product model of the planning request (in the simulation request, or as `product_model`):
    the snapshot holds only the parts in the product, not the ones an assembly run inserts next. Both also get
    the cell's travel-time table (in the simulation request, or as `travel_times`).

    `produces_nc_programs`: whether a cell-targeted `simulate` returns the NC programs for the real cell; the
    run controller refuses a cell-targeted run with a simulation that produces none (see `cell_target_refusal`).
    """

    @property
    def produces_nc_programs(self) -> bool: ...

    def simulate(self, request: SimulationRequest) -> SimulationReport: ...

    def check_feasibility(
        self,
        concurrent_group: Iterable[Operation],
        snapshot: Cell_Assembly_Snapshot,
        cell: Cell,
        *,
        product_model: ProductModel,
        travel_times: TravelTimeTable,
    ) -> FeasibilityAnswer: ...


def rejected_report(
    problem: str, reason_code: ReasonCode = ReasonCode.INVALID_REQUEST
) -> SimulationReport:
    """The report of a request that was not carried out at all."""
    return SimulationReport(
        False, (), None, PlanMetrics(0.0), rejection=Rejection(reason_code, problem)
    )


def rejected_answer(
    problem: str, reason_code: ReasonCode = ReasonCode.INVALID_REQUEST
) -> FeasibilityAnswer:
    """The answer to a feasibility query that was not carried out at all."""
    return FeasibilityAnswer(False, rejection=Rejection(reason_code, problem))


def as_group(concurrent_group):
    """A concurrent group as a tuple (a generator is read once); anything not iterable as it is."""
    try:
        return tuple(concurrent_group)
    except TypeError:
        return concurrent_group


def request_problem(request) -> str | None:
    """What is wrong with a malformed simulation request, or None."""
    if not isinstance(request, SimulationRequest):
        return f"expected a SimulationRequest, got {type(request).__name__}"
    for name, kind in (
        ("operation_plan", OperationPlan),
        ("start_snapshot", Cell_Assembly_Snapshot),
        ("cell", Cell),
        ("product_model", ProductModel),
        ("travel_times", TravelTimeTable),
    ):
        if not isinstance(getattr(request, name), kind):
            return f"{name} must be a {kind.__name__}"
    for op in request.operation_plan.operations:
        problem = _invalid_operation(op, request.cell)
        if problem is not None:
            return problem
    ids = [op.id for op in request.operation_plan.operations]
    for sync in request.operation_plan.sync_points:
        if (
            sync.before not in ids
            or sync.until not in ids
            or ids.index(sync.until) >= ids.index(sync.before)
        ):
            return (
                f"sync point of {sync.robot} before {sync.before} waits for {sync.until}, "
                "which is not an earlier operation"
            )
    if request.target not in ("simulation", "cell"):
        return f"unknown target {request.target!r}"
    if request.execution_mode not in ("dry", "physics"):
        return f"unknown execution mode {request.execution_mode!r}"
    return None


def group_problem(group, snapshot, cell, product_model, travel_times) -> str | None:
    """What is wrong with a malformed feasibility query (`group` as `as_group` gives it), or None."""
    if not isinstance(snapshot, Cell_Assembly_Snapshot) or not isinstance(cell, Cell):
        return "snapshot must be a Cell_Assembly_Snapshot and cell a Cell"
    if not isinstance(product_model, ProductModel):
        return "product_model must be a ProductModel"
    if not isinstance(travel_times, TravelTimeTable):
        return "travel_times must be a TravelTimeTable"
    if not isinstance(group, tuple):
        return "a concurrent group is a sequence of operations"
    if not group or not all(isinstance(op, Operation) for op in group):
        return "a concurrent group is a non-empty sequence of operations"
    for op in group:
        problem = _invalid_operation(op, cell)
        if problem is not None:
            return problem
    robots = [r for op in group for r in op.robots]
    if len(robots) != len(set(robots)):
        return "a concurrent group has at most one operation per robot"
    return None


def cell_target_refusal(target: str, cell_simulation: CellSimulation) -> str | None:
    """Why a run with `target` cannot use `cell_simulation`, or None (ADR-0008: NC programs for the real cell
    come only from a cell simulation that produces them)."""
    if target == "cell" and not cell_simulation.produces_nc_programs:
        return (
            f"a run on the real cell needs a cell simulation that produces NC programs; "
            f"{type(cell_simulation).__name__} produces none"
        )
    return None


def _invalid_operation(op, cell: Cell) -> str | None:
    """An operation no cell simulation can be asked about: no share, a robot the cell lacks, or a tool its
    robot cannot mount. A robot that cannot carry out the process is not malformed: that is infeasible."""
    if not isinstance(op, Operation):
        return f"expected an Operation, got {type(op).__name__}"
    if not op.shares:
        return f"operation {op.id} has no shares"
    robots = {r.id: r for r in cell.robots}
    for share in op.shares:
        robot = robots.get(share.robot)
        if robot is None:
            return f"{share.robot} of operation {op.id} is not a robot of the cell"
        if share.tool is not None and share.tool not in robot.tools:
            return f"{share.robot} cannot mount {share.tool} for operation {op.id}"
    return None
