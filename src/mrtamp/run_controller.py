"""The run controller carries out one run: plan -> simulate -> read the inspection -> decide.

A run that targets the cell simulation has one segment (every inspection of the plan is read from the one
report). A run that targets the cell goes segment by segment (ADR-0008): each simulation ends its NC at the
first inspection, which the run controller reads; when it found what was expected, the rest of the same plan is
simulated again from the inspected snapshot (re-simulation, the planner is not asked again). An infeasible
operation replans from the snapshot its segment started from, the first unexpected inspection from the
inspected snapshot; the raw feedback of the run so
far goes into each planning request (see `mrtamp.penalties` for the shared penalty helper).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import networkx as nx
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot
from diss_assembly_dataclasses.feedback import Feedback, OperationOutcome
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell, SupplySlot

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_simulation import (
    CellSimulation,
    SimulationReport,
    SimulationRequest,
    cell_target_refusal,
)
from mrtamp.planning import GoalUnreachable, Planner, PlanningRequest, RunSettings

__all__ = ["RunRecord", "RunSettings", "run"]


@dataclass
class RunRecordEntry:
    """One planning round: the feedback the planner got, its answer, the simulation report,
    the inspection read and what the run controller decided."""

    feedback: tuple[Feedback, ...]
    plan_result: OperationPlan | GoalUnreachable
    simulation_report: SimulationReport | None = None
    inspection: Cell_Assembly_Snapshot | None = None
    decision: str = ""
    produced_feedback: tuple[
        Feedback, ...
    ] = ()  # what this round added to the feedback of the run
    resimulation: bool = (
        False  # plan_result is the rest of the plan, sent again after an inspection
    )
    segment: int = 1  # of the plan: a cell-targeted plan runs segment by segment

    def to_dict(self) -> dict:
        plan = self.plan_result
        extra = (
            {"resimulation": True, "segment": self.segment} if self.resimulation else {}
        )
        return {
            **extra,
            "feedback": [f.to_dict() for f in self.feedback],
            "operation_plan": plan.to_dict()
            if isinstance(plan, OperationPlan)
            else None,
            "goal_unreachable": plan.reason
            if isinstance(plan, GoalUnreachable)
            else None,
            "simulation_report": None
            if self.simulation_report is None
            else self.simulation_report.to_dict(),
            "inspection": None
            if self.inspection is None
            else self.inspection.to_dict(),
            "decision": self.decision,
            "produced_feedback": [f.to_dict() for f in self.produced_feedback],
        }


@dataclass
class RunRecord:
    """A copy of the sequence graph (with the provenance of its export, if it came from one), the run
    settings, the warnings about the inputs (e.g. a fingerprint mismatch) and every planning round, in order."""

    sequence_graph: dict
    product_folder: str
    settings: RunSettings
    entries: list[RunRecordEntry] = field(default_factory=list)
    status: str = "running"
    final_snapshot: Cell_Assembly_Snapshot | None = None
    provenance: dict | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def operation_plans(self) -> list[OperationPlan]:
        return [
            e.plan_result
            for e in self.entries
            if isinstance(e.plan_result, OperationPlan) and not e.resimulation
        ]

    @property
    def simulation_reports(self) -> list[SimulationReport]:
        return [
            e.simulation_report for e in self.entries if e.simulation_report is not None
        ]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "settings": self.settings.to_dict(),
            "product_folder": self.product_folder,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
            "sequence_graph": self.sequence_graph,
            "entries": [e.to_dict() for e in self.entries],
            "final_snapshot": None
            if self.final_snapshot is None
            else self.final_snapshot.to_dict(),
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=_numpy_to_list),
            encoding="utf-8",
        )


def run(
    sequence_graph: nx.MultiDiGraph,
    product_model: ProductModel,
    cell: Cell,
    start_snapshot: Cell_Assembly_Snapshot,
    settings: RunSettings,
    planner: Planner,
    cell_simulation: CellSimulation,
    supply_slots: tuple[SupplySlot, ...] = (),
    max_rounds: int = 20,
    provenance: dict | None = None,
    warnings: tuple[str, ...] = (),
    simulation_only: bool = False,
    *,
    travel_times: TravelTimeTable,
) -> RunRecord:
    """Plan, simulate and read the inspection; replan after an infeasible or failed operation.
    `travel_times` is the cell's travel-time table (`CellFile.travel_times`); planners and the cell simulation
    take their times from it.

    Stops when the goal is reached, the planner reports the goal unreachable, an inspection cannot
    be read or `max_rounds` planning rounds have passed (a person decides then). Raises ValueError when the
    start snapshot's product state is not a node of the sequence graph.
    Raises ValueError, before planning, for a run that targets the real cell with a simulation-only cell file
    or with a cell simulation that produces no NC programs (ADR-0008).
    """
    if simulation_only and settings.target == "cell":
        raise ValueError(
            "a run on the real cell cannot use a simulation-only cell file; choose the real cell's cell file"
        )
    refusal = cell_target_refusal(settings.target, cell_simulation)
    if refusal is not None:
        raise ValueError(refusal)
    record = RunRecord(
        sequence_graph=nx.node_link_data(sequence_graph, edges="edges"),
        product_folder=str(product_model.folder),
        settings=settings,
        provenance=provenance,
        warnings=list(warnings),
    )
    if start_snapshot.product_state_key not in sequence_graph:
        raise ValueError(
            f"the start snapshot's product state {list(start_snapshot.product_state_key)} is not a product state of the sequence graph"
        )
    snapshot, feedback = start_snapshot, ()
    for round_number in range(1, max_rounds + 1):
        request = PlanningRequest(
            sequence_graph,
            product_model,
            cell,
            snapshot,
            settings,
            feedback,
            cell_simulation=cell_simulation,
            earlier_plans=tuple(record.operation_plans),
            supply_slots=supply_slots,
            travel_times=travel_times,
        )
        result = planner(request)
        if isinstance(result, OperationPlan) and round_number > 1:
            result = _with_ids_prefixed(
                result, f"r{round_number}_"
            )  # ids stay unique across the run
        entry = RunRecordEntry(feedback=feedback, plan_result=result)
        record.entries.append(entry)
        if isinstance(result, GoalUnreachable):
            return _stop(record, entry, "goal_unreachable")

        segment_plan, segment_start = result, snapshot
        while True:
            report = cell_simulation.simulate(
                SimulationRequest(
                    segment_plan,
                    segment_start,
                    cell,
                    settings.target,
                    product_model=product_model,
                    travel_times=travel_times,
                )
            )
            entry.simulation_report = report
            if not report.ok:
                return _stop(record, entry, "request_rejected")
            entry.produced_feedback = tuple(report.anomalies)
            if report.infeasible is not None:
                entry.produced_feedback = (report.infeasible, *report.anomalies)
                entry.decision = "replan_infeasible_operation"  # the product is unchanged: same snapshot
                feedback += entry.produced_feedback
                snapshot = segment_start
                break

            inspections = [
                op for op in segment_plan.operations if op.process is Process.INSPECTING
            ]
            if settings.target == "cell":
                inspections = inspections[
                    :1
                ]  # the NC of a segment ends at its first inspection
            observed, unexpected = None, False
            for inspection in inspections:
                observed = report.observed_snapshots.get(inspection.id)
                entry.inspection = observed
                if observed is None:
                    return _stop(
                        record, entry, "unexpected_inspection"
                    )  # nothing to replan from
                if observed.product_state_key != inspection.expected_product_state:
                    done = segment_plan.operations[
                        : segment_plan.operations.index(inspection)
                    ]
                    entry.produced_feedback = (
                        *_outcomes_read_off_inspection(done, segment_start, observed),
                        *report.anomalies,
                    )
                    entry.decision = "replan_failed_operation"  # replan from where the product actually is
                    unexpected = True
                    break
            if unexpected:
                feedback += entry.produced_feedback
                snapshot = observed
                break
            if observed is None and entry.resimulation:
                # operations after the last inspection: their NC exists, but nothing observed the goal
                record.final_snapshot = segment_start
                return _stop(record, entry, "goal_not_reached")
            if observed is None:
                return _stop(record, entry, "unexpected_inspection")
            rest = segment_plan.operations[
                segment_plan.operations.index(inspections[-1]) + 1 :
            ]
            if settings.target == "cell" and rest:
                entry.decision = (
                    "resimulate_after_inspection"  # the planner is not asked again
                )
                feedback += entry.produced_feedback
                segment_plan = _rest_of(segment_plan, rest, observed)
                segment_start = observed
                entry = RunRecordEntry(
                    feedback=(),
                    plan_result=segment_plan,
                    resimulation=True,
                    segment=entry.segment + 1,
                )
                record.entries.append(entry)
                continue
            record.final_snapshot = observed
            return _stop(
                record,
                entry,
                "goal_reached"
                if settings.goal.reached(observed.product_state_key)
                else "goal_not_reached",
            )
    return _stop(record, record.entries[-1], "replan_limit_reached")


def _outcomes_read_off_inspection(
    done, start: Cell_Assembly_Snapshot, observed: Cell_Assembly_Snapshot
):
    """Operation outcomes read off an unexpected inspection, given the operations before it: those up
    to the last state the inspection saw succeeded, the next one failed (the first, if it saw none of the states)."""
    operations = [op for op in done if op.process is not Process.INSPECTING]
    states = [start.product_state_key] + [
        op.expected_product_state for op in operations
    ]
    reached = max(
        (i for i, state in enumerate(states) if state == observed.product_state_key),
        default=0,
    )
    outcomes = [
        OperationOutcome(op.id, True, op.expected_product_state)
        for op in operations[:reached]
    ]
    if reached == len(operations):
        return (
            outcomes  # an inspection straight after another one: no operation to blame
        )
    failed = operations[reached]
    outcomes.append(OperationOutcome(failed.id, False, observed.product_state_key))
    return outcomes


def _rest_of(
    plan: OperationPlan, rest, observed: Cell_Assembly_Snapshot
) -> OperationPlan:
    """The request that re-simulates `rest`, the operations after an inspection, from the snapshot it observed:
    same ids, the sequences and the sync points inside `rest` (a sync point whose partner is done is met)."""
    ids = {op.id for op in rest}
    return replace(
        plan,
        operations=tuple(rest),
        sequences={
            r: kept
            for r, seq in plan.sequences.items()
            if (kept := tuple(i for i in seq if i in ids))
        },
        sync_points=tuple(
            s for s in plan.sync_points if s.before in ids and s.until in ids
        ),
        start_snapshot=observed,
    )


def _with_ids_prefixed(plan: OperationPlan, prefix: str) -> OperationPlan:
    return replace(
        plan,
        operations=tuple(replace(op, id=prefix + op.id) for op in plan.operations),
        sequences={
            r: tuple(prefix + i for i in ids) for r, ids in plan.sequences.items()
        },
        sync_points=tuple(
            replace(s, before=prefix + s.before, until=prefix + s.until)
            for s in plan.sync_points
        ),
    )


def _numpy_to_list(value):
    return value.tolist()  # numpy arrays and scalars in graph attributes


def _stop(record: RunRecord, entry: RunRecordEntry, decision: str) -> RunRecord:
    entry.decision = decision
    record.status = decision
    return record
