"""Model-level stand-in for the cell simulation: an adapter of `mrtamp.cell_simulation`, no motion planning, no NC.

It checks only what the model can check (cell state mismatch, unsupported process), times the plan
with nominal times plus tool change constants (an inspection with the cell's inspection time), adds no
motion-level sync points and reports the expected cell-assembly snapshot at each inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import (
    InfeasibleOperation,
    OperationOutcome,
    ReasonCode,
)
from diss_assembly_dataclasses.operation import Operation
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Cell

from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.travel_time_table import TravelTimeTable
from motion_planning.common import PLAN_FAILURE, PLAN_FAILURE_ROBOT
from mrtamp.cell_operations import (
    CELL_OPERATIONS,
    apply_cell_operation,
    cycle_time,
    with_mounted_tools,
)
from mrtamp.cell_simulation import (
    INSPECTION_TIME,
    FeasibilityAnswer,
    SimulationReport,
    SimulationRequest,
    as_group,
    group_problem,
    rejected_answer,
    rejected_report,
    request_problem,
)
from mrtamp.nominal_schedule import nominal_schedule, tool_change_time
from mrtamp.step_expansion import is_gripping_at_supply, update_holds


class ModelLevelStandIn:
    """Reads the product model from each request; an inspection observes the product model at the expected
    product state, so in an assembly run it sees the parts inserted so far."""

    produces_nc_programs = False

    def check_feasibility(
        self,
        concurrent_group,
        snapshot: Cell_Assembly_Snapshot,
        cell: Cell,
        *,
        product_model: ProductModel,
        travel_times: TravelTimeTable,
    ) -> FeasibilityAnswer:
        """Can the operations of a concurrent group (at most one per robot) all be carried out from `snapshot`?

        Checks what the model can check, each operation against the snapshot: cell state mismatch and unsupported
        process, and that no part is handled by two operations at once. Executes nothing, produces no NC.
        """
        concurrent_group = as_group(concurrent_group)
        problem = group_problem(
            concurrent_group, snapshot, cell, product_model, travel_times
        )
        if problem is not None:
            return rejected_answer(problem)
        state = set(snapshot.product_state_key)
        durations, handled = {}, {}
        for op in concurrent_group:
            infeasible = _check(op, _ProductAndCell(state, snapshot.cell_state, cell))
            if (
                infeasible is None
                and op.target is not None
                and handled.setdefault(op.target, op.id) != op.id
            ):
                infeasible = InfeasibleOperation(
                    op.id,
                    op.robots[0],
                    ReasonCode.CELL_STATE_MISMATCH,
                    f"{op.target} is also handled by {handled[op.target]} at the same time",
                )
            if infeasible is not None:
                return FeasibilityAnswer(False, infeasible, durations)
            durations[op.id] = max(
                share.path_duration
                + (
                    tool_change_time(
                        travel_times,
                        snapshot.cell_state.mounted_tools.get(share.robot),
                        share.tool,
                    )
                    if share.tool is not None
                    else 0.0
                )
                for share in _with_cell_times(op, travel_times).shares
            )
        return FeasibilityAnswer(True, None, durations)

    def simulate(self, request: SimulationRequest) -> SimulationReport:
        problem = request_problem(request)
        if problem is not None:
            return rejected_report(problem)
        cell, snapshot = request.cell, request.start_snapshot
        state = set(snapshot.product_state_key)
        cell_state = snapshot.cell_state
        feasible: list[Operation] = []
        cell_state_after: dict[str, CellState] = {}
        product_state_after: dict[str, tuple[str, ...]] = {}
        infeasible = None
        for op in request.operation_plan.operations:
            infeasible = _check(op, _ProductAndCell(state, cell_state, cell))
            if infeasible is not None:
                break
            feasible.append(op)
            cell_state = _after(op, cell_state, cell)
            if (
                op.process is not Process.INSPECTING
                and op.process not in CELL_OPERATIONS
            ):
                state = set(op.expected_product_state)
            cell_state_after[op.id] = cell_state
            # an operation on the cell state alone leaves the product state as it was
            product_state_after[op.id] = (
                tuple(sorted(state))
                if op.process in CELL_OPERATIONS
                else op.expected_product_state
            )

        timeline, metrics, _ = nominal_schedule(
            tuple(_with_cell_times(op, request.travel_times) for op in feasible),
            cell,
            snapshot.cell_state.mounted_tools,
            request.operation_plan.sync_points,
            travel_times=request.travel_times,
        )
        observed, outcomes = {}, []
        for op in feasible:
            if op.process is Process.INSPECTING:
                observed[op.id] = request.product_model.snapshot_at(
                    op.expected_product_state, cell_state_after[op.id]
                )
            else:
                outcomes.append(
                    OperationOutcome(op.id, True, product_state_after[op.id])
                )

        return SimulationReport(
            ok=True,
            feasible_operations=tuple(op.id for op in feasible),
            infeasible=infeasible,
            metrics=metrics,
            timeline=tuple(timeline),
            observed_snapshots=observed,
            outcomes=tuple(outcomes),
        )


def _after(op: Operation, cell_state: CellState, cell: Cell) -> CellState:
    """The cell state after feasible operation `op`: the robots' tools mounted, the holds or the pallets and
    tooling moved."""
    if op.process in CELL_OPERATIONS:
        return apply_cell_operation(op, cell_state, cell)
    holds = dict(cell_state.holds)
    if op.process is not Process.INSPECTING:
        update_holds(op, holds)
    return replace(with_mounted_tools(op, cell_state), holds=holds)


def _with_cell_times(op: Operation, travel_times: TravelTimeTable) -> Operation:
    """`op` with the durations the cell gives it: an inspection lasts the travel-time table's inspection time,
    if it has one; a machine operation its cycle time (see `mrtamp.cell_operations.cycle_time`), during which
    its robots wait (interim, until the cell simulation times it, AT 279)."""
    if op.process is Process.INSPECTING and INSPECTION_TIME in travel_times:
        time = travel_times.get(INSPECTION_TIME)
    elif op.process is Process.REWORK_MILLING:
        time = cycle_time(op, travel_times)
    else:
        time = None
    if time is None:
        return op
    return replace(op, shares=tuple(replace(s, path_duration=time) for s in op.shares))


@dataclass(frozen=True)
class _ProductAndCell:
    """What an operation meets: the parts in the assembly, the cell state and the cell."""

    state: set[str]
    cell_state: CellState
    cell: Cell

    @property
    def holds(self) -> dict[str, str]:
        return self.cell_state.holds


def _check(op: Operation, situation: _ProductAndCell) -> InfeasibleOperation | None:
    """Why `op` cannot be carried out in `situation`, or None. The checks run in order: inspection, plan failure,
    then for an operation on the cell state alone its own checks (`mrtamp.cell_operations`), for any other robot
    capability per share and the part the process needs (see `_MISMATCH_RULES`)."""
    if op.process is Process.INSPECTING:
        return _check_inspection(op, situation.cell)
    if PLAN_FAILURE in op.parameters:
        return InfeasibleOperation(
            op.id,
            op.parameters.get(PLAN_FAILURE_ROBOT, op.shares[0].robot),
            ReasonCode.NO_PATH_SHARE,
            op.parameters[PLAN_FAILURE],
        )
    if op.process in CELL_OPERATIONS:
        after = apply_cell_operation(op, situation.cell_state, situation.cell)
        return after if isinstance(after, InfeasibleOperation) else None
    unsupported = _check_capability(op, situation.cell)
    if unsupported is not None:
        return unsupported
    mismatch = _mismatch(op, situation)
    if mismatch is not None:
        return InfeasibleOperation(
            op.id, op.shares[0].robot, ReasonCode.CELL_STATE_MISMATCH, mismatch
        )
    return None


def _check_inspection(op: Operation, cell: Cell) -> InfeasibleOperation | None:
    robots = {r.id: r for r in cell.robots}
    if not any(
        Process.INSPECTING in robots[s.robot].capabilities
        for s in op.shares
        if s.robot in robots
    ):
        return InfeasibleOperation(
            op.id,
            None,
            ReasonCode.UNSUPPORTED_PROCESS,
            "no robot of the inspection can inspect",
        )
    return None


def _check_capability(op: Operation, cell: Cell) -> InfeasibleOperation | None:
    # a valid request names only robots of the cell (see `mrtamp.cell_simulation.request_problem`)
    robots = {r.id: r for r in cell.robots}
    for share in op.shares:
        if not robots[share.robot].can_carry_out(op.process, share.tool):
            return InfeasibleOperation(
                op.id,
                share.robot,
                ReasonCode.UNSUPPORTED_PROCESS,
                f"{share.robot} cannot carry out {op.process.value} with {share.tool}",
            )
    return None


def _mismatch(op: Operation, situation: _ProductAndCell) -> str | None:
    """Why the cell state does not fit `op`, or None. Gripping at the supply takes a part in from outside, so it
    needs the part absent, like screwing in; any other process needs the part present unless it has no target."""
    if is_gripping_at_supply(op):
        return _must_be_absent(op, situation)
    rule = _MISMATCH_RULES.get(op.process, _must_be_present)
    return rule(op, situation)


def _must_be_absent(op: Operation, situation: _ProductAndCell) -> str | None:
    return (
        f"{op.target} is already in the assembly"
        if op.target in situation.state
        else None
    )


def _must_be_present(op: Operation, situation: _ProductAndCell) -> str | None:
    return (
        f"{op.target} is not in the assembly"
        if op.target is not None and op.target not in situation.state
        else None
    )


def _held_by_robot(op: Operation, situation: _ProductAndCell) -> str | None:
    robot = op.shares[0].robot
    return (
        f"{robot} does not hold {op.target}"
        if situation.holds.get(robot) != op.target
        else None
    )


def _inserted(op: Operation, situation: _ProductAndCell) -> str | None:
    """Inserting takes the part the robot holds, which must not be in the assembly yet."""
    return _held_by_robot(op, situation) or _must_be_absent(op, situation)


_MISMATCH_RULES = {
    Process.SCREWING_IN: _must_be_absent,
    Process.INSERTING: _inserted,
    Process.RELEASING: _held_by_robot,
}
