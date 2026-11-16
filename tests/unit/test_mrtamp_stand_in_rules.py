"""The stand-in's feasibility rules, one per process: the part an operation needs in the assembly or in the robot's hand."""

from dataclasses import replace

import pytest
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import ReasonCode
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process

from motion_planning.common import SUPPLY_SLOT
from mrtamp.stand_in import ModelLevelStandIn
from tests.mrtamp_scenarios import (
    assembly_cell,
    base_only_snapshot,
    product_model,
    start_snapshot,
    travel_times,
)


def operation(
    process: Process, target: str | None, parameters: dict | None = None
) -> Operation:
    return Operation(
        "op_1",
        process,
        target,
        (Share("robot_1", None, path_duration=1.0),),
        expected_product_state=("base",),
        parameters=parameters or {},
    )


def check(op: Operation, snapshot: Cell_Assembly_Snapshot):
    return ModelLevelStandIn().check_feasibility(
        (op,),
        snapshot,
        assembly_cell(),
        product_model=product_model(),
        travel_times=travel_times(),
    )


def holding(part: str, snapshot: Cell_Assembly_Snapshot) -> Cell_Assembly_Snapshot:
    return replace(
        snapshot,
        cell_state=CellState(
            holds={"robot_1": part}, mounted_tools=snapshot.cell_state.mounted_tools
        ),
    )


@pytest.mark.parametrize(
    "op, snapshot, text",
    [
        (
            operation(Process.RELEASING, "lid"),
            base_only_snapshot(),
            "robot_1 does not hold lid",
        ),
        (
            operation(Process.INSERTING, "lid"),
            holding("lid", start_snapshot()),
            "lid is already in the assembly",
        ),
        (
            operation(Process.INSERTING, "lid"),
            base_only_snapshot(),
            "robot_1 does not hold lid",
        ),
        (
            operation(Process.SCREWING_IN, "screw"),
            start_snapshot(),
            "screw is already in the assembly",
        ),
        (
            operation(Process.GRIPPING_AND_REMOVING, "lid"),
            base_only_snapshot(),
            "lid is not in the assembly",
        ),
        (
            operation(Process.GRIPPING, "lid", {SUPPLY_SLOT: "lid_slot"}),
            start_snapshot(),
            "lid is already in the assembly",
        ),
    ],
    ids=[
        "release-not-held",
        "insert-already-in",
        "insert-not-held",
        "screw-in-already-in",
        "remove-not-in",
        "grip-at-supply-already-in",
    ],
)
def test_a_part_the_process_needs_in_the_assembly_or_the_hand_must_be_there(
    op, snapshot, text
):
    answer = check(op, snapshot)

    assert not answer.feasible
    assert answer.infeasible.reason_code is ReasonCode.CELL_STATE_MISMATCH
    assert answer.infeasible.text == text


def test_releasing_the_part_the_robot_holds_is_feasible():
    answer = check(
        operation(Process.RELEASING, "lid"), holding("lid", base_only_snapshot())
    )

    assert answer.feasible


def test_gripping_at_the_supply_is_feasible_for_a_part_outside_the_assembly():
    answer = check(
        operation(Process.GRIPPING, "lid", {SUPPLY_SLOT: "lid_slot"}),
        base_only_snapshot(),
    )

    assert answer.feasible
