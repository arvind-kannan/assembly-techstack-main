"""Plan functions at the v1 planner: shares get real paths, a failing plan function becomes an infeasible operation."""

from dataclasses import replace
from pathlib import Path

import pytest
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.feedback import ReasonCode
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.product_model import ProductModel
from motion_planning.common import PLAN_FAILURE, PLAN_FAILURE_ROBOT, PlanFailure
from mrtamp.plan_functions import fill_shares, removal_plan_functions
from mrtamp.planning import Goal, RunSettings
from mrtamp.run_controller import run
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.v1_planner import make_v1_planner
from tests.mrtamp_scenarios import (
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
)

PATH = ((0.0, 0.0, 10.0), (0.0, 0.0, 5.0), (0.0, 0.0, 30.0))
ROBOTS = {
    "robot_1": Robot("robot_1", 1, tools=("H5",)),
    "robot_2": Robot("robot_2", 2, tools=("H5",)),
}


def two_robot_operation():
    return Operation(
        "op_1",
        Process.GRIPPING_AND_REMOVING,
        "lid",
        (
            Share("robot_1", "H5", path_duration=2.0),
            Share("robot_2", "H5", path_duration=2.0),
        ),
    )


def planned_share(operation, robot, tool, duration):
    share = Share(robot.id, tool, path=PATH, path_duration=duration)
    return replace(operation, shares=(share,), direction=(0.0, 0.0, 1.0))


def run_with(plan_functions):
    return run(
        sequence_graph=lid_screwed_on_base_graph(),
        product_model=product_model(),
        cell=transformer_cell(),
        start_snapshot=start_snapshot(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        planner=make_v1_planner(plan_functions),
        cell_simulation=ModelLevelStandIn(),
        travel_times=travel_times(),
    )


def test_every_share_of_the_plan_gets_the_path_and_duration_of_its_plan_function():
    calls = []

    def plan_function(product_model, snapshot, operation, robot, tool):
        calls.append((operation.process, snapshot.product_state_key, robot.id, tool))
        return planned_share(operation, robot, tool, 2.5)

    record = run_with(
        {
            Process.UNSCREWING: plan_function,
            Process.GRIPPING_AND_REMOVING: plan_function,
        }
    )

    assert record.status == "goal_reached"
    unscrew, _, grip, inspection = record.operation_plans[
        -1
    ].operations  # an inspection sits between the steps
    assert [s.path for op in (unscrew, grip) for s in op.shares] == [PATH, PATH]
    assert grip.shares[0].path_duration == 2.5 and grip.direction == (0.0, 0.0, 1.0)
    assert inspection.shares[0].path == ()  # an inspection has no plan function
    # each plan function sees the assembly as it is before its step, with the robot and tool
    assert calls == [
        (Process.UNSCREWING, ("base", "lid", "screw"), "robot_1", "T6"),
        (Process.GRIPPING_AND_REMOVING, ("base", "lid"), "robot_1", "H5"),
    ]
    # the plan's metrics use the planned durations
    assert (
        record.operation_plans[-1].metrics.busy_time["robot_1"]
        == 2.5 + 2.5 + 160.0 + 20.0
    )  # plus the two inspections


def test_each_plan_function_sees_the_mounted_tools_as_they_are_before_its_step():
    mounted = []

    def plan_function(product_model, snapshot, operation, robot, tool):
        mounted.append(snapshot.cell_state.mounted_tools["robot_1"])
        return planned_share(operation, robot, tool, 2.5)

    run_with(
        {
            Process.UNSCREWING: plan_function,
            Process.GRIPPING_AND_REMOVING: plan_function,
        }
    )

    # robot_1 starts with H5 and has T6 mounted after unscrewing, before gripping and removing the lid
    assert mounted == ["H5", "T6"]


def test_a_programming_error_in_a_plan_function_propagates_instead_of_looking_infeasible():
    def broken(product_model, snapshot, operation, robot, tool):
        return {}["no such robot"]

    with pytest.raises(KeyError):
        fill_shares(
            two_robot_operation(),
            product_model(),
            start_snapshot(),
            ROBOTS,
            {Process.GRIPPING_AND_REMOVING: broken},
        )


def test_a_plan_failure_marks_the_operation_with_the_robot_of_the_failing_share():
    def fails_for_robot_2(product_model, snapshot, operation, robot, tool):
        if robot.id == "robot_2":
            raise PlanFailure("no grip point for robot_2")
        return operation

    planned = fill_shares(
        two_robot_operation(),
        product_model(),
        start_snapshot(),
        ROBOTS,
        {Process.GRIPPING_AND_REMOVING: fails_for_robot_2},
    )

    assert "no grip point for robot_2" in planned.parameters[PLAN_FAILURE]
    assert planned.parameters[PLAN_FAILURE_ROBOT] == "robot_2"


@pytest.mark.geom
def test_the_removal_plan_functions_raise_a_plan_failure_when_they_find_no_plan():
    product = (
        Path(__file__).resolve().parents[2]
        / "benchmark_cases"
        / "benchmark_bolt_and_nut_case"
    )
    model = ProductModel.find(product)  # its XML sits in the `xml` subfolder
    plan_functions = removal_plan_functions()

    def plan(process, target, tool):
        operation = Operation("op_1", process, target, (Share("robot_1", tool),))
        return plan_functions[process](
            model,
            Cell_Assembly_Snapshot(model.xml, CellState()),
            operation,
            ROBOTS["robot_1"],
            tool,
        )

    with pytest.raises(
        PlanFailure, match="part_i_bolt"
    ):  # the nut locks the bolt: no removal path
        plan(Process.GRIPPING_AND_REMOVING, "part_i_bolt", "H5")
    with pytest.raises(PlanFailure, match="no_such_part"):
        plan(Process.GRIPPING_AND_REMOVING, "no_such_part", "H5")
    with pytest.raises(PlanFailure, match="no screw connection"):
        plan(Process.UNSCREWING, "part_i_bolt", "T6")


def test_a_failing_plan_function_makes_the_operation_infeasible_instead_of_raising():
    def failing(product_model, snapshot, operation, robot, tool):
        raise PlanFailure("no free direction for lid")

    record = run_with({Process.GRIPPING_AND_REMOVING: failing})

    first = record.entries[
        0
    ]  # the run controller then replans around the infeasible operation
    infeasible = first.simulation_report.infeasible
    assert (infeasible.operation_id, infeasible.reason_code) == (
        "op_2",
        ReasonCode.NO_PATH_SHARE,
    )
    assert "no free direction for lid" in infeasible.text
    assert infeasible.robot == "robot_1"


def test_processes_without_a_plan_function_keep_their_nominal_share():
    record = run_with({})

    unscrew = record.operation_plans[-1].operations[0]
    assert unscrew.shares[0].path == () and unscrew.shares[0].path_duration == 5.0
    assert record.status == "goal_reached"
