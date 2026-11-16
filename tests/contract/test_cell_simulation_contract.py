"""Contract suite for cell simulations: the rules every adapter of `mrtamp.cell_simulation` shares.

Add a simulation with one entry in SIMULATIONS: a factory of the simulation. Every request carries the product
model, as the planning request does. Whether a
cell-targeted run gets NC programs from it is the adapter's own `produces_nc_programs`; the model-level stand-in
produces none. The cases are black-box: what goes in and what the report says, never an adapter's internals.
"""

from dataclasses import dataclass, replace
from typing import Callable

import pytest
from diss_assembly_dataclasses.feedback import ReasonCode
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.operation_plan import (
    OperationPlan,
    PlanMetrics,
    SyncPoint,
)
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.travel_time_table import TravelTimeTable
from motion_planning.common import PLAN_FAILURE, PLAN_FAILURE_ROBOT, SUPPLY_SLOT
from mrtamp.assembly_cell import ASSEMBLY_PROCESSES
from mrtamp.cell_simulation import INSPECTION_TIME, SimulationRequest
from mrtamp.planning import Goal, GoalUnreachable, RunSettings
from mrtamp.run_controller import run
from mrtamp.stand_in import ModelLevelStandIn
from tests.mrtamp_scenarios import (
    START_XML,
    TWO_SCREWS_XML,
    base_only_snapshot,
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
    two_robot_cell,
    two_screws_snapshot_both_on_t6,
)
from tests.sim_loopback import loopback_simulation, requires_sim

INFEASIBLE_CODES = {
    ReasonCode.NO_PATH_SHARE,
    ReasonCode.NO_PATH_TRANSIT,
    ReasonCode.OUT_OF_REACH,
    ReasonCode.COLLISION,
    ReasonCode.SYNC_DEADLOCK,
    ReasonCode.CELL_STATE_MISMATCH,
    ReasonCode.UNSUPPORTED_PROCESS,
}


@dataclass(frozen=True)
class SimulationUnderTest:
    factory: Callable  # () -> a cell simulation

    def make(self):
        return self.factory()


SIMULATIONS = [
    pytest.param(SimulationUnderTest(ModelLevelStandIn), id="model-level-stand-in"),
    pytest.param(
        SimulationUnderTest(lambda: loopback_simulation()[0]),
        id="crc-disassembly-sim",
        marks=requires_sim,
    ),
]


@pytest.fixture(params=SIMULATIONS)
def simulation(request):
    return request.param


def op(op_id, process, target, state, robot="robot_1", tool=None, duration=5.0):
    return Operation(
        op_id,
        process,
        target,
        (Share(robot, tool, path_duration=duration),),
        expected_product_state=state,
    )


def inspection(op_id, state):
    shares = tuple(
        Share(r.id, None, path_duration=10.0) for r in transformer_cell().robots
    )
    return Operation(
        op_id, Process.INSPECTING, None, shares, expected_product_state=state
    )


def unscrew(op_id="op_1"):
    return op(op_id, Process.UNSCREWING, "screw", ("base", "lid"), tool="T6")


def remove_lid(op_id="op_2"):
    return op(op_id, Process.GRIPPING_AND_REMOVING, "lid", ("base",), tool="H5")


def remove_ghost(op_id="op_2"):
    return op(op_id, Process.GRIPPING_AND_REMOVING, "ghost", ("base",), tool="H5")


def request_for(*operations, **overrides):
    snapshot = start_snapshot()
    plan = OperationPlan(tuple(operations), {}, (), PlanMetrics(0.0), snapshot)
    return SimulationRequest(
        **{
            "operation_plan": plan,
            "start_snapshot": snapshot,
            "cell": transformer_cell(),
            "product_model": product_model(),
            "travel_times": travel_times(),
            **overrides,
        }
    )


def test_reason_codes_come_only_from_the_closed_list(simulation):
    sim = simulation.make()
    milling_without_a_mill = op(
        "op_1", Process.MILLING, "lid", ("base", "screw"), robot="robot_2"
    )

    for operation in (milling_without_a_mill, remove_ghost("op_1")):
        report = sim.simulate(request_for(operation))
        assert report.ok and report.infeasible.reason_code in INFEASIBLE_CODES
        assert report.infeasible.text


def test_simulation_stops_at_the_first_infeasible_operation(simulation):
    report = simulation.make().simulate(
        request_for(
            unscrew(),
            remove_ghost(),
            remove_lid("op_3"),
            inspection("inspection_1", ("base",)),
        )
    )

    assert report.infeasible.operation_id == "op_2"
    assert report.feasible_operations == ("op_1",)
    assert [o.operation_id for o in report.outcomes] == ["op_1"]
    assert "inspection_1" not in report.observed_snapshots


def test_observed_snapshot_at_each_inspection_reached(simulation):
    report = simulation.make().simulate(
        request_for(
            unscrew(),
            inspection("inspection_1", ("base", "lid")),
            remove_lid(),
            inspection("inspection_2", ("base",)),
        )
    )

    assert set(report.observed_snapshots) == {"inspection_1", "inspection_2"}
    assert report.observed_snapshots["inspection_1"].product_state_key == (
        "base",
        "lid",
    )
    assert report.observed_snapshots["inspection_2"].product_state_key == ("base",)


def test_inspections_after_an_infeasible_operation_are_not_reached(simulation):
    report = simulation.make().simulate(
        request_for(
            unscrew(),
            inspection("inspection_1", ("base", "lid")),
            remove_ghost(),
            inspection("inspection_2", ("base",)),
        )
    )

    assert set(report.observed_snapshots) == {"inspection_1"}


def test_nc_programs_come_only_from_a_simulation_that_produces_them(simulation):
    sim = simulation.make()
    report = sim.simulate(
        request_for(
            unscrew(),
            inspection("inspection_1", ("base", "lid")),
            remove_lid(),
            inspection("inspection_2", ("base",)),
            target="cell",
        )
    )

    if not sim.produces_nc_programs:
        assert report.nc_programs is None
        return
    assert report.nc_programs and all(
        {"file_name", "text"} <= set(nc) for nc in report.nc_programs.values()
    )


def test_a_concurrent_group_may_be_any_iterable_of_operations(simulation):
    group = (
        unscrew_screw(1, "robot_1"),
        replace(unscrew_screw(1, "robot_2"), id="op_2"),
    )  # one part, two operations

    answer = simulation.make().check_feasibility(
        (op for op in group),
        two_screws_snapshot_both_on_t6(),
        two_robot_cell(),
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=travel_times(),
    )

    assert not answer.feasible and answer.infeasible.operation_id == "op_2"


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: {"operation_plan": {}}, id="dict"),
        pytest.param(
            lambda: replace(request_for(unscrew()), operation_plan=None), id="no-plan"
        ),
        pytest.param(
            lambda: replace(request_for(unscrew()), execution_mode="turbo"),
            id="unknown-mode",
        ),
        pytest.param(
            lambda: replace(request_for(unscrew()), target="moon"), id="unknown-target"
        ),
        pytest.param(
            lambda: request_for(unscrew(), product_model=START_XML),
            id="product-model-as-xml-text",
        ),
        pytest.param(
            lambda: replace(
                request_for(unscrew(), remove_lid()),
                operation_plan=replace(
                    request_for(unscrew(), remove_lid()).operation_plan,
                    sync_points=(
                        SyncPoint("robot_1", "op_1", "op_2", "planner", "order"),
                    ),
                ),
            ),
            id="sync-point-waits-for-a-later-operation",
        ),
        pytest.param(
            lambda: replace(
                request_for(unscrew()),
                operation_plan=replace(
                    request_for(unscrew()).operation_plan,
                    sync_points=(
                        SyncPoint("robot_1", "op_1", "op_ghost", "planner", "order"),
                    ),
                ),
            ),
            id="sync-point-names-an-unknown-operation",
        ),
        pytest.param(
            lambda: request_for(
                op(
                    "op_1",
                    Process.UNSCREWING,
                    "screw",
                    ("base", "lid"),
                    "robot_9",
                    "T6",
                )
            ),
            id="unknown-robot",
        ),
        pytest.param(
            lambda: request_for(
                op(
                    "op_1",
                    Process.UNSCREWING,
                    "screw",
                    ("base", "lid"),
                    "robot_2",
                    "T6",
                )
            ),
            id="tool-the-robot-cannot-mount",
        ),
        pytest.param(
            lambda: request_for(
                replace(unscrew(), shares=()),
            ),
            id="missing-share",
        ),
    ],
)
def test_malformed_request_gives_invalid_request(simulation, malformed):
    report = simulation.make().simulate(malformed())

    assert not report.ok
    assert report.rejection.reason_code is ReasonCode.INVALID_REQUEST
    assert report.nc_programs is None


# --- feasibility queries: one per concurrent group, nothing executed, no NC ----------------------------------


def unscrew_screw(n, robot):
    return op(
        f"op_{n}",
        Process.UNSCREWING,
        f"screw_{n}",
        ("base", "lid"),
        robot=robot,
        tool="T6",
    )


def query(simulation, *group):
    return simulation.make().check_feasibility(
        group,
        two_screws_snapshot_both_on_t6(),
        two_robot_cell(),
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=travel_times(),
    )


def test_feasibility_query_answers_feasible_with_the_estimated_duration_of_every_operation(
    simulation,
):
    answer = query(simulation, unscrew_screw(1, "robot_1"), unscrew_screw(2, "robot_2"))

    assert answer.feasible and answer.infeasible is None and answer.rejection is None
    assert set(answer.durations) == {"op_1", "op_2"} and all(
        d > 0 for d in answer.durations.values()
    )


def test_feasibility_query_names_the_infeasible_operation_with_a_reason_from_the_closed_list(
    simulation,
):
    answer = query(
        simulation,
        unscrew_screw(1, "robot_1"),
        op(
            "op_9",
            Process.GRIPPING_AND_REMOVING,
            "ghost",
            ("base",),
            robot="robot_2",
            tool="H5",
        ),
    )

    assert not answer.feasible
    assert (
        answer.infeasible.operation_id == "op_9"
        and answer.infeasible.reason_code in INFEASIBLE_CODES
    )
    assert answer.infeasible.text


def test_two_operations_on_one_part_at_the_same_time_are_infeasible(simulation):
    answer = query(
        simulation,
        unscrew_screw(1, "robot_1"),
        replace(unscrew_screw(1, "robot_2"), id="op_2"),
    )

    assert not answer.feasible and answer.infeasible.operation_id == "op_2"
    assert answer.infeasible.reason_code in INFEASIBLE_CODES


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(lambda: (), id="empty"),
        pytest.param(
            lambda: (unscrew_screw(1, "robot_1"), unscrew_screw(2, "robot_1")),
            id="two-on-one-robot",
        ),
        pytest.param(lambda: ("not an operation",), id="not-operations"),
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: (unscrew_screw(1, "robot_9"),), id="unknown-robot"),
        pytest.param(
            lambda: (replace(unscrew_screw(1, "robot_1"), shares=()),),
            id="missing-share",
        ),
        pytest.param(
            lambda: (
                op(
                    "op_1", Process.UNSCREWING, "screw_1", ("base", "lid"), tool="drill"
                ),
            ),
            id="tool-the-robot-cannot-mount",
        ),
    ],
)
def test_a_malformed_concurrent_group_gives_invalid_request(simulation, group):
    answer = simulation.make().check_feasibility(
        group(),
        two_screws_snapshot_both_on_t6(),
        two_robot_cell(),
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=travel_times(),
    )

    assert (
        not answer.feasible
        and answer.rejection.reason_code is ReasonCode.INVALID_REQUEST
    )


def test_a_feasibility_query_without_a_product_model_gives_invalid_request(simulation):
    answer = simulation.make().check_feasibility(
        (unscrew_screw(1, "robot_1"),),
        two_screws_snapshot_both_on_t6(),
        two_robot_cell(),
        product_model=TWO_SCREWS_XML,  # the XML text, not the product model
        travel_times=travel_times(),
    )

    assert (
        not answer.feasible
        and answer.rejection.reason_code is ReasonCode.INVALID_REQUEST
    )


def test_a_request_whose_times_are_not_a_travel_time_table_gives_invalid_request(
    simulation,
):
    times = travel_times().times  # the bare times, not the table
    answer = simulation.make().check_feasibility(
        (unscrew_screw(1, "robot_1"),),
        two_screws_snapshot_both_on_t6(),
        two_robot_cell(),
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=times,
    )
    report = simulation.make().simulate(
        request_for(unscrew_screw(1, "robot_1"), travel_times=times)
    )

    assert answer.rejection.reason_code is ReasonCode.INVALID_REQUEST
    assert report.rejection.reason_code is ReasonCode.INVALID_REQUEST


# --- infeasibility: the first infeasible operation and the robot whose share failed ------------------------------


def lid_removed_by_both(op_id="op_2", failing_robot="robot_2"):
    """A two-share removal whose plan function failed for `failing_robot`'s share."""
    return Operation(
        op_id,
        Process.GRIPPING_AND_REMOVING,
        "lid",
        (
            Share("robot_1", "H5", path_duration=4.0),
            Share("robot_2", "H5", path_duration=4.0),
        ),
        expected_product_state=("base",),
        parameters={
            PLAN_FAILURE: f"no grip point for {failing_robot}",
            PLAN_FAILURE_ROBOT: failing_robot,
        },
    )


@pytest.mark.parametrize("failing_robot", ["robot_1", "robot_2"])
def test_the_report_names_the_first_infeasible_operation_and_the_robot_whose_share_failed(
    simulation, failing_robot
):
    report = simulation.make().simulate(
        request_for(
            unscrew(),
            lid_removed_by_both(failing_robot=failing_robot),
            inspection("inspection_1", ("base",)),
            cell=two_robot_cell(),
        )
    )

    assert report.feasible_operations == ("op_1",)
    assert (
        report.infeasible.operation_id,
        report.infeasible.robot,
        report.infeasible.reason_code,
    ) == ("op_2", failing_robot, ReasonCode.NO_PATH_SHARE)


def test_a_feasibility_query_names_the_robot_whose_share_failed(simulation):
    answer = simulation.make().check_feasibility(
        (lid_removed_by_both(),),
        start_snapshot(),
        two_robot_cell(),
        product_model=product_model(),
        travel_times=travel_times(),
    )

    assert not answer.feasible
    assert (answer.infeasible.operation_id, answer.infeasible.robot) == (
        "op_2",
        "robot_2",
    )


# --- cell state: holds and mounted tools as observed at the next inspection -------------------------------------


def assembling_cell():
    """Both robots grip at the supply, insert and release; robot 2 also inspects."""
    cell = two_robot_cell()
    return replace(
        cell,
        robots=tuple(
            replace(
                r,
                capabilities=tuple(dict.fromkeys(r.capabilities + ASSEMBLY_PROCESSES)),
            )
            for r in cell.robots
        ),
    )


def grip_at_supply(op_id, robot):
    return replace(
        op(op_id, Process.GRIPPING, "lid", ("base",), robot=robot, tool="H5"),
        parameters={SUPPLY_SLOT: "lid_slot"},
    )


def test_holds_and_mounted_tools_at_each_inspection_follow_gripping_releasing_handover_and_tool_change(
    simulation,
):
    # handover: robot 1 releases the lid, robot 2 grips it at the supply (park 20261028-003635-554d)
    report = simulation.make().simulate(
        request_for(
            grip_at_supply("op_1", "robot_1"),
            inspection("inspection_1", ("base",)),
            op("op_2", Process.RELEASING, "lid", ("base",)),
            grip_at_supply("op_3", "robot_2"),
            inspection("inspection_2", ("base",)),
            op("op_4", Process.INSERTING, "lid", ("base", "lid"), "robot_2", "H5"),
            op("op_5", Process.RELEASING, "lid", ("base", "lid"), "robot_2"),
            op("op_6", Process.MILLING, "lid", ("base",), tool="mill"),
            inspection("inspection_3", ("base",)),
            start_snapshot=base_only_snapshot(),
            cell=assembling_cell(),
        )
    )

    assert report.infeasible is None
    observed = {
        i: (s.cell_state.holds, s.cell_state.mounted_tools)
        for i, s in report.observed_snapshots.items()
    }
    assert observed == {
        "inspection_1": ({"robot_1": "lid"}, {"robot_1": "H5", "robot_2": None}),
        "inspection_2": ({"robot_2": "lid"}, {"robot_1": "H5", "robot_2": "H5"}),
        "inspection_3": ({}, {"robot_1": "mill", "robot_2": "H5"}),
    }


# --- timing structure, not values --------------------------------------------------------------------------------


def tool_changes(report):
    return [
        (t["robot"], t["tool"]) for t in report.timeline if t["kind"] == "tool_change"
    ]


def test_a_tool_change_is_reported_only_when_the_tool_differs(simulation):
    sim = simulation.make()

    changed = sim.simulate(
        request_for(unscrew(), remove_lid())
    )  # H5 mounted at the start
    kept = sim.simulate(request_for(remove_lid("op_1")))

    assert tool_changes(changed) == [("robot_1", "T6"), ("robot_1", "H5")]
    assert tool_changes(kept) == []


def test_an_inspection_lasts_the_inspection_time_of_the_travel_time_table(simulation):
    table = TravelTimeTable({**travel_times().times, INSPECTION_TIME: 7.0})

    report = simulation.make().simulate(
        request_for(
            inspection("inspection_1", ("base", "lid", "screw")), travel_times=table
        )
    )  # each share declares 10 s

    spans = [
        t["end"] - t["start"]
        for t in report.timeline
        if t.get("operation_id") == "inspection_1"
    ]
    assert spans == [7.0, 7.0]


# --- sync points ----------------------------------------------------------------------------------------------


def test_a_task_level_sync_point_makes_the_robot_wait(simulation):
    snapshot = two_screws_snapshot_both_on_t6()
    first = replace(
        unscrew_screw(1, "robot_1"), expected_product_state=("base", "lid", "screw_2")
    )
    second = op(
        "op_2", Process.UNSCREWING, "screw_2", ("base", "lid"), "robot_2", "T6", 3.0
    )
    plan = OperationPlan(
        (first, second),
        {},
        (SyncPoint("robot_2", "op_2", "op_1", "planner", "order"),),
        PlanMetrics(0.0),
        snapshot,
    )

    report = simulation.make().simulate(
        SimulationRequest(
            plan,
            snapshot,
            two_robot_cell(),
            product_model=product_model(TWO_SCREWS_XML),
            travel_times=travel_times(),
        )
    )

    span = {
        t["operation_id"]: (t["start"], t["end"])
        for t in report.timeline
        if t["kind"] == "operation"
    }
    assert report.infeasible is None and span["op_2"][0] >= span["op_1"][1]


# --- cell target (ADR-0008) --------------------------------------------------------------------------------------


def test_a_cell_targeted_run_is_refused_before_planning_with_a_simulation_that_produces_no_nc(
    simulation,
):
    sim, asked = simulation.make(), []

    def planner(request):
        asked.append(request)
        return GoalUnreachable("the test stops here")

    def cell_run():
        return run(
            lid_screwed_on_base_graph(),
            product_model(),
            transformer_cell(),
            start_snapshot(),
            RunSettings(goal=Goal(absent=("lid",)), target="cell"),
            planner,
            sim,
            travel_times=travel_times(),
        )

    if sim.produces_nc_programs:
        assert cell_run().status == "goal_unreachable" and asked
        return
    with pytest.raises(ValueError, match="NC programs"):
        cell_run()
    assert asked == []
