"""The operation-plan builder: a planner commits the steps it chose, asks for robot clocks, step spans and mounted
tools, and finishes into an operation plan with inspections and plan metrics."""

from diss_assembly_dataclasses.operation_plan import OperationPlan, SyncPoint
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_steps import Step
from mrtamp.nominal_schedule import nominal_schedule, operation_spans
from mrtamp.operation_plan_builder import OperationPlanBuilder
from mrtamp.planning import Goal, PlanningRequest, RunSettings
from tests.mrtamp_scenarios import (
    TWO_SCREWS_XML,
    product_model,
    travel_times,
    two_robot_cell,
    two_screws_graph,
    two_screws_snapshot,
)

FULL = ("base", "lid", "screw_1", "screw_2")
ONLY_2, ONLY_1, BARE = (
    ("base", "lid", "screw_2"),
    ("base", "lid", "screw_1"),
    ("base", "lid"),
)
TOOL_CHANGE = 80.0  # transformer cell: 10 + 60 + 10


def request(**overrides):
    kwargs = dict(
        sequence_graph=two_screws_graph(screw_2_time=7.0),
        product_model=product_model(TWO_SCREWS_XML),
        cell=two_robot_cell(),
        start_snapshot=two_screws_snapshot(),  # robot_1 has T6, robot_2 no tool
        settings=RunSettings(goal=Goal(absent=("lid",))),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def graph_step(req, before, after) -> Step:
    (attrs,) = req.sequence_graph.get_edge_data(before, after).values()
    return Step.from_edge(attrs)


def robot(req, robot_id):
    return next(r for r in req.cell.robots if r.id == robot_id)


def test_committed_steps_are_numbered_in_commit_order_and_name_their_step():
    req = request()
    builder = OperationPlanBuilder(req)

    one = builder.commit(
        robot(req, "robot_1"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2)
    )
    two = builder.commit(
        robot(req, "robot_2"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE)
    )

    assert [op.id for op in one.planned.operations] == ["op_1"]
    assert two.planned.operations[0].id == "op_2"
    assert two.planned.operations[0].step.target == "screw_2"
    assert (one.robot, two.robot) == ("robot_1", "robot_2")


def test_a_committed_step_reports_its_nominal_span_and_advances_its_robots_clock():
    req = request()
    builder = OperationPlanBuilder(req)

    one = builder.commit(
        robot(req, "robot_1"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2)
    )
    two = builder.commit(
        robot(req, "robot_2"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE)
    )

    assert (one.start, one.end) == (0.0, 5.0)  # T6 already mounted
    assert (two.start, two.end) == (
        TOOL_CHANGE,
        TOOL_CHANGE + 7.0,
    )  # robot_2 mounts T6 first
    assert builder.clock == {"robot_1": 5.0, "robot_2": TOOL_CHANGE + 7.0}
    assert builder.mounted_tools == {"robot_1": "T6", "robot_2": "T6"}
    assert builder.committed == (one, two)


def test_a_step_that_waits_for_another_starts_when_that_one_ends_through_an_order_sync_point():
    req = request()
    builder = OperationPlanBuilder(req)
    one = builder.commit(
        robot(req, "robot_2"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2)
    )

    two = builder.commit(
        robot(req, "robot_1"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE), wait_for=one
    )

    assert two.start == one.end == TOOL_CHANGE + 5.0
    plan = builder.finish()
    assert plan.sync_points == (
        SyncPoint(
            "robot_1",
            two.planned.operations[0].id,
            one.planned.operations[-1].id,
            "planner",
            "order",
        ),
    )


def test_the_next_step_is_planned_from_the_cell_state_the_committed_steps_leave():
    req = request()
    builder = OperationPlanBuilder(req)
    builder.commit(robot(req, "robot_1"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2))
    builder.commit(robot(req, "robot_1"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE))

    lid = builder.commit(
        robot(req, "robot_1"), BARE, ("base",), graph_step(req, BARE, ("base",))
    )

    assert lid.start == 5.0 + 7.0 + TOOL_CHANGE  # T6 to H5
    assert builder.mounted_tools["robot_1"] == "H5"


def test_finish_places_inspections_and_reports_the_plan_metrics_of_its_operations():
    req = request()
    builder = OperationPlanBuilder(req)
    builder.commit(robot(req, "robot_1"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2))
    builder.commit(robot(req, "robot_2"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE))
    builder.commit(
        robot(req, "robot_1"), BARE, ("base",), graph_step(req, BARE, ("base",))
    )

    plan = builder.finish()

    assert isinstance(plan, OperationPlan)
    assert plan.operations[-1].process is Process.INSPECTING
    _, metrics, _ = nominal_schedule(
        plan.operations,
        req.cell,
        req.start_snapshot.cell_state.mounted_tools,
        plan.sync_points,
        travel_times=req.travel_times,
    )
    assert plan.metrics == metrics
    assert plan.start_snapshot == req.start_snapshot
    for r in req.cell.robots:
        assert plan.sequences[r.id] == tuple(
            op.id for op in plan.operations if r.id in op.robots
        )


def test_operation_spans_are_the_nominal_start_and_end_of_each_operation_of_a_plan():
    req = request()
    builder = OperationPlanBuilder(req)
    one = builder.commit(
        robot(req, "robot_2"), FULL, ONLY_2, graph_step(req, FULL, ONLY_2)
    )
    two = builder.commit(
        robot(req, "robot_1"), ONLY_2, BARE, graph_step(req, ONLY_2, BARE), wait_for=one
    )

    spans = operation_spans(builder.finish(), req.cell, travel_times=req.travel_times)

    assert spans[one.planned.operations[0].id] == (one.start, one.end)
    assert spans[two.planned.operations[0].id] == (two.start, two.end)
