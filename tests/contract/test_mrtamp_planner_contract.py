"""Contract suite for MRTAMP planners at the `plan(planning request)` seam.

Add a planner with one entry in PLANNERS; every check runs against it. Checks that need a ticket that
is not done yet are expected to fail (strict), naming the blocking ticket.
"""

import copy
import functools
import logging
from dataclasses import replace

import networkx as nx
import pytest
from diss_assembly_dataclasses.feedback import InfeasibleOperation, ReasonCode
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process

from mrtamp.b1_planner import plan as b1_plan
from mrtamp.nominal_schedule import nominal_schedule
from mrtamp.planning import Goal, GoalUnreachable, PlanningRequest, RunSettings
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
)

PLANNERS = [
    pytest.param(v1_plan, id="v1"),
    pytest.param(b1_plan, id="b1"),
    pytest.param(
        functools.partial(b1_plan, priority="critical_path"), id="b1-critical-path"
    ),
]


def xfail_until_ticket(ticket):
    return pytest.mark.xfail(strict=True, reason=f"waits for #{ticket}")


@pytest.fixture(params=PLANNERS)
def planner(request):
    return request.param


def make_request(goal=Goal(absent=("lid",)), **overrides):
    kwargs = dict(
        sequence_graph=lid_screwed_on_base_graph(),
        product_model=product_model(),
        cell=transformer_cell(),
        start_snapshot=start_snapshot(),
        settings=RunSettings(goal=goal),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def product_operations(plan):
    return [op for op in plan.operations if op.process is not Process.INSPECTING]


def test_plan_follows_steps_of_the_sequence_graph_from_the_start_to_the_goal(planner):
    request = make_request()
    plan = planner(request)

    assert isinstance(plan, OperationPlan)
    state = request.start_snapshot.product_state_key
    for op in product_operations(plan):
        steps = (
            request.sequence_graph.get_edge_data(state, op.expected_product_state) or {}
        )
        assert any(
            s["process"] == op.process.value and s["target_part"] == op.target
            for s in steps.values()
        ), op
        state = op.expected_product_state
    assert request.settings.goal.reached(state)


def test_plan_logs_no_error_on_a_graph_without_screw_groups(planner, caplog):
    with caplog.at_level(logging.ERROR):
        planner(make_request())

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_plan_uses_only_processes_and_tools_the_cells_robots_can_carry_out(planner):
    request = make_request()
    robots = {r.id: r for r in request.cell.robots}

    for op in product_operations(planner(request)):
        assert op.shares
        for share in op.shares:
            assert robots[share.robot].can_carry_out(op.process, share.tool), (
                op.id,
                share,
            )


def test_goal_unreachable_when_no_robot_can_carry_out_a_needed_process(planner):
    cell = transformer_cell()
    cannot = (Process.GRIPPING_AND_REMOVING, Process.MILLING)
    robots = tuple(
        replace(r, capabilities=tuple(p for p in r.capabilities if p not in cannot))
        for r in cell.robots
    )
    result = planner(make_request(cell=replace(cell, robots=robots)))

    assert isinstance(result, GoalUnreachable) and result.reason


def test_plan_respects_a_ban_from_the_feedback(planner):
    first = planner(make_request())
    banned = first.operations[0]
    feedback = (
        InfeasibleOperation(
            banned.id, banned.robots[0], ReasonCode.NO_PATH_SHARE, "no path"
        ),
    )

    again = planner(make_request(feedback=feedback, earlier_plans=(first,)))

    assert isinstance(again, OperationPlan)
    assert (banned.process, banned.target, banned.robots) not in [
        (o.process, o.target, o.robots) for o in again.operations
    ]


def test_plan_metrics_match_the_operations(planner):
    request = make_request()
    plan = planner(request)

    _, metrics, _ = nominal_schedule(
        plan.operations,
        request.cell,
        request.start_snapshot.cell_state.mounted_tools,
        plan.sync_points,
        travel_times=request.travel_times,
    )
    assert plan.metrics == metrics
    assert plan.metrics.makespan >= max(plan.metrics.busy_time.values())
    assert plan.metrics.inspections == sum(
        op.process is Process.INSPECTING for op in plan.operations
    )


def test_plan_ends_with_an_inspection_over_all_robots(planner):
    request = make_request()
    last = planner(request).operations[-1]

    assert last.process is Process.INSPECTING
    assert set(last.robots) == {r.id for r in request.cell.robots}


def test_every_risky_operation_is_covered_by_a_later_inspection(planner):
    ops = planner(make_request()).operations

    for i, op in enumerate(ops):
        if op.process is not Process.INSPECTING:
            assert any(later.process is Process.INSPECTING for later in ops[i + 1 :]), (
                op.id
            )


def test_inspection_comes_before_an_operation_that_depends_on_an_uninspected_risky_one(
    planner,
):
    ops = planner(make_request()).operations

    for before, after in zip(ops, ops[1:]):
        assert (
            before.process is Process.INSPECTING or after.process is Process.INSPECTING
        ), (before.id, after.id)


def test_planner_is_stateless_across_requests(planner):
    request = make_request()
    graph_before = copy.deepcopy(request.sequence_graph)

    first = planner(request)
    planner(
        make_request(goal=Goal(absent=("lid", "base")))
    )  # an unrelated request in between
    second = planner(request)

    assert first == second
    assert nx.utils.graphs_equal(request.sequence_graph, graph_before)


def test_goal_unreachable_instead_of_dropping_a_target_part(planner):
    result = planner(make_request(goal=Goal(absent=("lid", "base"))))

    assert isinstance(result, GoalUnreachable)
    assert "base" in result.reason
