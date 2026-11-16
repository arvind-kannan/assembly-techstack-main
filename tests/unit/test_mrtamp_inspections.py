"""Inspection placement (#103), checked at the v1 planner: where the inspections of a plan sit."""

from diss_assembly_dataclasses.process import Process

from mrtamp.planning import Goal, PlanningRequest, RunSettings
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    TWO_SCREWS_XML,
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
    two_screws_graph,
    two_screws_snapshot,
)


def two_screws_request(**settings):
    return PlanningRequest(
        two_screws_graph(),
        product_model(TWO_SCREWS_XML),
        transformer_cell(),
        two_screws_snapshot(),
        RunSettings(goal=Goal(absent=("lid",)), **settings),
        travel_times=travel_times(),
    )


def layout(plan):
    """The plan as a string of targets, inspections as '|'."""
    return " ".join(
        "|" if op.process is Process.INSPECTING else op.target for op in plan.operations
    )


def test_inspection_comes_before_an_operation_that_depends_on_an_uninspected_risky_one():
    request = PlanningRequest(
        lid_screwed_on_base_graph(),
        product_model(),
        transformer_cell(),
        start_snapshot(),
        RunSettings(goal=Goal(absent=("lid",))),
        travel_times=travel_times(),
    )

    assert layout(v1_plan(request)) == "screw | lid |"


def test_independent_risky_operations_share_one_inspection():
    plan = v1_plan(two_screws_request())

    assert layout(plan) == "screw_1 screw_2 | lid |"
    assert plan.metrics.inspections == 2


def test_changing_the_risky_processes_changes_where_inspections_are_placed():
    only_milling = v1_plan(two_screws_request(risky_processes=(Process.MILLING,)))
    only_removing = v1_plan(
        two_screws_request(risky_processes=(Process.GRIPPING_AND_REMOVING,))
    )
    all_changing = v1_plan(two_screws_request())

    assert (
        layout(only_milling) == "screw_1 screw_2 lid |"
    )  # nothing risky: just the end
    assert (
        layout(only_removing) == "screw_1 screw_2 lid |"
    )  # the removal is risky but nothing depends on it
    assert layout(all_changing) == "screw_1 screw_2 | lid |"


def test_inspection_count_in_the_plan_metrics_matches_the_plan():
    for settings in (
        {},
        {"risky_processes": (Process.UNSCREWING,)},
        {"risky_processes": ()},
    ):
        plan = v1_plan(two_screws_request(**settings))

        assert plan.metrics.inspections == sum(
            op.process is Process.INSPECTING for op in plan.operations
        )


def test_inspection_expects_the_product_state_it_follows_and_involves_every_robot():
    plan = v1_plan(two_screws_request())
    middle = next(op for op in plan.operations if op.process is Process.INSPECTING)

    assert middle.expected_product_state == ("base", "lid")
    assert set(middle.robots) == {"robot_1", "robot_2"}
    assert [op.id for op in plan.operations if op.process is Process.INSPECTING] == [
        "inspection_1",
        "inspection_2",
    ]


def test_a_cell_run_inspects_at_the_start_to_measure_the_reference_part():
    simulation = v1_plan(two_screws_request())
    cell = v1_plan(two_screws_request(target="cell"))

    assert layout(simulation).startswith("screw_1")
    assert layout(cell) == "| screw_1 screw_2 | lid |"
    assert (
        cell.operations[0].expected_product_state
        == two_screws_snapshot().product_state_key
    )
