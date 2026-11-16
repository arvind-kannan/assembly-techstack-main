"""The removal plan core decides once: failure and its reason, the unit removal direction, and whether the
result assembly really lost the part (AT 234)."""

import copy

import numpy as np

from motion_planning.removal_planning import (
    RemovalPlan,
    part_removed,
    plan_removal,
    removal_plan_of,
)
from tests.conftest import DummyGrippingAndRemoving, set_dof, successor_without_part


def planned(**data):
    return DummyGrippingAndRemoving(
        {"path": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], "path_duration": 1.0, **data}
    )


def test_a_successful_plan_has_a_unit_direction_and_the_assembly_without_the_part(
    simple_manip_assembly,
):
    result = successor_without_part(simple_manip_assembly, "part_1")

    plan = removal_plan_of(
        planned(direction=[0.0, 0.0, 2.0]), result, "part_1", simple_manip_assembly
    )

    assert isinstance(plan, RemovalPlan)
    assert plan.ok and plan.failure is None
    assert np.allclose(plan.direction, [0.0, 0.0, 1.0])
    assert plan.result is result


def test_a_planner_failure_is_a_failure_with_the_planners_stage_and_reason(
    simple_manip_assembly,
):
    result = successor_without_part(simple_manip_assembly, "part_1")
    op = planned(
        cost=-1,
        direction=[0.0, 0.0, 0.0],
        failure_stage="no_feasible_dof_candidate",
        failure_reason="part_1 has DOF candidates, but none passed",
        gripping_radius=3.0,
    )

    plan = removal_plan_of(op, result, "part_1", simple_manip_assembly)

    assert not plan.ok
    assert plan.failure.stage == "planning"
    assert plan.failure.reason == "no_feasible_dof_candidate"
    assert plan.failure.message == "part_1 has DOF candidates, but none passed"
    assert plan.failure.details["planner_failure_reason"] == plan.failure.message
    assert plan.failure.details["gripping_radius"] == 3.0
    assert plan.direction is None and plan.result is None


def test_a_negative_cost_alone_is_a_planner_failure():
    plan = removal_plan_of(planned(cost=-1, direction=[0.0, 0.0, 1.0]), None, "part_1")

    assert (plan.failure.stage, plan.failure.reason) == ("planning", "planning_failed")
    assert plan.failure.message == "the removal planner found no path"


def test_a_zero_direction_falls_back_to_the_parts_degrees_of_freedom(
    simple_manip_assembly,
):
    set_dof(simple_manip_assembly, [np.array([0.0, 0.0, -3.0])])

    plan = removal_plan_of(
        planned(direction=[0.0, 0.0, 0.0]), None, "part_1", simple_manip_assembly
    )

    assert plan.ok
    assert np.allclose(plan.direction, [0.0, 0.0, 1.0])  # the sign prefers +z


def test_a_zero_direction_without_a_fallback_is_a_direction_failure():
    plan = removal_plan_of(planned(direction=[0.0, 0.0, 0.0]), None, "part_1")

    assert not plan.ok
    assert (plan.failure.stage, plan.failure.reason) == ("direction", "bad_direction")
    assert plan.failure.details["why"] == "no_valid_direction"
    assert plan.direction is None


def test_a_missing_direction_without_usable_degrees_of_freedom_is_a_direction_failure(
    simple_manip_assembly,
):
    set_dof(simple_manip_assembly, [])

    plan = removal_plan_of(planned(), None, "part_1", simple_manip_assembly)

    assert (plan.failure.stage, plan.failure.reason) == (
        "direction",
        "missing_direction",
    )


def test_a_result_that_still_holds_the_part_is_no_result(simple_manip_assembly):
    still_there = copy.deepcopy(simple_manip_assembly)

    plan = removal_plan_of(
        planned(direction=[1.0, 0.0, 0.0]), still_there, "part_1", simple_manip_assembly
    )

    assert plan.ok
    assert plan.result is None


def test_part_removed_is_true_only_when_the_assembly_lost_the_part(
    simple_manip_assembly,
):
    assert not part_removed(simple_manip_assembly, "part_1")
    assert part_removed(successor_without_part(simple_manip_assembly), "part_1")
    assert not part_removed(None, "part_1")


def test_removing_a_part_that_is_not_in_the_assembly_is_a_planning_failure(
    simple_manip_assembly,
):
    plan = plan_removal(simple_manip_assembly, "no_such_part")

    assert plan.failure.stage == "planning"
    assert "no_such_part" in plan.failure.message


def test_an_operation_that_carries_its_removal_plan_is_judged_once():
    plan = removal_plan_of(planned(direction=[0.0, 1.0, 0.0]), None, "part_1")

    assert removal_plan_of(planned(removal_plan=plan), None, "part_1") is plan
