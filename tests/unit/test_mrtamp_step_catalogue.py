"""The step catalogue: the admissible steps from a product state for a robot, with their cost after the feedback."""

import pytest
from diss_assembly_dataclasses.feedback import (
    InfeasibleOperation,
    OperationOutcome,
    ReasonCode,
)
from diss_assembly_dataclasses.operation import StepRef
from diss_assembly_dataclasses.operation_plan import OperationPlan
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_reversal import AssemblyConstants, complete_by_reversal
from mrtamp.planning import Goal, PlanningRequest, RunSettings
from mrtamp.step_catalogue import StepCatalogue
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    ALL,
    SUPPLY_SLOTS,
    assembly_cell,
    base_only_snapshot,
    lid_screwed_on_base_graph,
    product_model,
    start_snapshot,
    transformer_cell,
    travel_times,
)

CONSTANTS = AssemblyConstants(
    screw_in_time=4.0,
    supply_travel_time=6.0,
    insert_approach_time=2.0,
    insert_feed_time=3.0,
)
TOOL_CHANGE = 80.0  # transformer cell: 10 + 60 + 10


def disassembly_request(**overrides):
    kwargs = dict(
        sequence_graph=lid_screwed_on_base_graph(),
        product_model=product_model(),
        cell=transformer_cell(),
        start_snapshot=start_snapshot(),
        settings=RunSettings(goal=Goal(absent=("lid",))),
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def assembly_request(**overrides):
    kwargs = dict(
        sequence_graph=complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS),
        product_model=product_model(),
        cell=assembly_cell(),
        start_snapshot=base_only_snapshot(),
        settings=RunSettings(goal=Goal(present=ALL)),
        supply_slots=SUPPLY_SLOTS,
        travel_times=travel_times(),
    )
    kwargs.update(overrides)
    return PlanningRequest(**kwargs)


def robot(request, robot_id):
    return next(r for r in request.cell.robots if r.id == robot_id)


def offered(catalogue, state, robot, tool=None):
    return {
        (s.step.process, s.step.target_part): s.cost
        for s in catalogue.steps(state, robot, tool)
    }


def test_the_catalogue_offers_the_robots_steps_with_nominal_time_plus_tool_change():
    request = disassembly_request()
    catalogue = StepCatalogue(request)

    assert offered(catalogue, ALL, robot(request, "robot_1"), "H5") == {
        (Process.UNSCREWING, "screw"): 5.0 + TOOL_CHANGE,
        (Process.MILLING, "lid"): 150.0 + TOOL_CHANGE,
    }
    assert offered(catalogue, ALL, robot(request, "robot_1"), "T6") == {
        (Process.UNSCREWING, "screw"): 5.0,
        (Process.MILLING, "lid"): 150.0 + TOOL_CHANGE,
    }


def test_the_catalogue_leaves_out_steps_the_robot_cannot_carry_out():
    request = disassembly_request()

    assert StepCatalogue(request).steps(ALL, robot(request, "robot_2")) == []


def test_in_an_assembly_run_the_catalogue_offers_only_the_inverse_steps_the_goal_allows():
    request = assembly_request()

    steps = StepCatalogue(request).steps(("base",), robot(request, "robot_1"), "H5")

    assert [(s.step.process, s.step.target_part, s.after) for s in steps] == [
        (Process.INSERTING, "lid", ("base", "lid"))
    ]


def test_a_step_without_a_supply_slot_for_its_part_is_left_out():
    request = assembly_request(supply_slots=())

    assert StepCatalogue(request).steps(("base",), robot(request, "robot_1")) == []


def lid_inserting_plan():
    plan = v1_plan(assembly_request())
    assert isinstance(plan, OperationPlan)
    return plan


def operation_of(plan, process):
    return next(
        op for op in plan.operations if op.process is process and op.target == "lid"
    )


def test_every_operation_of_an_inverse_inserting_step_names_that_step():
    plan = lid_inserting_plan()

    named = {op.process: op.step for op in plan.operations if op.target == "lid"}

    lid_step = StepRef(("base",), "lid", Process.INSERTING)
    assert named == {
        Process.GRIPPING: lid_step,
        Process.INSERTING: lid_step,
        Process.RELEASING: lid_step,
    }
    assert all(
        op.step is None for op in plan.operations if op.process is Process.INSPECTING
    )


@pytest.mark.parametrize(
    "failed", [Process.GRIPPING, Process.INSERTING, Process.RELEASING]
)
def test_a_failed_operation_of_an_inverse_inserting_step_penalises_and_then_excludes_the_step(
    failed,
):
    plan = lid_inserting_plan()
    operation = operation_of(plan, failed)
    failure = OperationOutcome(operation.id, False, ("base",))
    request = assembly_request()
    robot_1 = robot(request, "robot_1")

    def lid_cost(*feedback):
        catalogue = StepCatalogue(
            assembly_request(feedback=feedback, earlier_plans=(plan,))
        )
        return offered(catalogue, ("base",), robot_1, "H5").get(
            (Process.INSERTING, "lid")
        )

    nominal = lid_cost()
    assert lid_cost(failure) == 2 * nominal
    assert lid_cost(failure, failure) is None  # past the retry limit of 1


def test_an_infeasible_gripping_at_the_supply_excludes_its_inserting_step_at_once():
    plan = lid_inserting_plan()
    gripping = operation_of(plan, Process.GRIPPING)
    infeasible = InfeasibleOperation(
        gripping.id, "robot_1", ReasonCode.OUT_OF_REACH, "unreachable slot"
    )
    request = assembly_request(feedback=(infeasible,), earlier_plans=(plan,))

    assert StepCatalogue(request).steps(("base",), robot(request, "robot_1")) == []


def test_an_operation_without_a_step_carries_no_penalty():
    plan = lid_inserting_plan()
    inspection = next(op for op in plan.operations if op.process is Process.INSPECTING)
    failure = OperationOutcome(inspection.id, False, ("base",))
    request = assembly_request(feedback=(failure,), earlier_plans=(plan,))

    steps = StepCatalogue(request).steps(("base",), robot(request, "robot_1"), "H5")

    assert [s.cost for s in steps] == [
        s.cost
        for s in StepCatalogue(assembly_request()).steps(
            ("base",), robot(request, "robot_1"), "H5"
        )
    ]
