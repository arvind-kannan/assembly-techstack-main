"""Assembly plan functions (#107) on the angle grinder case: inserting, screwing in, gripping at the supply."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot, SupplySlot

from disassembly_pomdp.product_model import ProductModel
from motion_planning.assembly_plan_functions import (
    REFERENCE_PART,
    assembly_plan_functions,
)
from motion_planning.common import SUPPLY_SLOT, assembly_xml_with
from mrtamp.plan_functions import fill_shares

pytestmark = pytest.mark.geom

MODEL = ProductModel.find(
    Path(__file__).resolve().parents[2] / "cases" / "angle_grinder" / "model"
)
MODEL_XML = MODEL.xml
PARTS = [p.attrib["name"] for p in ET.fromstring(MODEL_XML).findall("part")]
SCREWS = sorted(n for n in PARTS if n.startswith("Schraube"))
GEAR, BASE = "BG_Tellerrad_V1p", "Getriebegehaeuse_400_V1_02p"
START = tuple(
    sorted(set(PARTS) - set(SCREWS) - {GEAR})
)  # the start product state of the case
GEAR_IN = tuple(sorted(START + (GEAR,)))
SCREW_1_IN = tuple(sorted(GEAR_IN + (SCREWS[0],)))
SLOTS = (
    SupplySlot("gear_slot", (0.0, 0.5, 0.1, 0.0, 0.0, 0.0), part=GEAR),
    SupplySlot("screw_slot", (0.0, 0.6, 0.1, 0.0, 0.0, 0.0), screw_type="H5"),
)
ROBOT = Robot(
    "robot_1",
    1,
    tools=("H5",),
    capabilities=(Process.GRIPPING, Process.INSERTING, Process.SCREWING_IN),
)


@pytest.fixture(scope="module")
def plan_functions():
    return assembly_plan_functions(SLOTS, reference_part=BASE)


def snapshot(state):
    return Cell_Assembly_Snapshot(
        assembly_xml_with(MODEL_XML, state), CellState(mounted_tools={"robot_1": "H5"})
    )


def shell(process, target, expected_state, **parameters):
    return Operation(
        "op_1",
        process,
        target,
        (Share("robot_1", "H5", path_duration=2.0),),
        expected_product_state=expected_state,
        parameters=parameters,
    )


def plan(plan_functions, operation, before):
    return fill_shares(
        operation, MODEL, snapshot(before), {"robot_1": ROBOT}, plan_functions
    )


def test_inserting_plans_the_reverse_of_the_removal_path_of_the_part_in_the_result_assembly(
    plan_functions,
):
    planned = plan(plan_functions, shell(Process.INSERTING, GEAR, GEAR_IN), START)

    [share] = planned.shares
    path = np.array(share.path)
    assert len(path) >= 2 and share.path_duration > 0
    assert np.allclose(
        planned.direction, (0, 0, -1), atol=0.25
    )  # the gear goes in along -z
    assert path[0][2] > path[-1][2]  # approaches from above, ends seated
    assert planned.parameters[REFERENCE_PART] == BASE
    assert "plan_failure" not in planned.parameters


def test_screwing_in_fetches_its_screw_at_the_supply_then_screws_it_in(plan_functions):
    planned = plan(
        plan_functions, shell(Process.SCREWING_IN, SCREWS[0], SCREW_1_IN), GEAR_IN
    )

    [share] = planned.shares
    path = np.array(share.path)
    assert np.allclose(
        path[0], SLOTS[1].pose[:3]
    )  # the screw is fetched within the operation
    assert len(path) >= 4 and share.path_duration > 0
    assert share.skill_parameters["supply_slot"] == "screw_slot"
    assert np.dot(planned.direction, (0, 0, -1)) > 0.9  # the screw goes in along -z
    assert planned.parameters[REFERENCE_PART] == BASE


def test_screwing_in_fetches_from_the_nest_plate_that_serves_its_robot():
    other_nest = SupplySlot(
        "nest_robot_2",
        (0.0, 0.7, 0.1, 0.0, 0.0, 0.0),
        screw_type="H5",
        robots=("robot_2",),
    )
    own_nest = replace(SLOTS[1], name="nest_robot_1", robots=("robot_1",))
    functions = assembly_plan_functions((other_nest, own_nest), reference_part=BASE)

    planned = plan(
        functions, shell(Process.SCREWING_IN, SCREWS[0], SCREW_1_IN), GEAR_IN
    )

    [share] = planned.shares
    assert share.skill_parameters["supply_slot"] == "nest_robot_1"
    assert np.allclose(share.path[0], own_nest.pose[:3])


def test_share_paths_are_in_metres_although_the_product_model_is_in_millimetres(
    plan_functions,
):
    inserted = plan(plan_functions, shell(Process.INSERTING, GEAR, GEAR_IN), START)
    screwed = plan(
        plan_functions, shell(Process.SCREWING_IN, SCREWS[0], SCREW_1_IN), GEAR_IN
    )

    for operation in (inserted, screwed):
        path = np.array(operation.shares[0].path)
        assert (
            np.abs(path).max() < 1.0
        )  # the grinder is about 0.3 m long; in mm its poses reach 190
        assert np.abs(path).max() > 0.01  # and not scaled twice


def test_gripping_at_the_supply_reads_the_supply_slot(plan_functions):
    operation = shell(Process.GRIPPING, GEAR, START, **{SUPPLY_SLOT: "gear_slot"})

    planned = plan(plan_functions, operation, START)

    [share] = planned.shares
    assert share.path == (SLOTS[0].pose[:3],)
    assert share.path_duration == 2.0  # the nominal supply time stays
    assert share.skill_parameters["supply_slot"] == "gear_slot"


def test_a_missing_supply_slot_makes_the_operation_infeasible_instead_of_raising(
    plan_functions,
):
    without_slots = assembly_plan_functions((), reference_part=BASE)

    for operation, before in (
        (shell(Process.GRIPPING, GEAR, START, **{SUPPLY_SLOT: "gear_slot"}), START),
        (shell(Process.SCREWING_IN, SCREWS[0], SCREW_1_IN), GEAR_IN),
    ):
        planned = plan(without_slots, operation, before)

        assert "supply slot" in planned.parameters["plan_failure"]


def test_a_nest_plate_of_another_robot_makes_screwing_in_infeasible():
    other_nest = replace(SLOTS[1], name="nest_robot_2", robots=("robot_2",))
    functions = assembly_plan_functions((other_nest,), reference_part=BASE)

    planned = plan(
        functions, shell(Process.SCREWING_IN, SCREWS[0], SCREW_1_IN), GEAR_IN
    )

    assert "that serves robot_1" in planned.parameters["plan_failure"]


def test_a_part_that_cannot_be_inserted_makes_the_operation_infeasible(plan_functions):
    planned = plan(
        plan_functions, shell(Process.INSERTING, "no_such_part", GEAR_IN), START
    )

    assert "no_such_part" in planned.parameters["plan_failure"]


def test_removal_plan_functions_are_unchanged():
    from mrtamp.plan_functions import removal_plan_functions

    assert set(removal_plan_functions()) == {
        Process.GRIPPING_AND_REMOVING,
        Process.UNSCREWING,
    }


def test_an_assembly_plan_of_the_angle_grinder_has_a_path_for_inserting_screwing_in_and_gripping_shares():
    import networkx as nx

    from mrtamp.planning import Goal, PlanningRequest, RunSettings
    from mrtamp.v1_planner import plan as v1_plan
    from tests.mrtamp_scenarios import assembly_cell, travel_times

    graph = nx.MultiDiGraph()
    graph.add_edge(
        START,
        GEAR_IN,
        process="inserting",
        tool="H5",
        target_part=GEAR,
        action_time=30.0,
        supply_time=10.0,
        inverse=True,
    )
    graph.add_edge(
        GEAR_IN,
        SCREW_1_IN,
        process="screwing_in",
        tool="H5",
        target_part=SCREWS[0],
        action_time=10.0,
        inverse=True,
    )
    cell = assembly_cell()
    cell = replace(
        cell, robots=(replace(cell.robots[0], tools=("H5",)), cell.robots[1])
    )
    request = PlanningRequest(
        graph,
        MODEL,
        cell,
        snapshot(START),
        RunSettings(goal=Goal(present=SCREW_1_IN)),
        supply_slots=SLOTS,
        travel_times=travel_times(),
    )

    plan = v1_plan(
        request,
        plan_functions=assembly_plan_functions(SLOTS, reference_part=BASE),
    )

    by_process = {op.process: op for op in plan.operations}
    for process in (Process.GRIPPING, Process.INSERTING, Process.SCREWING_IN):
        assert by_process[process].shares[0].path, process
    assert [
        op.parameters.get(REFERENCE_PART)
        for op in plan.operations
        if op.process in (Process.INSERTING, Process.SCREWING_IN)
    ] == [BASE, BASE]
    assert not any("plan_failure" in op.parameters for op in plan.operations)
