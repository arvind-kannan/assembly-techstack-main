"""The demo products on the Transformer cell: the two cell files and the sequence graph exports of the angle
grinder and the battery replacement product (#150)."""

import json
from pathlib import Path

import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_extraction import FULL_TEARDOWN
from disassembly_pomdp.graph_steps import Step
from mrtamp.cell_file import load_cell_file
from mrtamp.exported_graph import assembly_from_scratch_start, load_export
from mrtamp.step_expansion import expand_step
from mrtamp.supply import slot_for_part, slots_in_cell_frame

CASES = Path(__file__).resolve().parents[2] / "cases"
REAL_CELL, SIMULATION_CELL = (
    CASES / "transformer_cell.json",
    CASES / "transformer_cell_simulation.json",
)


def screwing_robots(cell_file):
    return {
        r.id
        for r in load_cell_file(cell_file).cell.robots
        if r.can_carry_out(Process.SCREWING_IN, "H5")
    }


def test_on_the_real_cell_only_robot_2_screws_in():
    assert screwing_robots(REAL_CELL) == {"robot_2"}


def test_on_the_real_cell_only_robot_1_inserts():
    cell = load_cell_file(REAL_CELL).cell
    assert {
        r.id
        for r in cell.robots
        if r.can_carry_out(Process.INSERTING, "default_gripper")
    } == {"robot_1"}


def test_in_the_cell_simulation_both_robots_screw_in():
    assert screwing_robots(SIMULATION_CELL) == {"robot_1", "robot_2"}


def slots_of(cell_file):
    return slots_in_cell_frame(load_cell_file(cell_file).cell)


def test_in_the_cell_simulation_each_screwing_robot_has_its_own_screw_nest_plate():
    slots = slots_of(SIMULATION_CELL)
    nests = {
        robot: [s.name for s in slots if s.screw_type == "H5" and s.serves(robot)]
        for robot in screwing_robots(SIMULATION_CELL)
    }
    assert all(len(names) == 1 for names in nests.values())
    assert nests["robot_1"] != nests["robot_2"]


def test_the_simulation_slots_stay_where_they_were_before_they_moved_into_the_supply_station():
    # the cell-frame poses of the cell file before AT 174, when slots were listed with absolute poses
    assert {s.name: s.pose[:3] for s in slots_of(SIMULATION_CELL)} == {
        "crown_gear_slot": pytest.approx((-1.2, -2.6, 0.8)),
        "lid_slot": pytest.approx((-0.8, -2.6, 0.8)),
        "screw_nest_plate_h5_robot_1": pytest.approx((-1.6, -1.0, 0.8)),
        "screw_nest_plate_h5_robot_2": pytest.approx((-1.8, -2.4, 0.8)),
    }


@pytest.mark.parametrize(
    "cell_file", [REAL_CELL, SIMULATION_CELL], ids=["real_cell", "cell_simulation"]
)
def test_both_cell_files_supply_the_crown_gear_the_lid_and_the_screws(cell_file):
    data, slots = load_cell_file(cell_file), slots_of(cell_file)
    assert slot_for_part(slots, "BG_Tellerrad_V1p") is not None
    assert slot_for_part(slots, "lid") is not None
    assert any(s.screw_type == "H5" for s in slots)
    assert [
        r.id for r in data.cell.robots if r.can_carry_out(Process.INSPECTING, None)
    ] == ["robot_2"]


@pytest.fixture(scope="module", params=["angle_grinder", "battery_replacement"])
def export(request, tmp_path_factory):
    from examples.export_demo_products import export_demo_product

    return request.param, export_demo_product(
        request.param, tmp_path_factory.mktemp(request.param)
    )


@pytest.mark.slow
def test_the_export_of_a_demo_product_loads_without_warnings(export):
    product, graph_file = export
    loaded = load_export(graph_file, product_folder=CASES / product / "model")
    assert loaded.warnings == ()
    assert loaded.provenance["product_setup"] == "product_setup.json"
    assert (
        loaded.provenance["cell_file"] is None
    )  # the export does not depend on the cell file


@pytest.mark.slow
def test_the_export_of_a_demo_product_is_a_full_teardown_down_to_its_base_parts(
    export,
):
    product, graph_file = export
    loaded = load_export(graph_file, product_folder=CASES / product / "model")
    setup = json.loads(
        (CASES / product / "model" / "product_setup.json").read_text(encoding="utf-8")
    )

    assert loaded.provenance["extraction_depth"] == FULL_TEARDOWN
    assert assembly_from_scratch_start(loaded.graph, setup["base_parts"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_file", [REAL_CELL, SIMULATION_CELL], ids=["real_cell", "cell_simulation"]
)
def test_some_robot_of_the_cell_carries_out_every_assembly_step(export, cell_file):
    product, graph_file = export
    cell = load_cell_file(cell_file)
    steps = [
        s
        for _, _, s in load_export(
            graph_file, product_folder=CASES / product / "model"
        ).graph.edges(data=True)
        if s.get("inverse")
    ]
    assert {s["process"] for s in steps} == {"inserting", "screwing_in"}
    for attrs in steps:
        step = Step.from_edge(attrs)
        operations = expand_step(step, slots_in_cell_frame(cell.cell))
        assert any(
            all(r.can_carry_out(op.process, step.tool) for op in operations)
            for r in cell.cell.robots
        ), attrs


def test_only_the_cell_simulation_file_is_simulation_only():
    assert load_cell_file(SIMULATION_CELL).simulation_only is True
    assert load_cell_file(REAL_CELL).simulation_only is False
