"""Reassembly of the angle grinder's bevel-gear-spindle assembly from the stored baseline export (AT 270).

The export (`tests/test_products/angle_grinder_baseline/`) is the Product explorer's default depth: the graph
stops at the disassembled states. Its one disassembled state is the grinder without `BG_Tellerrad_V1p` (the
target part) and without the four screws that hold it; the reassembly inserts the gear from the supply and
screws the four screws back in. B1 plans it against the model-level stand-in, so no cell simulation is needed.

Regenerate the export with `python examples/export_demo_products.py --baseline` after a change to the grinder's
model or product setup; the fingerprint test below fails until then.
"""

import json
from pathlib import Path

import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_extraction import DISASSEMBLED_STATES
from motion_planning.assembly_plan_functions import assembly_plan_functions
from mrtamp.b1_planner import make_b1_planner
from mrtamp.cell_file import load_cell_file
from mrtamp.exported_graph import disassembled_state, load_export, run_from_export
from mrtamp.planning import Goal, RunSettings
from mrtamp.screw_groups import part_positions
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.supply import slots_in_cell_frame

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests" / "test_products" / "angle_grinder_baseline"
MODEL = ROOT / "cases" / "angle_grinder" / "model"
CELL_FILE = ROOT / "cases" / "transformer_cell_simulation.json"
GEAR = "BG_Tellerrad_V1p"
SCREWS = tuple(f"Schraube_M4x16_540_Schr{i}" for i in range(1, 5))


@pytest.fixture(scope="module")
def baseline():
    return load_export(
        BASELINE / "sequence_graph.json", product_folder=MODEL, cell_file=CELL_FILE
    )


def test_the_baseline_is_current_with_the_product_model_and_setup(baseline):
    assert baseline.warnings == ()
    assert baseline.provenance["extraction_depth"] == DISASSEMBLED_STATES
    setup = json.loads((MODEL / "product_setup.json").read_text(encoding="utf-8"))
    assert setup["target_parts"] == [GEAR]


def test_the_baseline_has_one_disassembled_state_without_the_gear_and_its_screws(
    baseline,
):
    everything = tuple(sorted(max(baseline.graph.nodes, key=len)))
    disassembled = disassembled_state(baseline.graph)

    assert disassembled == tuple(sorted(set(everything) - {GEAR, *SCREWS}))
    assert not [
        e
        for _, _, e in baseline.graph.out_edges(disassembled, data=True)
        if not e.get("inverse")
    ]  # the graph stops at the disassembled state: no step takes a part out of it


@pytest.mark.integration
@pytest.mark.geom
def test_reassembly_from_the_disassembled_state_reaches_the_full_product(
    baseline, tmp_path
):
    cell = load_cell_file(CELL_FILE)
    everything = tuple(sorted(max(baseline.graph.nodes, key=len)))
    reference_part = json.loads(
        (MODEL / "product_setup.json").read_text(encoding="utf-8")
    )["base_parts"][0]
    functions = assembly_plan_functions(
        slots_in_cell_frame(cell.cell), reference_part=reference_part
    )

    record = run_from_export(
        baseline,
        cell.cell,
        RunSettings(goal=Goal(present=everything)),
        make_b1_planner(
            functions,
            part_positions=part_positions(
                baseline.product_model_xml, baseline.product_folder
            ),
        ),
        ModelLevelStandIn(),
        start_state=disassembled_state(baseline.graph),
        mounted_tools=cell.mounted_tools,
        supply_slots=slots_in_cell_frame(cell.cell),
        simulation_only=cell.simulation_only,
        travel_times=cell.travel_times,
    )

    assert record.status == "goal_reached"
    assert tuple(sorted(record.final_snapshot.product_state_key)) == everything
    processes = [
        op.process for plan in record.operation_plans for op in plan.operations
    ]
    assert processes.count(Process.INSERTING) == 1
    assert processes.count(Process.SCREWING_IN) == len(SCREWS)
    assert record.provenance["extraction_depth"] == DISASSEMBLED_STATES
