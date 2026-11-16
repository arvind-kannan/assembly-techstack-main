"""Minimum architecture example: the bolt-and-nut case from extraction to goal with v1 + stand-in."""

import json
import sys
from pathlib import Path

import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.travel_time_table import table_path
from tests.sim_loopback import loopback_simulation, requires_sim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))


@pytest.mark.integration
@pytest.mark.geom
def test_bolt_and_nut_case_runs_from_extraction_to_goal(tmp_path):
    from mrtamp_walking_skeleton import run_bolt_and_nut

    record, graph = run_bolt_and_nut(tmp_path)

    [(_, _, step)] = graph.edges(
        data=True
    )  # the nut is a rotational lock: only the lid comes off
    assert (step["target_part"], step["process"]) == (
        "part_l_cover_lid",
        Process.GRIPPING_AND_REMOVING.value,
    )
    assert record.status == "goal_reached"
    plan = record.operation_plans[-1]
    assert plan.metrics.inspections == 1 and plan.metrics.makespan > 0
    saved = json.loads((tmp_path / "run_record.json").read_text(encoding="utf-8"))
    assert saved["status"] == "goal_reached"
    assert saved["entries"][0]["operation_plan"]["metrics"] == plan.metrics.to_dict()


@pytest.mark.integration
@pytest.mark.geom
def test_bolt_and_nut_case_is_assembled_from_its_one_disassembled_state(
    tmp_path,
):
    from mrtamp_assembly_run import LID, assemble_bolt_and_nut

    from disassembly_pomdp.graph_extraction import DISASSEMBLED_STATES

    record, graph = assemble_bolt_and_nut(tmp_path)

    assert [n for n, d in graph.nodes(data=True) if d.get("is_goal")] == [
        ("part_i_bolt", "part_j_elbow_pipe", "part_k_nut")
    ]
    assert record.provenance["extraction_depth"] == DISASSEMBLED_STATES
    first = record.entries[0].plan_result.operations[0]
    assert (first.process, first.target) == (Process.GRIPPING, LID)
    assert any(
        s.get("inverse") and s["target_part"] == LID
        for _, _, s in graph.edges(data=True)
    )
    assert record.status == "goal_reached"
    plan = record.operation_plans[-1]
    assert [op.process for op in plan.operations] == [
        Process.GRIPPING,
        Process.INSERTING,
        Process.RELEASING,
        Process.INSPECTING,
    ]
    assert LID in record.final_snapshot.product_state_key
    assert (
        json.loads((tmp_path / "run_record.json").read_text(encoding="utf-8"))["status"]
        == "goal_reached"
    )


@pytest.mark.integration
@pytest.mark.geom
def test_bolt_and_nut_plan_has_a_path_for_every_share_from_the_removal_plan_function(
    tmp_path,
):
    from mrtamp_walking_skeleton import run_bolt_and_nut

    record, _ = run_bolt_and_nut(tmp_path)

    removal, inspection = record.operation_plans[-1].operations
    assert removal.direction is not None
    assert all(
        len(share.path) >= 2 and share.path_duration > 0 for share in removal.shares
    )
    assert record.simulation_reports[-1].infeasible is None


@pytest.mark.integration
@pytest.mark.geom
def test_the_walking_skeleton_runs_from_an_exported_file_without_extracting(tmp_path):
    from mrtamp_walking_skeleton import (
        export_case,
        extract_bolt_and_nut,
        run_bolt_and_nut,
        transformer_cell,
    )

    env, graph = extract_bolt_and_nut()
    graph_file, cell_file = export_case(tmp_path, graph, env, transformer_cell(graph))
    header = json.loads(graph_file.read_text(encoding="utf-8"))["provenance"]

    record, _ = run_bolt_and_nut(
        tmp_path / "run", graph_file
    )  # the cell file is the one named in the provenance

    assert record.status == "goal_reached" and record.warnings == []
    assert (
        header["fingerprints"]["product_model_xml"]
        and header["fingerprints"]["cell_file"]
    )
    saved = json.loads(
        (tmp_path / "run" / "run_record.json").read_text(encoding="utf-8")
    )
    assert saved["provenance"] == header and len(saved["sequence_graph"]["edges"]) == 1


@pytest.mark.integration
@pytest.mark.geom
def test_a_changed_cell_file_warns_but_the_run_goes_on(tmp_path):
    from mrtamp_walking_skeleton import (
        export_case,
        extract_bolt_and_nut,
        run_bolt_and_nut,
        transformer_cell,
    )

    env, graph = extract_bolt_and_nut()
    graph_file, cell_file = export_case(tmp_path, graph, env, transformer_cell(graph))
    table = table_path(cell_file)  # a changed time changes the cell (ADR-0012)
    table.write_text(
        table.read_text(encoding="utf-8").replace(
            '"inspection_time": 10.0', '"inspection_time": 12.0'
        ),
        encoding="utf-8",
    )

    record, _ = run_bolt_and_nut(tmp_path / "run", graph_file, cell_file)

    assert record.status == "goal_reached"
    [warning] = record.warnings
    assert "cell file" in warning


@pytest.mark.integration
@pytest.mark.geom
@requires_sim
def test_the_walking_skeleton_runs_against_the_real_cell_simulation_instead_of_the_stand_in(
    tmp_path,
):
    from mrtamp_walking_skeleton import run_bolt_and_nut

    simulation, _ = loopback_simulation()

    record, _ = run_bolt_and_nut(tmp_path, cell_simulation=simulation)

    assert record.status == "goal_reached"
    report = record.simulation_reports[-1]
    assert (
        report.infeasible is None and report.nc_programs is None
    )  # a simulation-targeted run needs no NC
    kinds = [t["kind"] for t in report.timeline]
    assert (
        "transit" in kinds and "operation" in kinds
    )  # the times come from planned motion, not nominal ones
    assert report.metrics != record.operation_plans[-1].metrics
