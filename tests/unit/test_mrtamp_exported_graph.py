"""The run controller plans from an exported sequence graph (#104)."""

import json

import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_export import (
    UnsupportedSchemaVersion,
    export_graph,
    make_provenance,
)
from disassembly_pomdp.travel_time_table import TravelTimeTable
from mrtamp.cell_file import CellFile, save_cell_file
from mrtamp.exported_graph import load_export, run_from_export
from mrtamp.planning import Goal, RunSettings
from mrtamp.run_controller import RunRecord
from mrtamp.stand_in import ModelLevelStandIn
from mrtamp.v1_planner import plan as v1_plan
from tests.mrtamp_scenarios import (
    START_XML,
    lid_screwed_on_base_graph,
    transformer_cell,
    travel_times,
)


@pytest.fixture
def exported(tmp_path):
    """A product folder with its model, and the export of the lid-on-base graph made from it."""
    folder = tmp_path / "lid_screwed_on_base"
    (folder / "xml").mkdir(parents=True)
    (folder / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
    file = tmp_path / "graph.json"
    export_graph(
        lid_screwed_on_base_graph(),
        file,
        make_provenance(
            "lid_screwed_on_base", folder, "xml/model.xml", export_folder=tmp_path
        ),
    )
    return file, folder


def test_load_gives_the_graph_and_the_provenance_and_no_warning_for_a_matching_product_folder(
    exported,
):
    file, folder = exported

    export = load_export(file, product_folder=folder)

    assert export.graph.number_of_edges() == 3
    assert export.provenance["product_name"] == "lid_screwed_on_base"
    assert export.product_folder == str(folder)
    assert export.warnings == ()


def test_assembly_of_any_product_state_is_the_product_model_minus_the_removed_parts(
    exported,
):
    export = load_export(exported[0], product_folder=exported[1])

    snapshot = export.snapshot_for(
        ("base", "lid"), mounted_tools={"robot_1": "H5", "robot_2": None}
    )

    assert snapshot.product_state_key == ("base", "lid")
    assert (
        'name="screw"' not in snapshot.assembly_xml
        and "screw_head" not in snapshot.assembly_xml
    )
    assert snapshot.cell_state.mounted_tools == {"robot_1": "H5", "robot_2": None}
    assert export.snapshot_for(("base", "lid", "screw")).product_state_key == (
        "base",
        "lid",
        "screw",
    )


def test_unknown_schema_version_is_refused(exported):
    file, _ = exported
    data = json.loads(file.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersion):
        load_export(file)


def test_fingerprint_mismatch_warns_and_the_run_record_keeps_the_warning(exported):
    file, folder = exported
    (folder / "xml" / "model.xml").write_text(
        START_XML.replace("8000.0", "7850.0"), encoding="utf-8"
    )

    export = load_export(file, product_folder=folder)
    record = run_from_export(
        export,
        transformer_cell(),
        RunSettings(goal=Goal(absent=("lid",))),
        v1_plan,
        ModelLevelStandIn(),
        mounted_tools={"robot_1": "H5", "robot_2": None},
        travel_times=travel_times(),
    )

    [warning] = export.warnings
    assert "product model" in warning and "xml/model.xml" in warning
    assert (
        record.status == "goal_reached"
    )  # a stale export warns but does not block the run
    assert record.warnings == [warning]
    assert json.loads(json.dumps(record.to_dict()))["warnings"] == [warning]


def test_a_missing_model_file_is_a_fingerprint_mismatch_too(exported):
    file, folder = exported
    (folder / "xml" / "model.xml").unlink()

    assert any(
        "product model" in w for w in load_export(file, product_folder=folder).warnings
    )


def test_the_product_folder_can_be_given_when_the_export_moved(exported, tmp_path):
    file, folder = exported
    moved = tmp_path / "elsewhere"
    folder.rename(moved)

    assert load_export(file, product_folder=folder).warnings  # the folder is gone
    assert load_export(file, product_folder=moved).warnings == ()


def test_run_from_an_export_keeps_a_copy_of_the_graph_and_the_provenance_in_the_run_record(
    exported,
):
    export = load_export(exported[0], product_folder=exported[1])

    record = run_from_export(
        export,
        transformer_cell(),
        RunSettings(goal=Goal(absent=("lid",))),
        v1_plan,
        ModelLevelStandIn(),
        mounted_tools={"robot_1": "H5", "robot_2": None},
        travel_times=travel_times(),
    )

    assert record.status == "goal_reached"
    assert [op.process for op in record.operation_plans[-1].operations][
        0
    ] is Process.UNSCREWING
    d = record.to_dict()
    assert (
        len(d["sequence_graph"]["edges"]) == 3
        and d["product_folder"] == export.product_folder
    )
    assert d["provenance"] == export.provenance and d["warnings"] == []
    assert isinstance(record, RunRecord)


def test_an_export_with_portable_paths_is_read_with_the_reader_unchanged(tmp_path):
    folder = tmp_path / "lid_screwed_on_base"
    (folder / "xml").mkdir(parents=True)
    (folder / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
    cell = tmp_path / "cell.json"
    cell.write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    provenance = make_provenance(
        "lid_screwed_on_base",
        folder,
        "xml/model.xml",
        cell_file=cell,
        export_folder=out,
    )
    export_graph(lid_screwed_on_base_graph(), out / "graph.json", provenance)

    export = load_export(out / "graph.json", product_folder=folder, cell_file=cell)

    assert export.provenance == provenance and export.warnings == ()
    assert provenance["product_folder"] == "../lid_screwed_on_base"
    assert provenance["cell_file"] == "../cell.json"
    assert "<assembly" in export.product_model_xml


def test_a_changed_travel_time_table_is_a_cell_fingerprint_mismatch(tmp_path):
    """The table belongs to the cell (ADR-0012): changing a time makes the export's graph possibly stale."""
    folder = tmp_path / "lid_screwed_on_base"
    (folder / "xml").mkdir(parents=True)
    (folder / "xml" / "model.xml").write_text(START_XML, encoding="utf-8")
    cell = tmp_path / "cell.json"
    save_cell_file(CellFile(transformer_cell(), travel_times=travel_times()), cell)
    provenance = make_provenance(
        "lid_screwed_on_base",
        folder,
        "xml/model.xml",
        cell_file=cell,
        export_folder=tmp_path,
    )
    export_graph(lid_screwed_on_base_graph(), tmp_path / "graph.json", provenance)
    save_cell_file(
        CellFile(
            transformer_cell(),
            travel_times=TravelTimeTable(
                {**travel_times().times, "toolchange_time_at_toolstorage": 30.0}
            ),
        ),
        cell,
    )

    export = load_export(tmp_path / "graph.json", product_folder=folder, cell_file=cell)

    assert any(
        "cell file" in w and "fingerprint mismatch" in w for w in export.warnings
    )


def test_the_export_loads_the_product_model_its_provenance_names_not_the_one_at_the_root(
    exported,
):
    file, folder = exported
    (folder / "stray.xml").write_text("<assembly name='stray'/>", encoding="utf-8")

    export = load_export(file, product_folder=folder)

    assert export.product_model.xml_path == folder / "xml" / "model.xml"
    assert export.warnings == ()
    assert export.snapshot_for(("base", "lid")).product_state_key == ("base", "lid")
