from pathlib import Path

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Geometric_Connection

from disassembly_pomdp.product_model import ProductModel, ProductModelError
from motion_planning import common

ROOT = Path(__file__).resolve().parents[2]
BOLT_AND_NUT = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"
ANGLE_GRINDER = ROOT / "cases" / "angle_grinder" / "model"
EMO = ROOT / "emo_sr_v2"


def _xml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<assembly name='p'/>", encoding="utf-8")
    return path


def test_the_rule_takes_the_single_xml_at_the_folder_root(tmp_path):
    xml = _xml(tmp_path / "p.xml")
    _xml(tmp_path / "xml" / "other.xml")

    assert ProductModel.find(tmp_path).xml_path == xml


def test_the_rule_takes_the_single_xml_in_the_xml_subfolder_when_the_root_has_none(
    tmp_path,
):
    xml = _xml(tmp_path / "xml" / "p.xml")

    assert ProductModel.find(tmp_path).xml_path == xml


def test_the_provenance_path_wins_over_the_root(tmp_path):
    _xml(tmp_path / "a.xml")
    named = _xml(tmp_path / "model" / "b.xml")

    model = ProductModel.find(tmp_path, provenance_xml="model/b.xml")

    assert model.xml_path == named
    assert model.relative_xml == "model/b.xml"


def test_the_provenance_path_is_taken_even_when_its_file_is_not_there(tmp_path):
    # the export loader warns about a missing product model; it does not refuse the export
    _xml(tmp_path / "a.xml")

    model = ProductModel.find(tmp_path, provenance_xml="gone.xml")

    assert model.xml_path == tmp_path / "gone.xml"


def test_a_folder_without_a_product_model_is_an_error_naming_the_folder(tmp_path):
    with pytest.raises(ProductModelError, match=str(tmp_path).replace("\\", "\\\\")):
        ProductModel.find(tmp_path)


def test_several_xmls_at_the_root_are_an_error_naming_them(tmp_path):
    _xml(tmp_path / "a.xml")
    _xml(tmp_path / "b.xml")
    _xml(tmp_path / "xml" / "c.xml")  # the root is not skipped for the subfolder

    with pytest.raises(ProductModelError, match=r"a\.xml.*b\.xml"):
        ProductModel.find(tmp_path)


def test_several_xmls_in_the_xml_subfolder_are_an_error_naming_them(tmp_path):
    _xml(tmp_path / "xml" / "a.xml")
    _xml(tmp_path / "xml" / "b.xml")

    with pytest.raises(ProductModelError, match=r"a\.xml.*b\.xml"):
        ProductModel.find(tmp_path)


def test_the_error_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        ProductModel.find(tmp_path)


@pytest.mark.parametrize(
    "folder, relative",
    [
        (BOLT_AND_NUT, "xml/rotational_bolt_nut_lock_benchmark.xml"),
        (ANGLE_GRINDER, "angle_grinder.xml"),
        (EMO, "EMO_SR_V2.xml"),
    ],
    ids=["bolt-and-nut benchmark", "case folder", "EMO SR V2"],
)
def test_the_repository_product_folders_load_through_the_rule(folder, relative):
    model = ProductModel.find(folder)

    assert model.relative_xml == relative
    assert model.assembly().parts
    assert model.folder == folder


def test_the_snapshot_at_a_product_state_matches_the_helper():
    model = ProductModel.find(BOLT_AND_NUT)
    state = ("part_j_elbow_pipe", "part_l_cover_lid")
    cell_state = CellState(mounted_tools={"robot_1": "H5"})

    assert model.snapshot_at(state, cell_state) == common.snapshot_at(
        model.xml, state, cell_state
    )


def test_the_snapshot_defaults_to_an_empty_cell_state():
    model = ProductModel.find(EMO)

    assert model.snapshot_at(("02_stator",)).cell_state == CellState()


def test_the_assembly_of_a_snapshot_resolves_paths_against_the_xml_folder():
    # the benchmark's mesh paths ("../stl/...") are relative to its xml subfolder
    model = ProductModel.find(BOLT_AND_NUT)
    snapshot = model.snapshot_at(("part_i_bolt", "part_j_elbow_pipe"))

    assembly = model.assembly_of(snapshot)
    expected = common.assembly_of(snapshot, BOLT_AND_NUT / "xml")

    assert sorted(assembly.parts) == ["part_i_bolt", "part_j_elbow_pipe"]
    assert [len(p.mesh.vertices) for p in assembly.parts.values()] == [
        len(p.mesh.vertices) for p in expected.parts.values()
    ]


def test_loading_a_snapshots_assembly_leaves_the_working_directory_unchanged(
    tmp_path, monkeypatch
):
    model = ProductModel.find(BOLT_AND_NUT)
    snapshot = model.snapshot_at(("part_i_bolt", "part_j_elbow_pipe"))
    monkeypatch.chdir(tmp_path)

    assembly = common.assembly_of(snapshot, BOLT_AND_NUT / "xml")

    assert Path.cwd() == tmp_path
    assert sorted(assembly.parts) == ["part_i_bolt", "part_j_elbow_pipe"]


def test_the_assembly_of_the_full_product_equals_the_one_the_core_loads(
    tmp_path, monkeypatch
):
    model = ProductModel.find(BOLT_AND_NUT)
    snapshot = model.snapshot_at(
        tuple(Assembly.load_from_xml(str(model.xml_path)).parts)
    )
    expected = Assembly.load_from_xml(str(model.xml_path))
    monkeypatch.chdir(tmp_path)

    assembly = common.assembly_of(snapshot, model.xml_path.parent)

    assert {n: p.mesh_path for n, p in assembly.parts.items()} == {
        n: p.mesh_path for n, p in expected.parts.items()
    }
    assert len(assembly.connections) == len(expected.connections)
    geometric = [c for c in expected.connections if isinstance(c, Geometric_Connection)]
    assert geometric
    for want in geometric:
        got = next(
            c
            for c in assembly.connections
            if isinstance(c, Geometric_Connection)
            and (c.part_name1, c.part_name2) == (want.part_name1, want.part_name2)
        )
        for attr in ("points", "normals", "distances"):
            np.testing.assert_array_equal(getattr(got, attr), getattr(want, attr))
