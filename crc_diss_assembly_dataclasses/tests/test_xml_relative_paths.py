# Relative paths in an assembly XML resolve against the XML file's folder.
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from diss_assembly_dataclasses.assembly import Assembly

SRC = Path(__file__).resolve().parent / "electric_motor"


@pytest.fixture
def case_folder(tmp_path):
    """A case folder with two meshes and one geometric connection, XML written with relative paths."""
    folder = tmp_path / "case"
    (folder / "meshes").mkdir(parents=True)
    for name in ("01_upper_lid.stl", "02_stator.stl"):
        shutil.copy(SRC / name, folder / "meshes" / name)
    xml = folder / "case.xml"
    data = folder / "case_data"
    data.mkdir()
    for kind in ("points", "normals", "distances"):
        np.save(data / f"a_b_{kind}.npy", np.zeros((1, 3)))
    xml.write_text(
        '<assembly name="case">\n'
        '<part name="a" mesh_path="meshes/01_upper_lid.stl" density="1.0" material="Generic"/>\n'
        '<part name="b" mesh_path="meshes' + chr(92) + '02_stator.stl" density="1.0" material="Generic"/>\n'
        '<connection type="geometric" part_name1="a" part_name2="b" '
        'points_filename="case_data/a_b_points.npy" normals_filename="case_data/a_b_normals.npy" '
        'distances_filename="case_data/a_b_distances.npy"/>\n'
        '</assembly>\n'
    )
    return xml


def test_load_resolves_relative_paths_from_any_working_directory(case_folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asm = Assembly.load_from_xml(str(case_folder))
    for part in asm.parts.values():
        assert os.path.isfile(part.mesh_path)
        assert part.mesh is not None
    assert len(asm.connections) == 1


def test_load_keeps_absolute_paths(case_folder, tmp_path, monkeypatch):
    absolute = (case_folder.parent / "meshes" / "01_upper_lid.stl").resolve()
    text = case_folder.read_text().replace("meshes/01_upper_lid.stl", absolute.as_posix())
    case_folder.write_text(text)
    monkeypatch.chdir(tmp_path)
    asm = Assembly.load_from_xml(str(case_folder))
    assert Path(asm.parts["a"].mesh_path) == absolute


def test_save_writes_paths_relative_to_the_xml_folder(case_folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asm = Assembly.load_from_xml(str(case_folder))
    out = tmp_path / "elsewhere" / "saved.xml"
    out.parent.mkdir()
    asm.save_as_xml(str(out))
    text = out.read_text()
    assert str(tmp_path).replace("\\", "/") not in text.replace("\\", "/")
    reloaded = Assembly.load_from_xml(str(out))
    assert set(reloaded.parts) == {"a", "b"}
    assert len(reloaded.connections) == 1
