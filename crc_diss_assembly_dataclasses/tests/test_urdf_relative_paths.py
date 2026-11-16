# The assembly URDF export writes mesh paths relative to the URDF file's folder.
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from diss_assembly_dataclasses.assembly import Assembly

SRC = Path(__file__).resolve().parent / "electric_motor"


@pytest.fixture
def assembly(tmp_path):
    folder = tmp_path / "case"
    (folder / "meshes").mkdir(parents=True)
    for name in ("01_upper_lid.stl", "02_stator.stl"):
        shutil.copy(SRC / name, folder / "meshes" / name)
    xml = folder / "case.xml"
    xml.write_text(
        '<assembly name="case">\n'
        '<part name="a" mesh_path="meshes/01_upper_lid.stl" density="1.0" material="Generic"/>\n'
        '<part name="b" mesh_path="meshes/02_stator.stl" density="1.0" material="Generic"/>\n'
        '</assembly>\n'
    )
    return Assembly.load_from_xml(str(xml))


def mesh_files(urdf):
    return [m.get("filename") for m in ET.parse(urdf).getroot().iter("mesh")]


def test_default_export_writes_relative_mesh_paths(assembly, tmp_path):
    out = tmp_path / "export"
    assembly.generate_urdf_files(str(out))
    urdfs = sorted(out.glob("*.urdf"))
    assert len(urdfs) == 2
    for urdf in urdfs:
        files = mesh_files(urdf)
        assert files
        for f in files:
            assert not Path(f).is_absolute()
            assert (urdf.parent / f).resolve() == (out / "mesh" / (urdf.stem + ".stl")).resolve()


def test_default_export_survives_moving_the_folder(assembly, tmp_path):
    out = tmp_path / "export"
    assembly.generate_urdf_files(str(out))
    moved = tmp_path / "other" / "moved"
    moved.parent.mkdir()
    shutil.move(str(out), str(moved))
    for urdf in moved.glob("*.urdf"):
        for f in mesh_files(urdf):
            assert (urdf.parent / f).is_file()


def test_explicit_mesh_folder_is_unchanged(assembly, tmp_path):
    out = tmp_path / "export"
    meshes = tmp_path / "shared_meshes"
    assembly.generate_urdf_files(str(out), str(meshes))
    for urdf in out.glob("*.urdf"):
        for f in mesh_files(urdf):
            assert f == "../shared_meshes/" + urdf.stem + ".stl"
            assert (urdf.parent / f).is_file()
