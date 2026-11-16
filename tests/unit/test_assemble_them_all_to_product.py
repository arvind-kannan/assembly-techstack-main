"""The Assemble-Them-All converter (#226): a synthetic assembly in dataset layout becomes a product folder."""

import importlib.util
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from product_explorer.product_folder import load_product_folder

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "assemble_them_all_to_product",
    ROOT / "tools" / "import_tool" / "assemble_them_all_to_product.py",
)
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

# Boxes as (min corner, max corner) in dataset units: a plate with a block standing on it and a second block
# beside the first, all in the assembled state.
BOXES = {
    "0": ((-2, -1.5, 0), (2, 1.5, 0.5)),
    "1": ((-1.5, -1, 0.5), (0, 1, 2)),
    "2": ((0, -1, 0.5), (1.5, 1, 1.5)),
}


def write_obj_box(path, lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    box = o3d.geometry.TriangleMesh.create_box(*(hi - lo))
    box.translate(lo)
    o3d.io.write_triangle_mesh(str(path), box)


@pytest.fixture
def case_dir(tmp_path):
    folder = tmp_path / "dataset" / "general" / "00001"
    folder.mkdir(parents=True)
    for stem, (lo, hi) in BOXES.items():
        write_obj_box(folder / f"{stem}.obj", lo, hi)
    return folder


@pytest.fixture
def product_xml(case_dir, tmp_path):
    return convert.convert_assembly(case_dir, tmp_path / "products")


def test_the_converter_writes_the_product_folder(product_xml):
    folder = product_xml.parent
    assert product_xml.name == "ata_general_00001.xml"
    assert (folder / "provenance.json").is_file()
    assert sorted(p.name for p in (folder / folder.name).glob("*.stl")) == [
        "part_0.stl",
        "part_1.stl",
        "part_2.stl",
    ]
    assert list((folder / f"{folder.name}_data").glob("*.npy"))


def test_the_product_loads_with_contacts_and_the_largest_part_as_base(product_xml):
    _, assembly, base, target = load_product_folder(product_xml.parent)
    assert sorted(assembly.parts) == ["part_0", "part_1", "part_2"]
    assert base == ["part_0"]
    assert target == ["part_1", "part_2"]
    touching = {frozenset((c.part_name1, c.part_name2)) for c in assembly.connections}
    assert frozenset(("part_0", "part_1")) in touching
    assert frozenset(("part_1", "part_2")) in touching


def test_the_parts_are_scaled_to_millimetres(product_xml):
    _, assembly, _, _ = load_product_folder(product_xml.parent)
    extent = assembly.parts["part_0"].mesh.get_axis_aligned_bounding_box().get_extent()
    assert np.allclose(extent, [40, 30, 5])


def test_the_xml_holds_only_relative_paths(product_xml):
    text = product_xml.read_text(encoding="utf-8")
    assert str(product_xml.parent) not in text
    assert "\\" not in text


def test_a_named_base_overrides_the_largest_part(case_dir, tmp_path):
    xml = convert.convert_assembly(case_dir, tmp_path / "p", base="part_2")
    setup = json.loads((xml.parent / "product_setup.json").read_text(encoding="utf-8"))
    assert setup["base_parts"] == ["part_2"]


def test_an_unknown_base_is_refused(case_dir, tmp_path):
    with pytest.raises(ValueError, match="part_9"):
        convert.convert_assembly(case_dir, tmp_path / "p", base="part_9")


def test_a_translation_file_is_refused_instead_of_ignored(case_dir, tmp_path):
    (case_dir / "translation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="translation.json"):
        convert.convert_assembly(case_dir, tmp_path / "p")


def test_a_folder_without_parts_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        convert.convert_assembly(tmp_path / "empty", tmp_path / "p")


ROTATION_ZIP = "https://people.csail.mit.edu/yunsheng/Assemble-Them-All/dataset_2403/joint_assembly_rotation.zip"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ATA_NETWORK"),
    reason="downloads 5.4 MB from people.csail.mit.edu; set ATA_NETWORK=1",
)
def test_a_real_two_part_assembly_converts(tmp_path):
    zip_path = tmp_path / "rotation.zip"
    urllib.request.urlretrieve(ROTATION_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_path / "data")
    case = tmp_path / "data" / "joint_assembly_rotation" / "general" / "06397"
    xml = convert.convert_assembly(case, tmp_path / "products")
    _, assembly, base, target = load_product_folder(xml.parent)
    assert len(assembly.parts) == 2 and len(base) == len(target) == 1
    assert assembly.connections
