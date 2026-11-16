"""The union test product (#284) keeps the old bolt-and-nut product as it was: the old parts, and the
connections between them, are byte for byte those of the benchmark; only the cap and its connection are new."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"
UNION = ROOT / "tests" / "test_products" / "bolt_and_nut_with_cap"
XML = "xml/rotational_bolt_nut_lock_benchmark.xml"
CAP = "part_m_cap"


def _elements(folder: Path, tag: str) -> list[ET.Element]:
    return list(ET.parse(folder / XML).getroot().iter(tag))


def _mesh_name(part: ET.Element) -> str:
    return Path(part.attrib["mesh_path"]).name


def test_the_shared_parts_keep_their_names_and_material_in_the_same_order():
    old, union = _elements(OLD, "part"), _elements(UNION, "part")

    assert [p.attrib["name"] for p in union] == [
        *(p.attrib["name"] for p in old),
        CAP,
    ]
    for o, u in zip(old, union):
        assert {k: v for k, v in u.attrib.items() if k != "mesh_path"} == {
            k: v for k, v in o.attrib.items() if k != "mesh_path"
        }
        assert _mesh_name(u) == _mesh_name(o)


def test_the_shared_parts_keep_their_poses_as_their_meshes_carry_them():
    # a part's pose is the one its mesh is modelled in: the mesh files are the same bytes
    for part in _elements(OLD, "part"):
        name = _mesh_name(part)
        assert (UNION / "stl" / name).read_bytes() == (OLD / "stl" / name).read_bytes()


def test_the_shared_connections_keep_their_parts_and_their_contact_data():
    old = _elements(OLD, "connection")
    union = _elements(UNION, "connection")

    assert [c.attrib for c in union[: len(old)]] == [c.attrib for c in old]
    data = "xml/rotational_bolt_nut_lock_benchmark_data"
    for c in old:
        for key in ("points_filename", "normals_filename", "distances_filename"):
            name = Path(c.attrib[key]).name
            assert np.array_equal(
                np.load(UNION / data / name), np.load(OLD / "xml" / Path(c.attrib[key]))
            )


def test_the_cap_has_one_connection_with_its_own_contact_data():
    old = len(_elements(OLD, "connection"))
    [new] = _elements(UNION, "connection")[old:]

    assert {new.attrib["part_name1"], new.attrib["part_name2"]} == {
        "part_j_elbow_pipe",
        CAP,
    }  # the cap touches the pipe only
    assert new.attrib["type"] == "geometric"
    assert (UNION / "stl" / f"{CAP}.stl").is_file()
    assert len(np.load(UNION / "xml" / new.attrib["points_filename"])) > 0
