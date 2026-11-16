"""Generate the union test product (#284) in `tests/test_products/bolt_and_nut_with_cap/`.

The bolt-and-nut benchmark plus one extra part, a disk cap on the free +Y opening of the elbow pipe: far from
the bolt and the lid, so it can come out first, before any old part. The product is the union product model
of a product that gains a part its old model lacks (#268): the old parts, the new one, in one product folder.

The four old parts keep their meshes and their two connections byte for byte; only the cap and its one
connection (to the pipe) are new, found by contact detection on that pair alone with the benchmark's
tolerances. Run it from the repository root after changing the cap, and commit the result:

    python tools/make_union_bolt_and_nut_product.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "crc_diss_assembly_dataclasses" / "src"))
from diss_assembly_dataclasses.assembly import Assembly  # noqa: E402
from diss_assembly_dataclasses.part import Part  # noqa: E402

import disassembly_pomdp  # noqa: E402,F401  (sets up the import paths)

BENCHMARK = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"
NAME = "bolt_and_nut_with_cap"
FOLDER = ROOT / "tests" / "test_products" / NAME
XML_NAME = "rotational_bolt_nut_lock_benchmark.xml"
CAP = "part_m_cap"  # sorts after the four old parts, so their order stays as it was
PIPE = "part_j_elbow_pipe"
SETUP = {
    "base_parts": [PIPE],
    "target_parts": ["part_l_cover_lid", CAP],
}

# From the benchmark's metadata: the +Y straight pipe is centred on x=51, z=0, 70 mm long, outer radius 21.
CAP_CENTRE_X = 51.0
CAP_FACE_Y = 70.0  # the pipe's free +Y end face
CAP_RADIUS = 21.0
CAP_THICKNESS = 6.0  # as the lid's
CAP_RESOLUTION = 96  # as the pipe's $fn


def _cap_mesh() -> o3d.geometry.TriangleMesh:
    """A disk standing on the pipe's +Y end face, its axis along Y."""
    cap = o3d.geometry.TriangleMesh.create_cylinder(
        CAP_RADIUS, CAP_THICKNESS, resolution=CAP_RESOLUTION, split=4
    )
    cap.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0)), center=(0, 0, 0)
    )
    cap.translate((CAP_CENTRE_X, CAP_FACE_Y + CAP_THICKNESS / 2, 0.0))
    cap.compute_triangle_normals()
    return cap


def main() -> None:
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    shutil.copytree(BENCHMARK / "stl", FOLDER / "stl")
    (FOLDER / "xml").mkdir()
    cap_path = FOLDER / "stl" / f"{CAP}.stl"
    o3d.io.write_triangle_mesh(str(cap_path), _cap_mesh())

    metadata = json.loads((BENCHMARK / "metadata.json").read_text(encoding="utf-8"))
    params = metadata["parameters"]
    assembly = Assembly.load_from_xml(str(BENCHMARK / "xml" / XML_NAME))
    assembly.contact_tollerance = params["contact_tollerance"]
    assembly.dot_product_threshold = params["dot_product_threshold"]
    for part in (
        assembly.parts.values()
    ):  # the copies in this folder, not the benchmark's meshes
        part.mesh_path = str(FOLDER / "stl" / Path(part.mesh_path).name)
    cap = Part(CAP, str(cap_path), density=8000.0)
    assembly.add_part(cap)
    # the pipe comes before the cap in the part order, as in a whole-product detection
    assembly.check_pair_for_contact_new(
        (assembly.parts[PIPE], assembly.parts[CAP]),
        contact_tollerance=assembly.contact_tollerance,
        dot_product_threshold=assembly.dot_product_threshold,
    )
    assembly.name = NAME
    assembly.save_as_xml(str(FOLDER / "xml" / XML_NAME))
    (FOLDER / "product_setup.json").write_text(
        json.dumps(SETUP, indent=2) + "\n", encoding="utf-8"
    )
    for connection in assembly.connections:
        print(connection)


if __name__ == "__main__":
    main()
