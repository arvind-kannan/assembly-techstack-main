"""Generate the tiny test product (#239) in `tests/test_products/tiny_test_product/`.

Four box parts on a base plate: the base part, the target part standing free on it, a cover held down by
a screw whose head overlaps the cover's top face. Contact detection runs here once; the explorer tests
load the committed contact data and never run it. Run it from the repository root after changing the
shapes, and commit the result:

    python tools/make_tiny_test_product.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diss_assembly_dataclasses.assembly import Assembly  # noqa: E402
from diss_assembly_dataclasses.connection import Screw_Connection  # noqa: E402
from diss_assembly_dataclasses.part import Part  # noqa: E402

import disassembly_pomdp  # noqa: E402,F401  (sets up the import paths)

NAME = "tiny_test_product"
FOLDER = ROOT / "tests" / "test_products" / NAME
SCREW_HEAD = "H5"  # a screwdriver of the default cell

# Boxes as (min corner, max corner) in mm; z is up.
SHAPES = {
    "01_base": [((0, 0, 0), (40, 30, 5))],
    "02_target": [((5, 7, 5), (20, 22, 20))],
    "03_cover": [((26, 8, 5), (36, 22, 9))],
    # a nail-shaped screw standing beside the cover, its head over the cover's top face
    "04_screw": [((23, 13, 5), (26, 16, 9)), ((23, 11, 9), (31, 18, 12))],
}
MAX_EDGE = 5.0  # mm; contact detection looks for faces near contact points, so a bare box is too coarse
SETUP = {"base_parts": ["01_base"], "target_parts": ["02_target"]}


def _box(lo, hi) -> o3d.geometry.TriangleMesh:
    lo, hi = np.array(lo, float), np.array(hi, float)
    box = o3d.geometry.TriangleMesh.create_box(*(hi - lo))
    box.translate(lo)
    while _longest_edge(box) > MAX_EDGE:
        box = box.subdivide_midpoint(1)
    return box


def _longest_edge(mesh) -> float:
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return max(
        np.linalg.norm(v[t[:, i]] - v[t[:, (i + 1) % 3]], axis=1).max()
        for i in range(3)
    )


def main() -> None:
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    (FOLDER / NAME).mkdir(parents=True)

    parts = []
    for name, boxes in SHAPES.items():
        mesh = _box(*boxes[0])
        for box in boxes[1:]:
            mesh += _box(*box)
        mesh.compute_triangle_normals()
        path = FOLDER / NAME / f"{name}.stl"
        o3d.io.write_triangle_mesh(str(path), mesh)
        parts.append(
            Part(
                name,
                str(path),
                density=8000.0,
                mesh=o3d.io.read_triangle_mesh(str(path)),
            )
        )

    assembly = Assembly(NAME, parts)
    assembly.find_contacts()
    assembly.add_connection(Screw_Connection("04_screw", "01_base", SCREW_HEAD))
    assembly.save_as_xml(str(FOLDER / f"{NAME}.xml"))
    (FOLDER / "product_setup.json").write_text(
        json.dumps(SETUP, indent=2) + "\n", encoding="utf-8"
    )
    for connection in assembly.connections:
        print(connection)


if __name__ == "__main__":
    main()
