"""Convert one Assemble-Them-All assembly into a planner-ready product folder (#226).

Assemble-Them-All (https://github.com/yunshengtian/Assemble-Them-All, MIT code; the data derives mostly from
the Fusion 360 Gallery Dataset, whose licence applies) stores an assembly as a folder of `.obj` part meshes in
the assembled state, normalized to a box of about 10 units (observed on the rotational set; the upstream loader was not read). This script scales the parts to millimetres, runs
the core's contact detection once, and writes the product folder the techstack loads:

    <out>/<name>/<name>.xml            product model, paths relative to the folder
    <out>/<name>/<name>/<part>.stl     part meshes
    <out>/<name>/<name>_data/*.npy     contact data
    <out>/<name>/product_setup.json    base and target parts
    <out>/<name>/provenance.json       source assembly, scale, base choice

Geometric connections only. Nothing of the dataset is committed: point `--source` at a local copy (the zips
under https://people.csail.mit.edu/yunsheng/Assemble-Them-All/dataset_2403/) and `--out` at the product root,
the 3D files repository per ADR-0011.

    python tools/import_tool/assemble_them_all_to_product.py \\
        --source /data/joint_assembly_rotation/general/06397 --out /data/products
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# isort: off
import disassembly_pomdp  # noqa: E402,F401  (sets up the import paths; must come first)
from diss_assembly_dataclasses.assembly import Assembly  # noqa: E402
from diss_assembly_dataclasses.part import Part  # noqa: E402

# isort: on

# The dataset normalizes every assembly into a box of about 10 units; 10 mm per unit gives a 100 mm product.
MM_PER_UNIT = 10.0
# Contact detection looks for faces near contact points, so coarse faces are subdivided first (see
# tools/make_tiny_test_product.py).
MAX_EDGE_MM = 5.0
DENSITY = 8000.0


def discover_obj_parts(case_dir: Path) -> list[Path]:
    """The .obj files of an assembly folder, in numeric order of their stems (0.obj, 1.obj, ..., 10.obj)."""
    return sorted(
        Path(case_dir).glob("*.obj"),
        key=lambda p: (0, int(p.stem), "") if p.stem.isdigit() else (1, 0, p.stem),
    )


def base_part(meshes: dict[str, o3d.geometry.TriangleMesh]) -> str:
    """The part with the largest bounding box volume: the body the others are assembled onto."""

    def volume(mesh):
        return float(np.prod(mesh.get_axis_aligned_bounding_box().get_extent()))

    return max(meshes, key=lambda name: volume(meshes[name]))


def convert_assembly(
    source: Path,
    out_root: Path,
    name: str | None = None,
    *,
    mm_per_unit: float = MM_PER_UNIT,
    base: str | None = None,
) -> Path:
    """Write the product folder of the assembly in `source` under `out_root`; return its XML path."""
    source, out_root = Path(source), Path(out_root)
    if (source / "translation.json").exists():
        raise NotImplementedError(
            f"{source} holds a translation.json, which this converter does not apply yet; "
            "the parts would land in the wrong pose"
        )
    objs = discover_obj_parts(source)
    if not objs:
        raise FileNotFoundError(f"no .obj parts in {source}")
    name = name or f"ata_{source.parent.name}_{source.name}"
    folder = out_root / name
    mesh_dir = folder / name
    mesh_dir.mkdir(parents=True, exist_ok=True)

    meshes, parts = {}, []
    for obj in objs:
        part_name = f"part_{obj.stem}"
        mesh = o3d.io.read_triangle_mesh(str(obj))
        if len(mesh.triangles) == 0:
            raise ValueError(f"{obj} holds no triangles")
        mesh.scale(mm_per_unit, center=(0, 0, 0))
        mesh = _refine(mesh)
        mesh.compute_triangle_normals()
        path = mesh_dir / f"{part_name}.stl"
        o3d.io.write_triangle_mesh(str(path), mesh)
        meshes[part_name] = mesh
        parts.append(
            Part(
                part_name,
                str(path),
                density=DENSITY,
                mesh=o3d.io.read_triangle_mesh(str(path)),
            )
        )

    base_by_default = base is None
    base = base or base_part(meshes)
    if base not in meshes:
        raise ValueError(f"base part {base!r} is not one of {sorted(meshes)}")

    assembly = Assembly(name, parts)
    assembly.find_contacts()
    xml = folder / f"{name}.xml"
    assembly.save_as_xml(str(xml))
    (folder / "product_setup.json").write_text(
        json.dumps(
            {
                "base_parts": [base],
                "target_parts": [p for p in meshes if p != base],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (folder / "provenance.json").write_text(
        json.dumps(
            {
                "source": "Assemble-Them-All",
                "assembly": f"{source.parent.name}/{source.name}",
                "mm_per_unit": mm_per_unit,
                "base_part": base,
                "base_chosen_by": "largest bounding box"
                if base_by_default
                else "--base",
                "parts": len(parts),
                "connections": len(assembly.connections),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return xml


def _refine(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Subdivide until no edge is longer than MAX_EDGE_MM (at most four passes: dataset meshes are fine)."""
    for _ in range(4):
        if _longest_edge(mesh) <= MAX_EDGE_MM:
            break
        mesh = mesh.subdivide_midpoint(1)
    return mesh


def _longest_edge(mesh: o3d.geometry.TriangleMesh) -> float:
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    return max(
        np.linalg.norm(v[t[:, i]] - v[t[:, (i + 1) % 3]], axis=1).max()
        for i in range(3)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source", type=Path, required=True, help="folder of the .obj parts"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="product root to write the folder under"
    )
    parser.add_argument("--name", help="product folder name (default ata_<group>_<id>)")
    parser.add_argument("--mm-per-unit", type=float, default=MM_PER_UNIT)
    parser.add_argument(
        "--base", help="base part name, e.g. part_0 (default: the largest part)"
    )
    args = parser.parse_args()
    xml = convert_assembly(
        args.source, args.out, args.name, mm_per_unit=args.mm_per_unit, base=args.base
    )
    print(xml)


if __name__ == "__main__":
    main()
