# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh", "shapely", "mapbox-earcut"]
# ///
"""
Part meshes of the battery replacement product model (ticket "Model both demo products as assemblies in the
techstack"), as built from README.md. Model coordinates, mm. Writes model/battery_replacement/*.stl:

  tray            frame (40x40 profiles with an 8 mm top slot), posts, stepped base plate, 41 slot nuts
                  (round, bore = screw shank) and the two locating pins: one part, it is fixtured on the cell
  lid             5 mm plate with the stepped outline, 41x D4.8 holes, pin hole and pin slot, plus the five
                  bonded pressure blocks: one part
  module_1..8     inner modules, plain 250x250x150 blocks on the base plate (grip feature and locating
                  corners left out), finely triangulated for the contact search
  lid_screw_NN    the angle grinder's M4x16 mesh, head bearing face on the lid top; NN = Yonghang screw number

Then build the assembly with make_model.py. Run from this folder (needs uv; shapely triangulates the lid):
    uv run make_model_meshes.py
"""

import csv
import math
from pathlib import Path

import numpy as np
import shapely
import trimesh
from make_build_files import (
    HOLE_D,
    OUTLINE,
    PIN_A,
    PIN_B,
    PIN_D,
    PIN_SLOT_LEN,
    PROFILES,
)
from render_frame import (
    BASE_OUTLINE,
    BASE_T,
    LID_T,
    MODULES,
    POSTS,
    PRESSURE_BLOCKS,
    PROFILE,
    TOP_Z,
)

HERE = Path(__file__).parent
OUT = HERE / "model" / "battery_replacement"
SCREW_MESH = (
    HERE.parent
    / "angle_grinder"
    / "model"
    / "angle_grinder"
    / "Schraube_M4x16_540_Schr1.stl"
)
SCREW_HEAD_FACE_Z = (
    22.5  # head bearing face of SCREW_MESH (shank r = 2.0 from z 6.5 to 22.5)
)
SLOT_W, SLOT_DEPTH = 8.0, 12.0
NUT_D, NUT_H = 8.0, 4.0
SHANK_D = 4.0
PIN_H = 10.0


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(
        extents=hi - lo,
        transform=trimesh.transformations.translation_matrix((lo + hi) / 2),
    )


def profile(axis, fixed, start, end):
    """40x40 profile under the lid, with an open top slot along its centreline (U-section of three boxes)."""
    h, s, z_slot = PROFILE / 2, SLOT_W / 2, TOP_Z - SLOT_DEPTH
    bands = [
        (-h, h, TOP_Z - PROFILE, z_slot),
        (-h, -s, z_slot, TOP_Z),
        (s, h, z_slot, TOP_Z),
    ]
    out = []
    for a, b, z0, z1 in bands:
        lo, hi = (
            ([start, fixed + a], [end, fixed + b])
            if axis == "x"
            else ([fixed + a, start], [fixed + b, end])
        )
        out.append(box([*lo, z0], [*hi, z1]))
    return out


def screw_holes():
    with open(HERE / "nut_positions.csv") as f:
        return sorted(
            (r["item"].split("_", 1)[1], float(r["x_model"]), float(r["y_model"]))
            for r in csv.DictReader(f)
            if not r["item"].startswith("pin")
        )


def lid_outline():
    pts = []
    for seg in OUTLINE:
        if seg[0] == "L":
            pts += [seg[1], seg[2]]
            continue
        c, p, q = seg[1:]
        a0, a1 = (math.atan2(v[1] - c[1], v[0] - c[0]) for v in (p, q))
        a1 = (
            a0 + (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        )  # shortest way round: the 90 deg fillet
        pts += [
            (c[0] + 20 * math.cos(a), c[1] + 20 * math.sin(a))
            for a in np.linspace(a0, a1, 9)
        ]
    return shapely.Polygon(pts)


def lid(holes):
    cutouts = [shapely.Point(x, y).buffer(HOLE_D / 2, 16) for _, x, y in holes]
    cutouts.append(shapely.Point(PIN_A).buffer(PIN_D / 2, 16))
    u = np.subtract(PIN_A, PIN_B) / math.dist(
        PIN_A, PIN_B
    )  # slot runs along the pin-to-pin line
    half = (PIN_SLOT_LEN - PIN_D) / 2
    cutouts.append(
        shapely.LineString([PIN_B - u * half, PIN_B + u * half]).buffer(PIN_D / 2, 8)
    )
    plate = trimesh.creation.extrude_polygon(
        lid_outline().difference(shapely.union_all(cutouts)), LID_T
    )
    plate.apply_translation((0, 0, TOP_Z))
    z1 = TOP_Z + LID_T
    blocks = [box([xa, ya, z1], [xb, yb, z1 + 5]) for xa, xb, ya, yb in PRESSURE_BLOCKS]
    return trimesh.util.concatenate([plate, *blocks])


def tray(holes):
    parts = [m for p in PROFILES.values() for m in profile(*p)]
    h = PROFILE / 2
    parts += [
        box([x - h, y - h, BASE_T], [x + h, y + h, TOP_Z - PROFILE]) for x, y in POSTS
    ]
    parts += [
        box([*lo, 0], [*hi, BASE_T]).subdivide_to_size(10.0) for lo, hi in BASE_OUTLINE
    ]  # modules stand on it
    z_nut = TOP_Z - SLOT_DEPTH
    for _, x, y in holes:
        nut = trimesh.creation.annulus(
            r_min=SHANK_D / 2, r_max=NUT_D / 2, height=NUT_H, sections=16
        )
        parts.append(nut.apply_translation((x, y, z_nut + NUT_H / 2)))
    for x, y in (PIN_A, PIN_B):
        pin = trimesh.creation.cylinder(radius=PIN_D / 2, height=PIN_H, sections=16)
        parts.append(pin.apply_translation((x, y, TOP_Z + PIN_H / 2)))
    return trimesh.util.concatenate(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    holes = screw_holes()
    assert len(holes) == 41, len(holes)
    meshes = {"tray": tray(holes), "lid": lid(holes)}
    for i, (x, y) in enumerate(MODULES, 1):
        # fine triangles here and on the base plate: the core's contact search only finds faces near the FCL
        # contact points, which misses large faces
        meshes[f"module_{i}"] = box(
            [x - 125, y - 125, BASE_T], [x + 125, y + 125, BASE_T + 150]
        ).subdivide_to_size(10.0)
    screw = trimesh.load(SCREW_MESH)
    cx, cy = screw.bounds.mean(axis=0)[:2]
    for name, x, y in holes:
        meshes[name] = screw.copy().apply_translation(
            (x - cx, y - cy, TOP_Z + LID_T - SCREW_HEAD_FACE_Z)
        )
    for name, m in meshes.items():
        m.export(OUT / f"{name}.stl")
    print(f"{len(meshes)} meshes in {OUT}")


if __name__ == "__main__":
    main()
