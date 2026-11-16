"""
Simple rendering of the battery replacement product as built from README.md (frame, posts, base plate,
slot nuts, pins, lid, screws, pressure blocks, inner modules). Model coordinates, mm.

Writes battery_replacement.glb + battery_replacement_exploded.glb (rotate them in Windows 3D Viewer / any
glTF viewer) and three PNGs: render_frame.png (frame, posts, base, nuts, pins only), render_exploded.png
(lid lifted) and render_assembled.png. The lid is drawn without its corner radii.

Run from this folder with a venv that has trimesh and pyrender:
    python render_frame.py
"""

import csv
from pathlib import Path

import numpy as np
import trimesh
from make_build_files import PIN_A, PIN_B, PROFILES

HERE = Path(__file__).parent
TOP_Z = 178.0  # top of the top ring = lid underside
PROFILE = 40.0
BASE_T = 10.0
LID_T = 5.0
POSTS = [
    (-681, -417),
    (-681, 417),
    (-101, -417),
    (-101, 417),
    (479, -417),
    (479, 417),
    (681, -267),
    (681, 253),
]
BASE_OUTLINE = [
    ((-701, -437), (499, 437)),
    ((499, -287), (701, 273)),
]  # stepped, like the frame
MODULES = [(x, y) for x in (-512, -238, 36, 310) for y in (-174, 174)]
PRESSURE_BLOCKS = [
    (-633, -613, -39.1, 42.9),
    (-398, -362, -37.1, 42.9),
    (-140, -104, -37.1, 42.9),
    (118, 154, -37.1, 42.9),
    (366, 386, -39.1, 42.9),
]
COLORS = {
    "profile": [190, 195, 200, 255],
    "post": [150, 155, 160, 255],
    "base": [200, 170, 120, 255],
    "nut": [230, 140, 30, 255],
    "pin": [200, 30, 30, 255],
    "lid": [70, 90, 110, 210],
    "screw": [40, 40, 40, 255],
    "block": [240, 240, 240, 255],
    "module": [90, 150, 210, 255],
}


def box(lo, hi, color):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(
        extents=hi - lo,
        transform=trimesh.transformations.translation_matrix((lo + hi) / 2),
    )
    m.visual.face_colors = color
    return m


def cyl(x, y, z0, z1, d, color):
    m = trimesh.creation.cylinder(radius=d / 2, height=z1 - z0, sections=24)
    m.apply_translation((x, y, (z0 + z1) / 2))
    m.visual.face_colors = color
    return m


def parts(exploded=False):
    lift = 300.0 if exploded else 0.0
    h = PROFILE / 2
    out = {}
    for name, (axis, fixed, start, end) in PROFILES.items():
        lo, hi = (
            ([start, fixed - h], [end, fixed + h])
            if axis == "x"
            else ([fixed - h, start], [fixed + h, end])
        )
        out[f"profile_{name}"] = box(
            [*lo, TOP_Z - PROFILE], [*hi, TOP_Z], COLORS["profile"]
        )
    for i, (x, y) in enumerate(POSTS):
        out[f"post_{i}"] = box(
            [x - h, y - h, BASE_T], [x + h, y + h, TOP_Z - PROFILE], COLORS["post"]
        )
    for i, (lo, hi) in enumerate(BASE_OUTLINE):
        out[f"base_plate_{i}"] = box([*lo, 0], [*hi, BASE_T], COLORS["base"])
    with open(HERE / "nut_positions.csv") as f:
        for r in csv.DictReader(f):
            x, y = float(r["x_model"]), float(r["y_model"])
            out[f"nut_{r['item']}"] = box(
                [x - 8, y - 5, TOP_Z - 8], [x + 8, y + 5, TOP_Z - 4], COLORS["nut"]
            )
            if not r["item"].startswith("pin"):
                out[f"screw_{r['item']}"] = cyl(
                    x,
                    y,
                    TOP_Z + LID_T + lift,
                    TOP_Z + LID_T + 4 + lift,
                    7,
                    COLORS["screw"],
                )
    for name, (x, y) in (("pin_A", PIN_A), ("pin_B", PIN_B)):
        out[name] = cyl(x, y, TOP_Z, TOP_Z + 10, 8, COLORS["pin"])
    z0, z1 = (
        TOP_Z + lift,
        TOP_Z + LID_T + lift,
    )  # lid without corner radii: main bay + stepped end
    out["lid_main"] = box([-691, -427, z0], [489, 427, z1], COLORS["lid"])
    out["lid_step"] = box([489, -277, z0], [691, 263, z1], COLORS["lid"])
    for i, (xa, xb, ya, yb) in enumerate(PRESSURE_BLOCKS):
        out[f"pressure_block_{i + 1}"] = box(
            [xa, ya, z1], [xb, yb, z1 + 5], COLORS["block"]
        )
    for i, (x, y) in enumerate(MODULES):
        out[f"module_{i + 1}"] = box(
            [x - 125, y - 125, BASE_T],
            [x + 125, y + 125, BASE_T + 150],
            COLORS["module"],
        )
    return out


def frame_only():
    keep = ("profile_", "post_", "base_plate", "nut_", "pin_")
    return {k: m for k, m in parts().items() if k.startswith(keep)}


def render(meshes, path, eye):
    import pyrender

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255], ambient_light=[0.45, 0.45, 0.45]
    )
    for m in meshes.values():
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    target = np.array([0.0, 0.0, 90.0])
    eye = np.asarray(eye, float)
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, eye
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(35)), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.5), pose=pose)
    r = pyrender.OffscreenRenderer(1600, 1000)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    import matplotlib.pyplot as plt

    plt.imsave(path, color)


def main():
    assembled, exploded = parts(), parts(exploded=True)
    trimesh.Scene(assembled).export(HERE / "battery_replacement.glb")
    trimesh.Scene(exploded).export(HERE / "battery_replacement_exploded.glb")
    render(frame_only(), HERE / "render_frame.png", eye=[-1300, -1500, 1100])
    render(exploded, HERE / "render_exploded.png", eye=[-1500, -2000, 1500])
    render(assembled, HERE / "render_assembled.png", eye=[-1500, -1900, 1500])
    print(
        "written:",
        *sorted(p.name for p in HERE.glob("battery_replacement*.glb")),
        "render_*.png",
    )


if __name__ == "__main__":
    main()
