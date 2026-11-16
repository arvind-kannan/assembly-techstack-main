from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from diss_assembly_dataclasses.assembly import Assembly

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DEFAULT_XML = (
    ROOT
    / "benchmark_cases"
    / "benchmark_bolt_and_nut_case"
    / "xml"
    / "rotational_bolt_nut_lock_benchmark.xml"
)

DEFAULT_OUT_DIR = ROOT / "figures" / "benchmark_bolt_and_nut_case"

DEFAULT_OUT = DEFAULT_OUT_DIR / "bolt_nut_lock_benchmark_overview.png"


# ---------------------------------------------------------------------
# KIT / wbk colors
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
WBK_GREY = "#3B4F58"
WBK_LIGHT_GREY = "#C4D0CF"

WHITE = "#FFFFFF"
EDGE_COLOR = "#1F2F36"


# ---------------------------------------------------------------------
# Benchmark part names
# ---------------------------------------------------------------------

PART_I_BOLT = "part_i_bolt"
PART_J_ELBOW_PIPE = "part_j_elbow_pipe"
PART_K_NUT = "part_k_nut"
PART_L_COVER_LID = "part_l_cover_lid"


PART_COLORS = {
    PART_I_BOLT: KIT_YELLOW,  # bolt-like locking part
    PART_J_ELBOW_PIPE: KIT_BLUE_GREEN,  # large carrier
    PART_K_NUT: KIT_RED,  # nut-like constraint
    PART_L_COVER_LID: KIT_DEEP_BLUE,  # cover lid
}

PART_ALPHA = {
    PART_I_BOLT: 1.0,
    PART_J_ELBOW_PIPE: 0.42,
    PART_K_NUT: 1.0,
    PART_L_COVER_LID: 0.55,
}


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def hex_to_rgb01(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )


def hex_to_rgba01(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = hex_to_rgb01(color)
    return r, g, b, float(alpha)


def make_material(
    color: str,
    *,
    alpha: float = 1.0,
    roughness: float = 0.62,
    metallic: float = 0.0,
) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency" if alpha < 0.999 else "defaultLit"
    mat.base_color = hex_to_rgba01(color, alpha)
    mat.base_roughness = roughness
    mat.base_metallic = metallic
    return mat


def make_line_material(color: str, width: float = 0.75) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = hex_to_rgba01(color)
    mat.line_width = width
    return mat


def load_assembly(xml_path: Path) -> Assembly:
    xml_path = xml_path.resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"Assembly XML not found: {xml_path}")

    with pushd(xml_path.parent):
        return Assembly.load_from_xml(str(xml_path.name))


def collect_meshes(
    assembly: Assembly,
    *,
    hidden_parts: Iterable[str] | None = None,
) -> list[tuple[str, o3d.geometry.TriangleMesh, str, float]]:
    hidden = set(hidden_parts or [])
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str, float]] = []

    for part in assembly.parts.values():
        if part.name in hidden:
            continue

        mesh = o3d.geometry.TriangleMesh(part.mesh)
        if mesh.is_empty():
            print(f"[SKIP] Empty mesh: {part.name}")
            continue

        mesh.compute_vertex_normals()

        color = PART_COLORS.get(part.name, WBK_GREY)
        alpha = PART_ALPHA.get(part.name, 1.0)

        mesh.paint_uniform_color(hex_to_rgb01(color))
        meshes.append((part.name, mesh, color, alpha))

    if not meshes:
        raise ValueError("No renderable meshes were found in the assembly.")

    return meshes


def compute_bbox(
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str, float]],
) -> o3d.geometry.AxisAlignedBoundingBox:
    min_bounds = []
    max_bounds = []

    for _, mesh, _, _ in meshes:
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bounds.append(np.asarray(bbox.get_min_bound(), dtype=float))
        max_bounds.append(np.asarray(bbox.get_max_bound(), dtype=float))

    min_bound = np.min(np.vstack(min_bounds), axis=0)
    max_bound = np.max(np.vstack(max_bounds), axis=0)

    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def choose_up_vector(view_dir: np.ndarray) -> np.ndarray:
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(view_dir, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    return up


def configure_lighting(renderer: rendering.OffscreenRenderer) -> None:
    renderer.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        (-0.45, -0.55, -1.0),
    )

    scene = renderer.scene.scene

    if hasattr(scene, "set_sun_light"):
        scene.set_sun_light(
            direction=[-0.45, -0.55, -1.0],
            color=[1.0, 1.0, 1.0],
            intensity=85000,
        )

    if hasattr(scene, "enable_sun_light"):
        scene.enable_sun_light(True)


def render_benchmark_view(
    assembly: Assembly,
    out_path: Path,
    *,
    hidden_parts: Iterable[str] | None = None,
    width: int = 2000,
    height: int = 1500,
    view_dir: tuple[float, float, float] = (-1.35, -1.10, 0.88),
    up_override: tuple[float, float, float] | None = None,
    zoom: float = 1.62,
    show_edges: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = collect_meshes(assembly, hidden_parts=hidden_parts)
    bbox = compute_bbox(meshes)

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(hex_to_rgba01(WHITE))
    configure_lighting(renderer)

    for part_name, mesh, color, alpha in meshes:
        renderer.scene.add_geometry(
            part_name,
            mesh,
            make_material(color, alpha=alpha),
        )

        if show_edges:
            edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            edges.paint_uniform_color(hex_to_rgb01(EDGE_COLOR))
            renderer.scene.add_geometry(
                f"{part_name}_edges",
                edges,
                make_line_material(EDGE_COLOR, width=0.75),
            )

    center = np.asarray(bbox.get_center(), dtype=float)
    extent = np.asarray(bbox.get_extent(), dtype=float)
    diag = float(np.linalg.norm(extent))

    if diag <= 1e-9:
        diag = 1.0

    direction = np.asarray(view_dir, dtype=float)
    direction = direction / np.linalg.norm(direction)

    eye = center + direction * diag * zoom
    if up_override is None:
        up = choose_up_vector(direction)
    else:
        up = np.asarray(up_override, dtype=float)
        up = up / np.linalg.norm(up)

    renderer.setup_camera(
        42.0,
        center,
        eye,
        up,
    )

    image = renderer.render_to_image()
    o3d.io.write_image(str(out_path), image, quality=9)

    if hasattr(renderer, "release_resources"):
        renderer.release_resources()
    else:
        del renderer

    print(f"[SAVED] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the bolt-nut locking benchmark case."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Path to the benchmark assembly XML file.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Rendered image width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Rendered image height in pixels.",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=1.62,
        help="Camera distance multiplier.",
    )

    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Disable mesh edge overlay.",
    )

    parser.add_argument(
        "--view",
        choices=[
            "default",
            "side",
            "top",
            "top_tilt30_x",
            "top_tilt30_x_flipped",
            "top_tilt30_y",
            "top_tilt30_y_flipped",
        ],
        default="default",
        help="Camera view preset.",
    )

    args = parser.parse_args()

    assembly = load_assembly(args.xml)

    print(f"[INFO] Loaded assembly: {args.xml}")
    print(f"[INFO] Number of parts: {len(assembly.parts)}")
    print("[INFO] Parts:")
    for part_name in sorted(assembly.parts):
        print(f"  - {part_name}")

    up_override = None

    up_override = None

    if args.view == "default":
        view_dir = (-1.35, -1.10, 0.88)

    elif args.view == "side":
        view_dir = (0.0, -1.0, 0.25)

    elif args.view == "top":
        view_dir = (0.0, 0.0, 1.0)

    elif args.view == "top_tilt30_x":
        # Start from +Z top view and tilt 30 deg around the X axis.
        # This introduces a Y component into the camera direction.
        view_dir = (0.0, -0.5, 0.8660254)

    elif args.view == "top_tilt30_x_flipped":
        # Opposite side of the same X-axis tilt.
        view_dir = (0.0, 0.5, 0.8660254)

    elif args.view == "top_tilt30_y":
        # Start from +Z top view and tilt 30 deg around the Y axis.
        # In this benchmark, the long bolt/nut pipe branch is along +Y,
        # so this keeps the long branch as the tilt axis.
        view_dir = (0.5, 0.0, 0.8660254)

    elif args.view == "top_tilt30_y_flipped":
        # Opposite side of the same Y-axis tilt.
        view_dir = (-0.5, 0.0, 0.8660254)

    else:
        raise ValueError(f"Unknown view preset: {args.view}")

    render_benchmark_view(
        assembly,
        args.out,
        width=args.width,
        height=args.height,
        zoom=args.zoom,
        show_edges=not args.no_edges,
        view_dir=view_dir,
        up_override=up_override,
    )


if __name__ == "__main__":
    main()
