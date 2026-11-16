from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly
from open3d.visualization import rendering

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_XML = (
    ROOT
    / "tests"
    / "internal_cases"
    / "Modell 2-Plannungssoftware"
    / "Stl"
    / "Assembly 1"
    / "Assembly_1.xml"
)

DEFAULT_OUT_DIR = ROOT / "figures" / "assembly_renders" / "assembly_1_highlight_views"


# ---------------------------------------------------------------------
# KIT / wbk style colors
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"

WBK_GREY = "#3B4F58"
WBK_GREEN = "#009682"
WBK_LIGHT_GREY = "#C4D0CF"
WBK_LIGHT_BLUE = "#3C98D9"
WBK_LIGHT_GREEN = "#27FFE2"

TEXT_COLOR = "#222222"
WHITE = "#FFFFFF"

# Render-specific colors.
BASE_GREY = "#4F6670"
DARK_DETAIL = "#263B44"
EDGE_COLOR = "#1F2F36"

# Main semantic colors for this figure set.
DEFAULT_PART_COLOR = KIT_BLUE_GREEN
HIGHLIGHT_COLOR = KIT_YELLOW


# ---------------------------------------------------------------------
# Assembly-specific part names
# ---------------------------------------------------------------------

SCREWS = {
    "Schraube_M4x16_540_Schr1",
    "Schraube_M4x16_540_Schr2",
    "Schraube_M4x16_540_Schr3",
    "Schraube_M4x16_540_Schr4",
}

# Interpreted as the screw-covered head / upper cover component in this XML.
UPPER_COVER = "BG_Tellerrad_V1p"

# Rotor component in this XML.
ROTOR = "BG_Rotor_p"


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


@contextlib.contextmanager
def pushd(path: Path):
    """Temporarily switch the working directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def hex_to_rgb01(color: str) -> tuple[float, float, float]:
    """Convert a hex color string to RGB values in [0, 1]."""
    color = color.lstrip("#")
    return (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )


def hex_to_rgba01(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a hex color string to RGBA values in [0, 1]."""
    r, g, b = hex_to_rgb01(color)
    return r, g, b, float(alpha)


def make_material(
    color: str,
    *,
    roughness: float = 0.62,
    metallic: float = 0.0,
) -> rendering.MaterialRecord:
    """Create a lit Open3D material for triangle meshes."""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = hex_to_rgba01(color)
    mat.base_roughness = roughness
    mat.base_metallic = metallic
    return mat


def make_line_material(color: str) -> rendering.MaterialRecord:
    """Create an unlit Open3D material for mesh edge lines."""
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = hex_to_rgba01(color)
    mat.line_width = 0.75
    return mat


def load_assembly(xml_path: Path) -> Assembly:
    """
    Load an assembly while using the XML directory as working directory.

    Many assembly XML files store mesh and contact-data paths relatively.
    Changing into the XML directory avoids path-resolution problems when the
    script is launched from another folder.
    """
    xml_path = xml_path.resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"Assembly XML not found: {xml_path}")

    with pushd(xml_path.parent):
        return Assembly.load_from_xml(str(xml_path.name))


def color_for_part(
    part_name: str,
    *,
    highlighted_parts: set[str],
    default_color: str = DEFAULT_PART_COLOR,
    highlight_color: str = HIGHLIGHT_COLOR,
) -> str:
    """Assign one highlight color to selected parts and a default color to others."""
    if part_name in highlighted_parts:
        return highlight_color

    return default_color


def collect_meshes(
    assembly: Assembly,
    *,
    hidden_parts: Iterable[str] | None = None,
    highlighted_parts: Iterable[str] | None = None,
    default_color: str = DEFAULT_PART_COLOR,
    highlight_color: str = HIGHLIGHT_COLOR,
) -> list[tuple[str, o3d.geometry.TriangleMesh, str]]:
    """Collect renderable meshes from the assembly."""
    hidden = set(hidden_parts or [])
    highlighted = set(highlighted_parts or [])

    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str]] = []

    for part in assembly.parts.values():
        if part.name in hidden:
            continue

        mesh = o3d.geometry.TriangleMesh(part.mesh)
        if mesh.is_empty():
            print(f"[SKIP] Empty mesh: {part.name}")
            continue

        mesh.compute_vertex_normals()

        color = color_for_part(
            part.name,
            highlighted_parts=highlighted,
            default_color=default_color,
            highlight_color=highlight_color,
        )

        mesh.paint_uniform_color(hex_to_rgb01(color))
        meshes.append((part.name, mesh, color))

    if not meshes:
        raise ValueError("No renderable meshes were found in the assembly.")

    return meshes


def compute_bbox(
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str]],
) -> o3d.geometry.AxisAlignedBoundingBox:
    """Compute a combined axis-aligned bounding box for all meshes."""
    min_bounds = []
    max_bounds = []

    for _, mesh, _ in meshes:
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bounds.append(np.asarray(bbox.get_min_bound(), dtype=float))
        max_bounds.append(np.asarray(bbox.get_max_bound(), dtype=float))

    min_bound = np.min(np.vstack(min_bounds), axis=0)
    max_bound = np.max(np.vstack(max_bounds), axis=0)

    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def choose_up_vector(view_dir: np.ndarray) -> np.ndarray:
    """Choose a stable camera up vector for the given viewing direction."""
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(view_dir, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    return up


def configure_lighting(renderer: rendering.OffscreenRenderer) -> None:
    """
    Configure print-friendly lighting.

    A no-shadow profile keeps the object readable in thesis figures and avoids
    overly dark walls or occluded faces.
    """
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


def render_assembly_view(
    assembly: Assembly,
    out_path: Path,
    *,
    hidden_parts: Iterable[str] | None = None,
    highlighted_parts: Iterable[str] | None = None,
    width: int = 1800,
    height: int = 1400,
    view_dir: tuple[float, float, float] = (-1.15, 1.65, 1.20),
    zoom: float = 1.68,
    show_edges: bool = True,
    default_color: str = DEFAULT_PART_COLOR,
    highlight_color: str = HIGHLIGHT_COLOR,
) -> None:
    """Render one assembly view with optional hidden and highlighted parts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = collect_meshes(
        assembly,
        hidden_parts=hidden_parts,
        highlighted_parts=highlighted_parts,
        default_color=default_color,
        highlight_color=highlight_color,
    )

    bbox = compute_bbox(meshes)

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(hex_to_rgba01(WHITE))
    configure_lighting(renderer)

    for part_name, mesh, color in meshes:
        renderer.scene.add_geometry(
            part_name,
            mesh,
            make_material(color),
        )

        if show_edges:
            edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            edges.paint_uniform_color(hex_to_rgb01(EDGE_COLOR))
            renderer.scene.add_geometry(
                f"{part_name}_edges",
                edges,
                make_line_material(EDGE_COLOR),
            )

    center = np.asarray(bbox.get_center(), dtype=float)
    extent = np.asarray(bbox.get_extent(), dtype=float)
    diag = float(np.linalg.norm(extent))

    if diag <= 1e-9:
        diag = 1.0

    direction = np.asarray(view_dir, dtype=float)
    direction = direction / np.linalg.norm(direction)

    eye = center + direction * diag * zoom
    up = choose_up_vector(direction)

    renderer.setup_camera(
        42.0,
        center,
        eye,
        up,
    )

    image = renderer.render_to_image()

    # Open3D PNG quality uses range [0, 9].
    o3d.io.write_image(str(out_path), image, quality=9)

    if hasattr(renderer, "release_resources"):
        renderer.release_resources()
    else:
        del renderer

    print(f"[SAVED] {out_path}")


def render_highlight_figure_set(
    assembly: Assembly,
    out_dir: Path,
    *,
    width: int,
    height: int,
    zoom: float,
    show_edges: bool,
    view_dir: tuple[float, float, float],
) -> None:
    """Render the two requested Assembly_1 highlight views."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1:
    # Hide the four screws and highlight the upper cover / head-cover component.
    render_assembly_view(
        assembly,
        out_dir / "assembly_1_upper_cover_highlight_screws_hidden.png",
        hidden_parts=SCREWS,
        highlighted_parts={UPPER_COVER},
        width=width,
        height=height,
        zoom=zoom,
        show_edges=show_edges,
        view_dir=view_dir,
    )

    # Figure 2:
    # Hide the four screws and the upper cover, then highlight the rotor.
    render_assembly_view(
        assembly,
        out_dir / "assembly_1_rotor_highlight_screws_and_cover_hidden.png",
        hidden_parts=SCREWS | {UPPER_COVER},
        highlighted_parts={ROTOR},
        width=width,
        height=height,
        zoom=zoom,
        show_edges=show_edges,
        view_dir=view_dir,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render two highlighted Assembly_1 views."
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Path to the assembly XML file.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the rendered images.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1800,
        help="Rendered image width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1400,
        help="Rendered image height in pixels.",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=1.68,
        help="Camera distance multiplier.",
    )

    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="Disable mesh edge overlay.",
    )

    parser.add_argument(
        "--view",
        choices=["default", "flipped"],
        default="default",
        help="Camera view preset.",
    )

    args = parser.parse_args()

    assembly = load_assembly(args.xml)

    print(f"[INFO] Loaded assembly: {args.xml}")
    print(f"[INFO] Number of parts: {len(assembly.parts)}")

    missing_parts = [
        part_name
        for part_name in sorted(SCREWS | {UPPER_COVER, ROTOR})
        if part_name not in assembly.parts
    ]

    if missing_parts:
        print("[WARNING] Some configured part names were not found:")
        for part_name in missing_parts:
            print(f"  - {part_name}")

    if args.view == "default":
        view_dir = (-1.15, 1.65, 1.20)
    else:
        view_dir = (1.15, -1.65, 1.20)

    render_highlight_figure_set(
        assembly,
        args.out_dir,
        width=args.width,
        height=args.height,
        zoom=args.zoom,
        show_edges=not args.no_edges,
        view_dir=view_dir,
    )


if __name__ == "__main__":
    main()
