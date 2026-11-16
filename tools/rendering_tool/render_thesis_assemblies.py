from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly
from open3d.visualization import rendering

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "emo_sr_v2" / "EMO_SR_V2.xml"
DEFAULT_OUT_DIR = ROOT / "figures" / "assembly_renders"


# ---------------------------------------------------------------------
# KIT / wbk style colors
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
WBK_GREY = "#3B4F58"
TEXT_COLOR = "#222222"
WHITE = "#FFFFFF"
WBK_GREEN = "#009682"
WBK_LIGHT_GREY = "#C4D0CF"
WBK_LIGHT_BLUE = "#3C98D9"
WBK_LIGHT_GREEN = "#27FFE2"
LIGHT_GREY = WBK_LIGHT_GREY

# Render-specific colors.
# These are intentionally brighter than the previous dark base colors, because
# dark blue/black surfaces become hard to read in PDF and print.
BASE_GREY = "#4F6670"
DARK_DETAIL = "#263B44"
EDGE_COLOR = "#1F2F36"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.titlesize": 11,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------
# Assembly-specific semantic groups
# ---------------------------------------------------------------------

SCREW_PREFIXES = (
    "04_screw_",
    "05_screw_",
    "06_screw_",
    "07_screw_",
)

UPPER_LID = "01_upper_lid"
STATOR = "02_stator"
ROTOR = "03_rotor_short"
LOWER_LID = "08_lower_lid"

BASE_PARTS = {
    STATOR,
    LOWER_LID,
}

GOAL_REMOVED_PARTS = {
    UPPER_LID,
    "04_screw_1",
    "05_screw_2",
    "06_screw_3",
    "07_screw_4",
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


def is_screw(part_name: str) -> bool:
    return part_name.startswith(SCREW_PREFIXES)


def make_material(
    color: str,
    *,
    roughness: float = 0.62,
    metallic: float = 0.0,
) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = hex_to_rgba01(color)
    mat.base_roughness = roughness
    mat.base_metallic = metallic
    return mat


def make_line_material(color: str) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = hex_to_rgba01(color)
    mat.line_width = 0.75
    return mat


def load_assembly(xml_path: Path) -> Assembly:
    """
    Load the assembly while using the XML directory as working directory.

    The EMO XML stores mesh and contact-data paths relatively, so this avoids
    path-resolution problems when the script is launched from another folder.
    """
    xml_path = xml_path.resolve()
    with pushd(xml_path.parent):
        return Assembly.load_from_xml(str(xml_path.name))


def part_color_for_mode(part_name: str, mode: str) -> str:
    """
    Assign colors for thesis rendering modes.

    Modes:
        neutral:
            Clean object overview.
        mdp_targets:
            Highlight the parts relevant to the MDP baseline task.
        eol_pomdp:
            Highlight representative EOL/POMDP-related components.
    """
    if mode == "neutral":
        if is_screw(part_name):
            return DARK_DETAIL
        if part_name in BASE_PARTS:
            return BASE_GREY
        if part_name == UPPER_LID:
            return KIT_BLUE_GREEN
        if part_name == ROTOR:
            return LIGHT_GREY
        return LIGHT_GREY

    if mode == "mdp_targets":
        if is_screw(part_name):
            return KIT_YELLOW
        if part_name == UPPER_LID:
            return WBK_GREEN
        if part_name == ROTOR:
            return KIT_RED
        if part_name in BASE_PARTS:
            return BASE_GREY
        return LIGHT_GREY

    if mode == "eol_pomdp":
        if part_name == UPPER_LID:
            return KIT_BLUE_GREEN
        if is_screw(part_name):
            return KIT_YELLOW
        if part_name == ROTOR:
            return KIT_RED
        if part_name in BASE_PARTS:
            return BASE_GREY
        return LIGHT_GREY

    return LIGHT_GREY


def collect_meshes(
    assembly: Assembly,
    *,
    mode: str,
    hidden_parts: Iterable[str] | None = None,
) -> list[tuple[str, o3d.geometry.TriangleMesh, str]]:
    hidden = set(hidden_parts or [])
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str]] = []

    for part in assembly.parts.values():
        if part.name in hidden:
            continue

        mesh = o3d.geometry.TriangleMesh(part.mesh)
        if mesh.is_empty():
            continue

        mesh.compute_vertex_normals()
        color = part_color_for_mode(part.name, mode)
        mesh.paint_uniform_color(hex_to_rgb01(color))
        meshes.append((part.name, mesh, color))

    return meshes


def compute_bbox(
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str]],
) -> o3d.geometry.AxisAlignedBoundingBox:
    if not meshes:
        raise ValueError("No meshes available for rendering.")

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
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(view_dir, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    return up


def configure_lighting(renderer: rendering.OffscreenRenderer) -> None:
    """
    Configure publication-friendly lighting.

    The light direction is chosen to brighten the front-facing side of the
    assembly and reduce the previous dark-wall effect on the base body.
    """
    # A soft no-shadow profile keeps surface details readable in print.
    renderer.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        (-0.45, -0.55, -1.0),
    )

    # Additional sun light, supported by most recent Open3D versions.
    scene = renderer.scene.scene
    if hasattr(scene, "set_sun_light"):
        scene.set_sun_light(
            direction=[-0.45, -0.55, -1.0],
            color=[1.0, 1.0, 1.0],
            intensity=85000,
        )
    if hasattr(scene, "enable_sun_light"):
        scene.enable_sun_light(True)


def render_assembly(
    assembly: Assembly,
    out_path: Path,
    *,
    mode: str,
    hidden_parts: Iterable[str] | None = None,
    width: int = 1800,
    height: int = 1400,
    view_dir: tuple[float, float, float] = (1.15, -1.65, 1.20),
    zoom: float = 1.68,
    show_edges: bool = True,
) -> None:
    """
    Render one assembly state to a PNG image.

    The function uses Open3D OffscreenRenderer. For headless Linux setups, run
    with OPEN3D_CPU_RENDERING=true if EGL causes problems.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = collect_meshes(
        assembly,
        mode=mode,
        hidden_parts=hidden_parts,
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

    # Open3D PNG quality uses range [0, 9], not [0, 100].
    o3d.io.write_image(str(out_path), image, quality=9)

    # Open3D versions differ: some OffscreenRenderer objects do not expose
    # release_resources(). Deleting the renderer is sufficient here.
    if hasattr(renderer, "release_resources"):
        renderer.release_resources()
    else:
        del renderer

    print(f"[SAVED] {out_path}")


def make_panel_figure(
    image_paths: list[Path],
    titles: list[str],
    out_path: Path,
    *,
    figsize: tuple[float, float] = (11.0, 4.0),
) -> None:
    """
    Combine rendered PNGs into one publication-friendly panel figure.
    """
    if len(image_paths) != len(titles):
        raise ValueError("image_paths and titles must have the same length.")

    fig, axes = plt.subplots(1, len(image_paths), figsize=figsize)

    if len(image_paths) == 1:
        axes = [axes]

    for idx, (ax, path, title) in enumerate(zip(axes, image_paths, titles)):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(f"({chr(ord('a') + idx)}) {title}", pad=6)
        ax.axis("off")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path.with_suffix('.png')}")
    print(f"[SAVED] {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------
# Figure sets
# ---------------------------------------------------------------------


def render_6_1_mdp_baseline_set(
    assembly: Assembly,
    out_dir: Path,
) -> None:
    """
    Render the recommended figure set for Section 6.1.

    The single target figure is the recommended one for the thesis text.
    The panel figure is still generated as an optional overview.
    """
    fig_dir = out_dir / "6_1_mdp_baseline"
    fig_dir.mkdir(parents=True, exist_ok=True)

    initial = fig_dir / "emo_6_1_initial_overview.png"
    targets = fig_dir / "emo_6_1_mdp_targets.png"
    goal = fig_dir / "emo_6_1_target_completed_state.png"

    render_assembly(
        assembly,
        initial,
        mode="neutral",
        view_dir=(1.15, -1.65, 1.20),
    )

    render_assembly(
        assembly,
        targets,
        mode="mdp_targets",
        view_dir=(1.15, -1.65, 1.20),
    )

    render_assembly(
        assembly,
        goal,
        mode="mdp_targets",
        hidden_parts=GOAL_REMOVED_PARTS,
        view_dir=(1.15, -1.65, 1.20),
    )

    make_panel_figure(
        [initial, targets, goal],
        [
            "Initial assembly",
            "MDP action targets",
            "Rotor-accessible state",
        ],
        fig_dir / "emo_6_1_mdp_baseline_panels",
        figsize=(11.2, 4.0),
    )


def render_6_3_pomdp_eol_set(
    assembly: Assembly,
    out_dir: Path,
) -> None:
    """
    Render a compact figure set for Section 6.3.

    This is kept for later use. For now, Section 6.3 will be rendered from the
    OpenSCAD ring cases instead of EMO_SR_V2.
    """
    fig_dir = out_dir / "6_3_pomdp_eol"
    fig_dir.mkdir(parents=True, exist_ok=True)

    nominal = fig_dir / "emo_6_3_nominal_reference.png"
    missing = fig_dir / "emo_6_3_missing_screw_example.png"
    stuck = fig_dir / "emo_6_3_stuck_upper_lid_example.png"

    render_assembly(
        assembly,
        nominal,
        mode="neutral",
        view_dir=(1.15, -1.65, 1.20),
    )

    render_assembly(
        assembly,
        missing,
        mode="mdp_targets",
        hidden_parts={"04_screw_1"},
        view_dir=(1.15, -1.65, 1.20),
    )

    render_assembly(
        assembly,
        stuck,
        mode="eol_pomdp",
        view_dir=(1.15, -1.65, 1.20),
    )

    make_panel_figure(
        [nominal, missing, stuck],
        [
            "Nominal reference",
            "Missing screw example",
            "Stuck upper-lid example",
        ],
        fig_dir / "emo_6_3_pomdp_eol_panels",
        figsize=(11.2, 4.0),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render thesis assembly figures for EMO_SR_V2."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Path to EMO_SR_V2.xml.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for rendered figures.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "6.1", "6.3"],
        default="all",
        help="Select which figure set to render.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    assembly = load_assembly(args.xml)

    if args.only in {"all", "6.1"}:
        render_6_1_mdp_baseline_set(assembly, args.out_dir)


if __name__ == "__main__":
    main()
