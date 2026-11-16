from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
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

DEFAULT_XML_DIR = (
    ROOT
    / "tests"
    / "internal_cases"
    / "visibility_ring_case"
    / "benchmark_ring_case_assembly"
    / "xml"
)

DEFAULT_OUT_DIR = ROOT / "figures" / "visibility_ring_renders"


# ---------------------------------------------------------------------
# KIT / wbk colors
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"

WBK_GREEN = "#009682"
WBK_GREY = "#3B4F58"
WBK_LIGHT_GREY = "#C4D0CF"
WBK_LIGHT_BLUE = "#3C98D9"

TEXT_COLOR = "#222222"
WHITE = "#FFFFFF"

# Render-specific colors.
HOUSING_COLOR = "#4F6670"
COVER_COLOR = KIT_BLUE_GREEN
TARGET_RING_COLOR = KIT_RED
ROD_COLOR = WBK_LIGHT_GREY
OPENING_EDGE_COLOR = KIT_YELLOW
EDGE_COLOR = "#1F2F36"
RAY_COLOR = KIT_DEEP_BLUE
BLOCKED_RAY_COLOR = WBK_GREY
HIDDEN_TARGET_COLOR = WBK_LIGHT_GREY

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
# Case metadata
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RingCaseParams:
    """Parameters shown in the annotation boxes."""

    hole_radius: float
    ring_inner_radius: float = 4.0
    ring_outer_radius: float = 8.0
    ring_thickness: float = 3.0
    ring_depth: float = 27.0
    unit: str = "mm"


@dataclass(frozen=True)
class VisibilityRenderCase:
    """One rendered panel in a visibility-ring figure."""

    title: str
    filename: str
    params: RingCaseParams
    view_angle_deg: float
    visible_target: bool
    blocked_ray: bool
    xml_tokens: tuple[str, ...] = ()
    transparent_context: bool = True


# ---------------------------------------------------------------------
# Figure case definitions
# ---------------------------------------------------------------------

TOPVIEW_TRANSITION_CASES = [
    VisibilityRenderCase(
        title="Below threshold",
        filename="visibility_topview_below_threshold.png",
        params=RingCaseParams(hole_radius=3.0),
        view_angle_deg=0.0,
        visible_target=False,
        blocked_ray=True,
        xml_tokens=("r3.00",),
    ),
    VisibilityRenderCase(
        title="Visibility onset",
        filename="visibility_topview_at_threshold.png",
        params=RingCaseParams(hole_radius=4.0),
        view_angle_deg=0.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r4.00",),
    ),
    VisibilityRenderCase(
        title="Above threshold",
        filename="visibility_topview_above_threshold.png",
        params=RingCaseParams(hole_radius=5.0),
        view_angle_deg=0.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r5.00",),
    ),
]

OBLIQUE_TRANSITION_CASES = [
    VisibilityRenderCase(
        title="Top-down view",
        filename="visibility_angle_0deg.png",
        params=RingCaseParams(hole_radius=8.0),
        view_angle_deg=0.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r8.00",),
    ),
    VisibilityRenderCase(
        title="Moderately oblique view",
        filename="visibility_angle_20deg.png",
        params=RingCaseParams(hole_radius=8.0),
        view_angle_deg=20.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r8.00",),
    ),
    VisibilityRenderCase(
        title="Occlusion-limited view",
        filename="visibility_angle_35deg.png",
        params=RingCaseParams(hole_radius=8.0),
        view_angle_deg=35.0,
        visible_target=False,
        blocked_ray=True,
        xml_tokens=("r8.00",),
    ),
]

# Adjust xml_tokens if your generated XML files use different names.
# Recommended visual logic:
#   fixed theta = 30 deg, fixed opening radius = 10 mm,
#   fixed ring geometry, varied target depth.
DEPTH_TRANSITION_CASES = [
    VisibilityRenderCase(
        title="Shallow target",
        filename="visibility_depth_shallow.png",
        params=RingCaseParams(
            hole_radius=10.0,
            ring_inner_radius=4.0,
            ring_outer_radius=8.0,
            ring_thickness=3.0,
            ring_depth=17.5,
        ),
        view_angle_deg=30.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r10.00",),
    ),
    VisibilityRenderCase(
        title="Depth threshold",
        filename="visibility_depth_threshold.png",
        params=RingCaseParams(
            hole_radius=10.0,
            ring_inner_radius=4.0,
            ring_outer_radius=8.0,
            ring_thickness=3.0,
            ring_depth=27.0,
        ),
        view_angle_deg=30.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r10.00",),
    ),
    VisibilityRenderCase(
        title="Deep target",
        filename="visibility_depth_deep.png",
        params=RingCaseParams(
            hole_radius=10.0,
            ring_inner_radius=4.0,
            ring_outer_radius=8.0,
            ring_thickness=3.0,
            ring_depth=37.0,
        ),
        view_angle_deg=30.0,
        visible_target=False,
        blocked_ray=True,
        xml_tokens=("r10.00",),
    ),
]

# Adjust xml_tokens if your generated XML files use different names.
# Recommended visual logic:
#   fixed theta = 30 deg, fixed opening radius = 12 mm,
#   fixed inner radius and depth, varied outer radius.
OUTER_RADIUS_TRANSITION_CASES = [
    VisibilityRenderCase(
        title="Small target footprint",
        filename="visibility_outer_radius_small.png",
        params=RingCaseParams(
            hole_radius=12.0,
            ring_inner_radius=4.0,
            ring_outer_radius=5.0,
            ring_thickness=3.0,
            ring_depth=27.0,
        ),
        view_angle_deg=30.0,
        visible_target=False,
        blocked_ray=True,
        xml_tokens=("r10.00",),
    ),
    VisibilityRenderCase(
        title="Footprint threshold",
        filename="visibility_outer_radius_threshold.png",
        params=RingCaseParams(
            hole_radius=12.0,
            ring_inner_radius=4.0,
            ring_outer_radius=7.0,
            ring_thickness=3.0,
            ring_depth=27.0,
        ),
        view_angle_deg=30.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r10.00",),
    ),
    VisibilityRenderCase(
        title="Large target footprint",
        filename="visibility_outer_radius_large.png",
        params=RingCaseParams(
            hole_radius=12.0,
            ring_inner_radius=4.0,
            ring_outer_radius=9.0,
            ring_thickness=3.0,
            ring_depth=27.0,
        ),
        view_angle_deg=30.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r10.00",),
    ),
]


# ---------------------------------------------------------------------
# Utilities
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


def hex_to_rgba01(
    color: str,
    alpha: float = 1.0,
) -> tuple[float, float, float, float]:
    red, green, blue = hex_to_rgb01(color)
    return red, green, blue, float(alpha)


def make_material(
    color: str,
    *,
    alpha: float = 1.0,
    roughness: float = 0.62,
    metallic: float = 0.0,
) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency" if alpha < 1.0 else "defaultLit"
    mat.base_color = hex_to_rgba01(color, alpha=alpha)
    mat.base_roughness = roughness
    mat.base_metallic = metallic
    return mat


def make_line_material(
    color: str,
    *,
    alpha: float = 1.0,
    line_width: float = 1.0,
) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = hex_to_rgba01(color, alpha=alpha)
    mat.line_width = line_width
    return mat


def load_assembly(xml_path: Path) -> Assembly:
    xml_path = xml_path.resolve()
    with pushd(xml_path.parent):
        return Assembly.load_from_xml(str(xml_path.name))


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


def choose_up_vector(view_dir: np.ndarray) -> np.ndarray:
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(view_dir, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    return up


def part_color_for_ring_case(
    part_name: str,
    *,
    visible_target: bool,
) -> str:
    if part_name == "ring":
        return TARGET_RING_COLOR if visible_target else HIDDEN_TARGET_COLOR

    if part_name == "top_cap_with_hole":
        return COVER_COLOR

    if part_name == "outer_shell":
        return HOUSING_COLOR

    if part_name == "bottom_cap_with_rod":
        return ROD_COLOR

    return WBK_LIGHT_GREY


def ring_alpha_for_part(
    part_name: str,
    *,
    transparent_context: bool,
) -> float:
    if not transparent_context:
        return 1.0

    if part_name in {"outer_shell", "top_cap_with_hole", "bottom_cap_with_rod"}:
        return 0.24

    if part_name == "ring":
        return 1.0

    return 0.45


def collect_ring_meshes(
    assembly: Assembly,
    *,
    visible_target: bool,
    transparent_context: bool,
    hidden_parts: Iterable[str] | None = None,
) -> list[tuple[str, o3d.geometry.TriangleMesh, str, float]]:
    hidden = set(hidden_parts or [])
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str, float]] = []

    for part in assembly.parts.values():
        if part.name in hidden:
            continue

        mesh = o3d.geometry.TriangleMesh(part.mesh)
        if mesh.is_empty():
            continue

        mesh.compute_vertex_normals()

        color = part_color_for_ring_case(
            part.name,
            visible_target=visible_target,
        )
        alpha = ring_alpha_for_part(
            part.name,
            transparent_context=transparent_context,
        )

        mesh.paint_uniform_color(hex_to_rgb01(color))
        meshes.append((part.name, mesh, color, alpha))

    return meshes


def compute_bbox(
    meshes: list[tuple[str, o3d.geometry.TriangleMesh, str, float]],
) -> o3d.geometry.AxisAlignedBoundingBox:
    if not meshes:
        raise ValueError("No meshes available for rendering.")

    min_bounds = []
    max_bounds = []

    for _, mesh, _, _ in meshes:
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bounds.append(np.asarray(bbox.get_min_bound(), dtype=float))
        max_bounds.append(np.asarray(bbox.get_max_bound(), dtype=float))

    min_bound = np.min(np.vstack(min_bounds), axis=0)
    max_bound = np.max(np.vstack(max_bounds), axis=0)

    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def find_part_mesh(
    assembly: Assembly,
    part_name: str,
) -> o3d.geometry.TriangleMesh | None:
    for part in assembly.parts.values():
        if part.name == part_name:
            mesh = o3d.geometry.TriangleMesh(part.mesh)
            if mesh.is_empty():
                return None
            return mesh
    return None


# ---------------------------------------------------------------------
# Geometry overlays
# ---------------------------------------------------------------------


def add_line(
    renderer: rendering.OffscreenRenderer,
    name: str,
    points: np.ndarray,
    color: str,
    *,
    alpha: float = 1.0,
    line_width: float = 2.0,
) -> None:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(
        [[idx, idx + 1] for idx in range(len(points) - 1)]
    )
    line_set.paint_uniform_color(hex_to_rgb01(color))

    renderer.scene.add_geometry(
        name,
        line_set,
        make_line_material(color, alpha=alpha, line_width=line_width),
    )


def add_circle_line(
    renderer: rendering.OffscreenRenderer,
    *,
    name: str,
    center: np.ndarray,
    radius: float,
    z: float,
    color: str,
    alpha: float = 1.0,
    line_width: float = 2.5,
    num_points: int = 160,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=True)

    points = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full_like(angles, z),
        ]
    )

    add_line(
        renderer,
        name,
        points,
        color,
        alpha=alpha,
        line_width=line_width,
    )


def add_opening_highlight(
    renderer: rendering.OffscreenRenderer,
    assembly: Assembly,
    *,
    hole_radius: float,
) -> None:
    top_mesh = find_part_mesh(assembly, "top_cap_with_hole")
    if top_mesh is None:
        return

    bbox = top_mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=float)
    z_top = float(bbox.get_max_bound()[2]) + 0.02

    add_circle_line(
        renderer,
        name="opening_radius_highlight",
        center=center,
        radius=hole_radius,
        z=z_top,
        color=OPENING_EDGE_COLOR,
        alpha=1.0,
        line_width=3.0,
    )


def add_observation_ray(
    renderer: rendering.OffscreenRenderer,
    bbox: o3d.geometry.AxisAlignedBoundingBox,
    *,
    view_angle_deg: float,
    blocked: bool,
) -> None:
    center = np.asarray(bbox.get_center(), dtype=float)
    extent = np.asarray(bbox.get_extent(), dtype=float)
    diag = float(np.linalg.norm(extent))

    theta = np.deg2rad(view_angle_deg)
    direction = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=float)
    direction = direction / np.linalg.norm(direction)

    start = center - direction * diag * 0.68 + np.array([0.0, 0.0, diag * 0.22])
    end = center + direction * diag * 0.22

    color = BLOCKED_RAY_COLOR if blocked else RAY_COLOR

    add_line(
        renderer,
        f"observation_ray_{view_angle_deg:.0f}",
        np.vstack([start, end]),
        color,
        alpha=1.0,
        line_width=3.0,
    )


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def render_ring_assembly(
    assembly: Assembly,
    out_path: Path,
    *,
    params: RingCaseParams,
    visible_target: bool,
    show_ray: bool,
    view_angle_deg: float,
    blocked_ray: bool,
    width: int = 1800,
    height: int = 1400,
    view_dir: tuple[float, float, float] = (1.25, -1.55, 1.20),
    zoom: float = 1.65,
    show_edges: bool = True,
    transparent_context: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = collect_ring_meshes(
        assembly,
        visible_target=visible_target,
        transparent_context=transparent_context,
    )
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
            edge_alpha = 0.18 if alpha < 1.0 else 0.55
            edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            edges.paint_uniform_color(hex_to_rgb01(EDGE_COLOR))
            renderer.scene.add_geometry(
                f"{part_name}_edges",
                edges,
                make_line_material(
                    EDGE_COLOR,
                    alpha=edge_alpha,
                    line_width=0.55,
                ),
            )

    add_opening_highlight(
        renderer,
        assembly,
        hole_radius=params.hole_radius,
    )

    if show_ray:
        add_observation_ray(
            renderer,
            bbox,
            view_angle_deg=view_angle_deg,
            blocked=blocked_ray,
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
    o3d.io.write_image(str(out_path), image, quality=9)

    if hasattr(renderer, "release_resources"):
        renderer.release_resources()
    else:
        del renderer

    print(f"[SAVED] {out_path}")


# ---------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------


def annotate_visibility_image(
    image_path: Path,
    out_path: Path,
    *,
    params: RingCaseParams,
    view_angle_deg: float,
    annotate_hole: bool = True,
    annotate_depth: bool = True,
    annotate_ring: bool = True,
    annotate_view: bool = True,
) -> Path:
    """
    Add compact parameter boxes to a rendered visibility image.

    The labels are intentionally placed in the upper-left corner without
    connector lines. The yellow opening highlight in the render identifies the
    opening visually, while the boxes document the parameter values.
    """
    img = mpimg.imread(image_path)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.imshow(img)
    ax.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor=WBK_GREY,
        linewidth=0.8,
        alpha=0.92,
    )

    entries: list[str] = []

    if annotate_view:
        entries.append(f"viewing direction\n$\\theta$ = {view_angle_deg:.0f}°")

    if annotate_hole:
        entries.append(f"opening\n$r_h$ = {params.hole_radius:.1f} {params.unit}")

    if annotate_depth:
        entries.append(f"target depth\n$d$ = {params.ring_depth:.1f} {params.unit}")

    if annotate_ring:
        entries.append(
            "ring target\n"
            f"$r_i$ = {params.ring_inner_radius:.1f} {params.unit}\n"
            f"$r_o$ = {params.ring_outer_radius:.1f} {params.unit}\n"
            f"$t$ = {params.ring_thickness:.1f} {params.unit}"
        )

    x_box = 0.065
    y_start = 0.90
    y_gap = 0.125

    for idx, text in enumerate(entries):
        ax.text(
            x_box,
            y_start - idx * y_gap,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=box_style,
        )

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path.with_suffix('.png')}")
    print(f"[SAVED] {out_path.with_suffix('.pdf')}")

    return out_path.with_suffix(".png")


def make_panel_figure(
    image_paths: list[Path],
    titles: list[str],
    out_path: Path,
    *,
    figsize: tuple[float, float] = (11.2, 4.0),
) -> None:
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
# XML selection
# ---------------------------------------------------------------------


def list_xml_files(xml_dir: Path) -> list[Path]:
    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"No XML files found in {xml_dir}")
    return files


def pick_xml_by_tokens(
    xml_files: list[Path],
    tokens: Iterable[str],
) -> Path:
    token_list = [token.lower() for token in tokens if token]

    if not token_list:
        return xml_files[0]

    for path in xml_files:
        name = path.stem.lower()
        if all(token in name for token in token_list):
            return path

    available = "\n".join(path.name for path in xml_files)
    raise FileNotFoundError(
        "Could not find an XML file matching all tokens: "
        f"{token_list}\n\nAvailable XML files:\n{available}"
    )


# ---------------------------------------------------------------------
# Figure rendering helpers
# ---------------------------------------------------------------------


def render_case(
    case: VisibilityRenderCase,
    *,
    xml_files: list[Path],
    fig_dir: Path,
    annotation_flags: dict[str, bool] | None = None,
) -> Path:
    xml_path = pick_xml_by_tokens(xml_files, case.xml_tokens)
    assembly = load_assembly(xml_path)

    raw_path = fig_dir / f"raw_{case.filename}"
    annotated_path = fig_dir / case.filename

    render_ring_assembly(
        assembly,
        raw_path,
        params=case.params,
        visible_target=case.visible_target,
        show_ray=True,
        view_angle_deg=case.view_angle_deg,
        blocked_ray=case.blocked_ray,
        transparent_context=case.transparent_context,
    )

    flags = annotation_flags or {
        "view": True,
        "hole": True,
        "depth": True,
        "ring": True,
    }

    return annotate_visibility_image(
        raw_path,
        annotated_path,
        params=case.params,
        view_angle_deg=case.view_angle_deg,
        annotate_view=flags["view"],
        annotate_hole=flags["hole"],
        annotate_depth=flags["depth"],
        annotate_ring=flags["ring"],
    )


def values_vary(values: list[float], *, tol: float = 1e-9) -> bool:
    if not values:
        return False

    first = float(values[0])
    return any(abs(float(value) - first) > tol for value in values[1:])


def annotation_flags_for_group(
    cases: list[VisibilityRenderCase],
) -> dict[str, bool]:
    """
    Decide which annotation boxes should be shown for a panel group.

    Only parameters that vary within the current figure group are annotated.
    This avoids repeating identical labels in every panel. For example:
        - top-view transition: only opening radius varies
        - oblique transition: only viewing angle varies
        - depth transition: only target depth varies
        - outer-radius transition: only ring geometry varies
    """
    view_varies = values_vary([case.view_angle_deg for case in cases])

    hole_varies = values_vary([case.params.hole_radius for case in cases])

    depth_varies = values_vary([case.params.ring_depth for case in cases])

    ring_inner_varies = values_vary([case.params.ring_inner_radius for case in cases])
    ring_outer_varies = values_vary([case.params.ring_outer_radius for case in cases])
    ring_thickness_varies = values_vary([case.params.ring_thickness for case in cases])

    ring_varies = ring_inner_varies or ring_outer_varies or ring_thickness_varies

    flags = {
        "view": view_varies,
        "hole": hole_varies,
        "depth": depth_varies,
        "ring": ring_varies,
    }

    # For single-panel figures, show all metadata.
    if len(cases) == 1:
        return {
            "view": True,
            "hole": True,
            "depth": True,
            "ring": True,
        }

    # Safety fallback: if nothing varies, keep all labels rather than showing none.
    if not any(flags.values()):
        return {
            "view": True,
            "hole": True,
            "depth": True,
            "ring": True,
        }

    return flags


def render_case_group(
    cases: list[VisibilityRenderCase],
    *,
    xml_files: list[Path],
    out_dir: Path,
    group_name: str,
    panel_filename: str,
    figsize: tuple[float, float] = (11.2, 4.0),
) -> None:
    fig_dir = out_dir / "6_3_visibility_ring"
    fig_dir.mkdir(parents=True, exist_ok=True)

    annotation_flags = annotation_flags_for_group(cases)

    image_paths = []
    titles = []

    for case in cases:
        image_path = render_case(
            case,
            xml_files=xml_files,
            fig_dir=fig_dir,
            annotation_flags=annotation_flags,
        )
        image_paths.append(image_path)
        titles.append(case.title)

    make_panel_figure(
        image_paths,
        titles,
        fig_dir / panel_filename,
        figsize=figsize,
    )

    print(f"[DONE] {group_name}")


# ---------------------------------------------------------------------
# Figure sets
# ---------------------------------------------------------------------


def render_visibility_setup(
    xml_files: list[Path],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "6_3_visibility_ring"
    fig_dir.mkdir(parents=True, exist_ok=True)

    case = VisibilityRenderCase(
        title="Parametric visibility benchmark",
        filename="visibility_ring_setup_reference.png",
        params=RingCaseParams(hole_radius=4.0),
        view_angle_deg=0.0,
        visible_target=True,
        blocked_ray=False,
        xml_tokens=("r4.00",),
        transparent_context=True,
    )

    image_path = render_case(
        case,
        xml_files=xml_files,
        fig_dir=fig_dir,
    )

    make_panel_figure(
        [image_path],
        [case.title],
        fig_dir / "visibility_ring_setup_panel",
        figsize=(5.0, 4.0),
    )


def render_topview_transition(
    xml_files: list[Path],
    out_dir: Path,
) -> None:
    render_case_group(
        TOPVIEW_TRANSITION_CASES,
        xml_files=xml_files,
        out_dir=out_dir,
        group_name="top-view transition",
        panel_filename="visibility_topview_transition_panels",
        figsize=(11.2, 4.0),
    )


def render_oblique_transition(
    xml_files: list[Path],
    out_dir: Path,
) -> None:
    render_case_group(
        OBLIQUE_TRANSITION_CASES,
        xml_files=xml_files,
        out_dir=out_dir,
        group_name="oblique transition",
        panel_filename="visibility_oblique_transition_panels",
        figsize=(11.2, 4.0),
    )


def render_depth_transition(
    xml_files: list[Path],
    out_dir: Path,
) -> None:
    render_case_group(
        DEPTH_TRANSITION_CASES,
        xml_files=xml_files,
        out_dir=out_dir,
        group_name="depth transition",
        panel_filename="visibility_depth_transition_panels",
        figsize=(11.2, 4.0),
    )


def render_outer_radius_transition(
    xml_files: list[Path],
    out_dir: Path,
) -> None:
    render_case_group(
        OUTER_RADIUS_TRANSITION_CASES,
        xml_files=xml_files,
        out_dir=out_dir,
        group_name="outer-radius transition",
        panel_filename="visibility_outer_radius_transition_panels",
        figsize=(11.2, 4.0),
    )


def render_all_visibility_figures(
    xml_dir: Path,
    out_dir: Path,
    *,
    only: str,
    list_only: bool,
) -> None:
    xml_files = list_xml_files(xml_dir)

    print("=" * 80)
    print("Available XML files")
    print("=" * 80)
    for path in xml_files:
        print(path.name)

    if list_only:
        return

    if only in {"all", "setup"}:
        render_visibility_setup(xml_files, out_dir)

    if only in {"all", "topview"}:
        render_topview_transition(xml_files, out_dir)

    if only in {"all", "oblique"}:
        render_oblique_transition(xml_files, out_dir)

    if only in {"all", "depth"}:
        render_depth_transition(xml_files, out_dir)

    if only in {"all", "outer"}:
        render_outer_radius_transition(xml_files, out_dir)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render thesis figures for the visibility ring benchmark."
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=DEFAULT_XML_DIR,
        help="Directory containing visibility ring XML files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for rendered figures.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "setup", "topview", "oblique", "depth", "outer"],
        default="all",
        help="Select which figure set to render.",
    )
    parser.add_argument(
        "--list-xml",
        action="store_true",
        help="Only list available XML files and exit.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    render_all_visibility_figures(
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        only=args.only,
        list_only=args.list_xml,
    )


if __name__ == "__main__":
    main()
