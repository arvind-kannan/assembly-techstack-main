from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ---------------------------------------------------------------------
# KIT / wbk colors
# ---------------------------------------------------------------------
KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838f"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
KIT_GREY = "#D9DEE2"
DARK = "#222222"
MID_GREY = "#666666"
LIGHT_GREY = "#E5E8EB"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def cuboid_vertices(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0

    x = [cx - sx, cx + sx]
    y = [cy - sy, cy + sy]
    z = [cz - sz, cz + sz]

    return np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ],
        dtype=float,
    )


def cuboid_faces(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> list[list[np.ndarray]]:
    vertices = cuboid_vertices(center, size)
    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    return [[vertices[j] for j in face] for face in face_indices]


def add_cuboid(
    ax,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: str,
    *,
    alpha: float = 0.25,
    edgecolor: str | None = None,
    linewidth: float = 0.7,
) -> None:
    collection = Poly3DCollection(
        cuboid_faces(center, size),
        facecolors=color,
        edgecolors=edgecolor or color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def add_wire_cuboid(
    ax,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: str,
    *,
    alpha: float = 0.8,
    linewidth: float = 1.0,
) -> None:
    vertices = cuboid_vertices(center, size)
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]],
    ]
    ax.add_collection3d(
        Line3DCollection(edges, colors=color, linewidths=linewidth, alpha=alpha)
    )


def setup_3d(ax) -> None:
    ax.set_facecolor(WHITE)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.8, 1.0, 0.65))
    ax.view_init(elev=22, azim=-55)
    ax.set_xlim(-14, 66)
    ax.set_ylim(-27, 27)
    ax.set_zlim(-17, 17)
    ax.set_axis_off()


def arrow3d(
    ax,
    start,
    vector,
    *,
    color: str,
    linewidth: float = 2.2,
    arrow_length_ratio: float = 0.08,
    head_width_ratio: float = 0.38,
) -> None:
    start = np.asarray(start, dtype=float).reshape(3)
    vector = np.asarray(vector, dtype=float).reshape(3)

    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return

    direction = vector / length
    end = start + vector

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        lw=linewidth,
        solid_capstyle="round",
        zorder=100,
    )

    preferred_side = np.array([0.0, 1.0, 0.0], dtype=float)
    side = preferred_side - np.dot(preferred_side, direction) * direction

    if np.linalg.norm(side) < 1e-9:
        preferred_side = np.array([0.0, 0.0, 1.0], dtype=float)
        side = preferred_side - np.dot(preferred_side, direction) * direction

    side = side / (np.linalg.norm(side) + 1e-12)

    head_length = max(1.2, min(length * arrow_length_ratio, 6.0))
    head_width = head_length * head_width_ratio

    p1 = end - head_length * direction + head_width * side
    p2 = end - head_length * direction - head_width * side

    ax.plot(
        [end[0], p1[0]],
        [end[1], p1[1]],
        [end[2], p1[2]],
        color=color,
        lw=linewidth,
        solid_capstyle="round",
        zorder=101,
    )
    ax.plot(
        [end[0], p2[0]],
        [end[1], p2[1]],
        [end[2], p2[2]],
        color=color,
        lw=linewidth,
        solid_capstyle="round",
        zorder=101,
    )


def add_panel_label(ax, label: str, *, is_3d: bool = True) -> None:
    if is_3d:
        ax.text2D(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=DARK,
        )
    else:
        # For 2D plots, place the label slightly outside the plotting area
        # to avoid overlap with data, ticks, or filled regions.
        ax.text(
            -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=DARK,
            va="bottom",
            ha="left",
            clip_on=False,
        )


def add_short_note_3d(ax, text: str) -> None:
    ax.text2D(
        0.03,
        0.08,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color=DARK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GREY,
            "alpha": 0.9,
        },
    )


def add_short_note_2d(ax, text: str) -> None:
    ax.text(
        0.03,
        0.07,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color=DARK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GREY,
            "alpha": 0.9,
        },
    )


# ---------------------------------------------------------------------
# Scene primitives
# ---------------------------------------------------------------------
def draw_housing(ax, *, alpha: float = 0.16, wire_alpha: float = 0.72) -> None:
    """Draw a simplified surrounding part with an aperture at x = 35 mm."""
    x_pos = 35.0
    thickness = 4.0

    blocks = [
        ((x_pos, 0, 12.5), (thickness, 52, 11)),
        ((x_pos, 0, -12.5), (thickness, 52, 11)),
        ((x_pos, 18.5, 0), (thickness, 15, 14)),
        ((x_pos, -18.5, 0), (thickness, 15, 14)),
    ]

    for center, size in blocks:
        add_cuboid(
            ax,
            center,
            size,
            KIT_DEEP_BLUE,
            alpha=alpha,
            edgecolor=KIT_DEEP_BLUE,
            linewidth=0.5,
        )
        add_wire_cuboid(
            ax,
            center,
            size,
            KIT_DEEP_BLUE,
            alpha=wire_alpha,
            linewidth=0.8,
        )

    yy = [-11, 11, 11, -11, -11]
    zz = [-7, -7, 7, 7, -7]
    ax.plot([x_pos] * 5, yy, zz, color=KIT_BLUE_GREEN, lw=2.0)


def draw_target(
    ax, *, x: float = 0.0, alpha: float = 0.78, ghost: bool = False
) -> None:
    face_alpha = 0.20 if ghost else alpha
    edge_alpha = 0.55 if ghost else 0.85

    add_cuboid(
        ax,
        (x, 0, 0),
        (16, 24, 12),
        KIT_YELLOW,
        alpha=face_alpha,
        edgecolor="#8A6D00",
        linewidth=0.6,
    )
    add_cuboid(
        ax,
        (x, 13, 0),
        (12, 7, 8),
        KIT_YELLOW,
        alpha=face_alpha,
        edgecolor="#8A6D00",
        linewidth=0.6,
    )
    add_wire_cuboid(
        ax,
        (x, 0, 0),
        (16, 24, 12),
        "#8A6D00",
        alpha=edge_alpha,
        linewidth=0.9,
    )
    add_wire_cuboid(
        ax,
        (x, 13, 0),
        (12, 7, 8),
        "#8A6D00",
        alpha=edge_alpha,
        linewidth=0.9,
    )


def footprint_polygon_yz() -> np.ndarray:
    """Return the 2D footprint in the plane orthogonal to the pull-out direction.

    Coordinates are [u, v] = [y, z] in this simplified example.
    """
    return np.array(
        [
            [-12, -6],
            [16, -4],
            [16, 4],
            [12, 6],
            [-12, 6],
        ],
        dtype=float,
    )


def point_in_poly(point: np.ndarray, polygon: np.ndarray) -> bool:
    x_val, y_val = np.asarray(point, dtype=float)
    inside = False
    n_points = len(polygon)

    for i in range(n_points):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n_points]

        crosses_horizontal_ray = (y1 > y_val) != (y2 > y_val)
        if crosses_horizontal_ray:
            x_cross = (x2 - x1) * (y_val - y1) / (y2 - y1 + 1e-12) + x1
            if x_val < x_cross:
                inside = not inside

    return inside


def sample_ray_origins(spacing: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    polygon = footprint_polygon_yz()
    min_uv = polygon.min(axis=0)
    max_uv = polygon.max(axis=0)

    us = np.arange(min_uv[0], max_uv[0] + 0.1, spacing)
    vs = np.arange(min_uv[1], max_uv[1] + 0.1, spacing)

    points: list[list[float]] = []
    for u_val in us:
        for v_val in vs:
            if point_in_poly(np.array([u_val, v_val]), polygon):
                points.append([u_val, v_val])

    points.append([0.0, 0.0])
    points.extend(polygon.tolist())

    points_array = np.unique(np.round(np.asarray(points, dtype=float), 6), axis=0)
    return points_array, polygon


def ray_hits_wall(u_val: float, v_val: float) -> bool:
    """Simplified aperture model: rays outside the opening hit the surrounding part."""
    return not (abs(u_val) <= 11.0 and abs(v_val) <= 7.0)


# ---------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------
def draw_panel_a(ax, *, show_legend: bool = True) -> None:
    setup_3d(ax)
    add_panel_label(ax, "(a)")

    draw_housing(ax)
    draw_target(ax, x=0)
    draw_target(ax, x=55, ghost=True)

    arrow3d(
        ax,
        (-2, -10, -9),
        (7, 0, 0),
        color=KIT_BLUE_GREEN,
        linewidth=3.0,
        arrow_length_ratio=0.25,
    )
    arrow3d(
        ax,
        (0, 2, 4),
        (60, 0, 0),
        color=KIT_RED,
        linewidth=2.4,
        arrow_length_ratio=0.08,
    )

    add_short_note_3d(ax, "Local motion can pass,\nwhile the full path is blocked.")

    if show_legend:
        handles = [
            Line2D(
                [0], [0], color=KIT_BLUE_GREEN, lw=3, label="local translation check"
            ),
            Line2D([0], [0], color=KIT_RED, lw=3, label="full pull-out path"),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=KIT_YELLOW,
                edgecolor="#8A6D00",
                alpha=0.7,
                label="target part",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=KIT_DEEP_BLUE,
                edgecolor=KIT_DEEP_BLUE,
                alpha=0.2,
                label="surrounding part",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            frameon=False,
        )


def draw_panel_b(ax, *, show_legend: bool = True) -> None:
    setup_3d(ax)
    add_panel_label(ax, "(b)")

    draw_housing(ax, alpha=0.06, wire_alpha=0.30)
    draw_target(ax, x=0, alpha=0.48)

    points, polygon = sample_ray_origins(spacing=5.0)
    min_uv = polygon.min(axis=0)
    max_uv = polygon.max(axis=0)

    y_grid, z_grid = np.meshgrid(
        np.linspace(min_uv[0] - 3, max_uv[0] + 3, 2),
        np.linspace(min_uv[1] - 3, max_uv[1] + 3, 2),
    )
    x_grid = np.zeros_like(y_grid)

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=KIT_BLUE_GREEN,
        alpha=0.13,
        edgecolor=KIT_BLUE_GREEN,
        linewidth=0.4,
    )

    closed = np.vstack([polygon, polygon[0]])
    ax.plot(
        np.zeros(closed.shape[0]),
        closed[:, 0],
        closed[:, 1],
        color=KIT_RED,
        lw=2.2,
    )
    ax.scatter(
        np.zeros(points.shape[0]),
        points[:, 0],
        points[:, 1],
        color=KIT_DEEP_BLUE,
        s=18,
        depthshade=False,
    )

    arrow3d(
        ax,
        (0, 0, 0),
        (32, 0, 0),
        color=KIT_RED,
        linewidth=2.0,
        arrow_length_ratio=0.10,
    )

    add_short_note_3d(ax, "Project target part onto\nplane normal to direction.")

    if show_legend:
        handles = [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=KIT_BLUE_GREEN,
                edgecolor=KIT_BLUE_GREEN,
                alpha=0.18,
                label="projection plane",
            ),
            Line2D([0], [0], color=KIT_RED, lw=2.5, label="projected footprint"),
            Line2D(
                [0],
                [0],
                color=KIT_DEEP_BLUE,
                marker="o",
                lw=0,
                markersize=5,
                label="ray origins",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            frameon=False,
        )


def draw_panel_c(ax, *, show_legend: bool = True) -> None:
    points, polygon = sample_ray_origins(spacing=5.0)
    closed = np.vstack([polygon, polygon[0]])

    add_panel_label(ax, "(c)", is_3d=False)

    ax.fill(
        closed[:, 0],
        closed[:, 1],
        color=KIT_YELLOW,
        alpha=0.28,
        label="projected footprint",
    )
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=KIT_RED,
        lw=2.2,
        label="convex hull",
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=30,
        color=KIT_DEEP_BLUE,
        label="ray origins",
        zorder=3,
    )
    ax.scatter(
        [0],
        [0],
        s=90,
        color=KIT_BLUE_GREEN,
        marker="x",
        linewidths=2.4,
        label="centerline ray",
        zorder=4,
    )

    for point in polygon:
        ax.add_patch(
            Circle(
                (point[0], point[1]),
                0.35,
                color=KIT_RED,
                alpha=0.9,
                zorder=5,
            )
        )

    ax.set_xlabel("footprint axis $u$ [mm]")
    ax.set_ylabel("footprint axis $v$ [mm]")

    # Add slight padding so the footprint and panel label do not compete for space.
    ax.set_xlim(-13.5, 17.5)
    ax.set_ylim(-7.2, 7.4)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=LIGHT_GREY)

    add_short_note_2d(ax, "Sample ray origins over\nthe projected footprint.")

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
        )


def draw_panel_d(ax, *, show_legend: bool = True) -> None:
    points, _ = sample_ray_origins(spacing=5.0)

    setup_3d(ax)
    add_panel_label(ax, "(d)")

    draw_housing(ax, alpha=0.14, wire_alpha=0.70)
    draw_target(ax, x=0, alpha=0.28)

    clear_segments: list[list[tuple[float, float, float]]] = []
    blocked_segments: list[list[tuple[float, float, float]]] = []
    hit_points = []

    for u_val, v_val in points:
        if ray_hits_wall(u_val, v_val):
            blocked_segments.append([(1.0, u_val, v_val), (35.0, u_val, v_val)])
            hit_points.append([35.0, u_val, v_val])
        else:
            clear_segments.append([(1.0, u_val, v_val), (60.0, u_val, v_val)])

    ax.add_collection3d(
        Line3DCollection(
            clear_segments,
            colors=KIT_BLUE_GREEN,
            linewidths=1.1,
            alpha=0.75,
        )
    )
    ax.add_collection3d(
        Line3DCollection(
            blocked_segments,
            colors=KIT_RED,
            linewidths=1.7,
            alpha=0.85,
        )
    )

    if hit_points:
        hits = np.asarray(hit_points)
        ax.scatter(
            hits[:, 0],
            hits[:, 1],
            hits[:, 2],
            color=KIT_RED,
            s=22,
            depthshade=False,
        )

    add_short_note_3d(ax, "Blocking hits reject\nthe candidate action.")

    if show_legend:
        handles = [
            Line2D([0], [0], color=KIT_BLUE_GREEN, lw=2.0, label="clear ray"),
            Line2D([0], [0], color=KIT_RED, lw=2.5, label="blocking hit"),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=KIT_DEEP_BLUE,
                edgecolor=KIT_DEEP_BLUE,
                alpha=0.18,
                label="obstacle",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=3,
            frameon=False,
        )


# ---------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------
def save_figure(
    fig,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    *,
    tight: bool = True,
) -> list[Path]:
    out_paths = []

    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(
            path,
            bbox_inches="tight" if tight else None,
            facecolor=WHITE,
        )
        out_paths.append(path)

    plt.close(fig)
    return out_paths


# ---------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------
def figure_local_and_path(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    draw_panel_a(ax, show_legend=True)
    return save_figure(fig, out_dir, "01_local_motion_vs_blocked_path", formats)


def figure_projection_plane(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    draw_panel_b(ax, show_legend=True)
    return save_figure(fig, out_dir, "02_projection_plane_and_footprint", formats)


def figure_footprint_sampling(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_panel_c(ax, show_legend=True)
    fig.tight_layout()
    return save_figure(fig, out_dir, "03_2d_footprint_sampling", formats)


def figure_ray_bundle(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    draw_panel_d(ax, show_legend=True)
    return save_figure(fig, out_dir, "04_ray_bundle_blocking_hits", formats)


# ---------------------------------------------------------------------
# Combined thesis figure
# ---------------------------------------------------------------------
def figure_combined(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    """Render the recommended 2x2 thesis figure.

    The detailed explanation should be placed in the LaTeX caption. The figure
    itself keeps only compact labels and small explanatory notes.
    """
    fig = plt.figure(figsize=(11.7, 8.3))

    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_c = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4, projection="3d")

    draw_panel_a(ax_a, show_legend=False)
    draw_panel_b(ax_b, show_legend=False)
    draw_panel_c(ax_c, show_legend=False)
    draw_panel_d(ax_d, show_legend=False)

    # A compact global legend prevents repeated legends in each panel.
    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=KIT_YELLOW,
            edgecolor="#8A6D00",
            alpha=0.7,
            label="target part / projected footprint",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=KIT_DEEP_BLUE,
            edgecolor=KIT_DEEP_BLUE,
            alpha=0.18,
            label="surrounding part / obstacle",
        ),
        Line2D(
            [0],
            [0],
            color=KIT_BLUE_GREEN,
            lw=2.4,
            label="local check / clear ray",
        ),
        Line2D(
            [0],
            [0],
            color=KIT_RED,
            lw=2.4,
            label="full path / blocking hit",
        ),
        Line2D(
            [0],
            [0],
            color=KIT_DEEP_BLUE,
            marker="o",
            lw=0,
            markersize=5,
            label="ray origins",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.96,
        bottom=0.11,
        wspace=0.06,
        hspace=0.18,
    )

    return save_figure(
        fig,
        out_dir,
        "secondary_geometric_feasibility_check",
        formats,
        tight=True,
    )


def figure_pipeline(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    ax.axis("off")

    boxes = [
        ("candidate\noperation", KIT_DEEP_BLUE),
        ("local\ntranslation", KIT_BLUE_GREEN),
        ("projection\nplane", KIT_BLUE_GREEN),
        ("footprint\nsampling", KIT_YELLOW),
        ("parallel\nray bundle", KIT_YELLOW),
        ("accept /\nreject", KIT_RED),
    ]

    left = 0.04
    width = 0.13
    gap = 0.028
    y_pos = 0.44
    height = 0.34

    for i, (text, color) in enumerate(boxes):
        x_pos = left + i * (width + gap)

        ax.add_patch(
            Rectangle(
                (x_pos, y_pos),
                width,
                height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                alpha=0.13,
                linewidth=1.6,
            )
        )
        ax.text(
            x_pos + width / 2,
            y_pos + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color=DARK,
            transform=ax.transAxes,
        )

        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_pos + width + gap * 0.75, y_pos + height / 2),
                xytext=(x_pos + width + gap * 0.20, y_pos + height / 2),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": MID_GREY, "lw": 1.4},
            )

    ax.text(
        0.04,
        0.21,
        "Secondary geometric feasibility check",
        fontsize=11,
        fontweight="bold",
        color=DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.08,
        "Rejects candidate manipulation actions whose local motion is valid but whose full extraction corridor is blocked.",
        fontsize=9,
        color=MID_GREY,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return save_figure(fig, out_dir, "05_secondary_feasibility_pipeline", formats)


def figure_contact_sheet(out_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    pngs = [
        out_dir / "01_local_motion_vs_blocked_path.png",
        out_dir / "02_projection_plane_and_footprint.png",
        out_dir / "03_2d_footprint_sampling.png",
        out_dir / "04_ray_bundle_blocking_hits.png",
    ]
    pngs = [path for path in pngs if path.exists()]

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))

    for axis, path in zip(axes.flat, pngs):
        image = plt.imread(path)
        axis.imshow(image)
        axis.axis("off")

    for axis in axes.flat[len(pngs) :]:
        axis.axis("off")

    fig.tight_layout()
    return save_figure(
        fig,
        out_dir,
        "secondary_geometric_feasibility_contact_sheet",
        formats,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render KIT-style explanatory figures for the secondary geometric "
            "feasibility check based on projected-footprint ray bundles."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/data/secondary_feasibility_explainer_clean"),
        help="Output directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip archive with all generated figures and this script.",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only render the combined thesis figure and optional pipeline.",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Do not render the compact pipeline diagram.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Do not render the PNG contact sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = tuple(args.formats)
    generated: list[Path] = []

    # Recommended thesis figure.
    generated.extend(figure_combined(out_dir, formats))

    # Optional individual panels for debugging, slides, or appendix use.
    if not args.no_individual:
        generated.extend(figure_local_and_path(out_dir, formats))
        generated.extend(figure_projection_plane(out_dir, formats))
        generated.extend(figure_footprint_sampling(out_dir, formats))
        generated.extend(figure_ray_bundle(out_dir, formats))

    # Optional small workflow diagram.
    if not args.no_pipeline:
        generated.extend(figure_pipeline(out_dir, formats))

    # PNG-only contact sheet for quick inspection.
    if "png" in formats and not args.no_contact_sheet and not args.no_individual:
        generated.extend(figure_contact_sheet(out_dir, ("png",)))

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(generated):
                archive.write(path, arcname=path.name)

            script_path = Path(__file__) if "__file__" in globals() else None
            if script_path and script_path.exists():
                archive.write(script_path, arcname=script_path.name)

        print(zip_path)

    for path in sorted(generated):
        print(path)


if __name__ == "__main__":
    main()
