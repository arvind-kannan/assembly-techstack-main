from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =============================================================================
# Thesis-ready visibility plotting script
# =============================================================================
# Assumptions:
# - All geometric length values are given in millimetres [mm].
# - Missing threshold_hole_radius values (NaN) indicate that the ring did not
#   become visible within the tested opening-radius range.
# - Figure titles are intentionally omitted. In a thesis/report, the main
#   explanation should be provided in the figure caption.
# =============================================================================


RESULTS = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "internal_cases"
    / "visibility_ring_case"
    / "benchmark_ring_case_results"
)

FIG_DIR = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath(
        "Bau_Yuan/02_Figures/04_Chapter_6/03_Section_6.3/visibility_margin_results"
    )
)


# -----------------------------------------------------------------------------
# CSV paths
# -----------------------------------------------------------------------------

CSV_TOP_FINE = RESULTS / "visibility_results_hole_radii_3_4.csv"
CSV_TOP_COARSE = RESULTS / "visibility_results.csv"

CSV_VIEW_BASELINE = RESULTS / "threshold_vs_view_angle_baseline.csv"
CSV_VIEW_CRITICAL = RESULTS / "threshold_vs_view_angle_critical_region.csv"

CSV_TOP_DEPTH = RESULTS / "threshold_vs_ring_z_top_view_only.csv"
CSV_TOP_GEOMETRY = RESULTS / "threshold_vs_ring_geometry_top_view_only.csv"

CSV_CRITICAL_RING_INNER = RESULTS / "critical_angle_vs_ring_inner_radius.csv"
CSV_FIXED_OUTER = RESULTS / "critical_angle_fixed_outer_vary_inner.csv"
CSV_FIXED_INNER = RESULTS / "critical_angle_fixed_inner_vary_radial_width.csv"
CSV_DEPTH_OBLIQUE = RESULTS / "depth_effect_under_oblique_views.csv"


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838f"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
KIT_GREY = "#777777"

COLORS = [
    KIT_DEEP_BLUE,
    KIT_BLUE_GREEN,
    KIT_YELLOW,
    KIT_RED,
    KIT_GREY,
    "#4C72B0",
    "#55A868",
    "#C44E52",
]

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]

plt.rcParams.update(
    {
        # Good default for half-page thesis figures.
        "figure.figsize": (6.0, 4.0),
        # Keep text readable after insertion into an A4 thesis page.
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Clean report style.
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Export settings.
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] Missing CSV file: {path.name}")
        return None
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Return a copy of df with selected columns converted to numeric values."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_figure(fig: Figure, filename: str) -> None:
    """Save the same figure as PNG, PDF, and SVG.

    PDF/SVG are preferred for vector-based thesis figures. PNG is provided as a
    fallback for Word or presentation workflows.
    """
    png_path = FIG_DIR / f"{filename}.png"
    pdf_path = FIG_DIR / f"{filename}.pdf"
    svg_path = FIG_DIR / f"{filename}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def finite_threshold_df(df: pd.DataFrame) -> pd.DataFrame:
    df = numeric(
        df, ["threshold_hole_radius", "threshold_hole_diameter", "view_angle_deg"]
    )
    return df.dropna(subset=["threshold_hole_radius"])


def infeasible_y_value(df: pd.DataFrame, margin: float = 1.0) -> float:
    """Return an artificial y-level used to display infeasible/NaN thresholds."""
    df = numeric(df, ["threshold_hole_radius"])
    finite = df["threshold_hole_radius"].dropna()
    if finite.empty:
        return 1.0
    return float(finite.max() + margin)


def plot_infeasible_points(
    ax: Axes,
    df: pd.DataFrame,
    x_col: str,
    y_value: float,
    label: str | None = None,
    x_offset: float = 0.0,
    color: str = KIT_RED,
) -> None:
    """Plot NaN threshold rows as x markers at an artificial y-level."""
    df = numeric(df, ["threshold_hole_radius", x_col])
    infeasible = df[df["threshold_hole_radius"].isna()]
    if infeasible.empty:
        return

    ax.scatter(
        infeasible[x_col] + x_offset,
        np.full(len(infeasible), y_value),
        marker="x",
        s=70,
        color=color,
        linewidths=1.8,
        label=label,
        zorder=4,
    )


def add_note_bottom_right(ax: Axes, text: str) -> None:
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
    )


def add_infeasible_axis_margin(ax: Axes, y_infeasible: float) -> None:
    """Ensure artificial infeasible markers remain visible."""
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, max(y_max, y_infeasible + 0.8))


# -----------------------------------------------------------------------------
# Figure 1: Top-view threshold behaviour
# -----------------------------------------------------------------------------


def plot_top_view_threshold() -> None:
    df = read_csv_optional(CSV_TOP_FINE)
    if df is None:
        df = read_csv_required(CSV_TOP_COARSE)

    df = numeric(
        df,
        [
            "hole_radius",
            "hole_diameter",
            "ring_visible",
            "ring_inner_radius",
            "ring_outer_radius",
            "depth_from_opening_plane",
        ],
    )

    df = df.sort_values("hole_radius")

    visible_rows = df[df["ring_visible"] == 1]
    threshold = visible_rows["hole_radius"].min() if not visible_rows.empty else np.nan

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.step(
        df["hole_radius"],
        df["ring_visible"],
        where="post",
        color=KIT_DEEP_BLUE,
        linewidth=2.0,
        label="Visibility state",
    )

    ax.scatter(
        df["hole_radius"],
        df["ring_visible"],
        color=KIT_BLUE_GREEN,
        s=38,
        label="Tested configuration",
        zorder=3,
    )

    if np.isfinite(threshold):
        ax.axvline(
            threshold,
            color=KIT_RED,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold $r_h^*={threshold:.1f}$ mm",
        )

        ax.text(
            threshold,
            0.05,
            f"$r_h^*={threshold:.1f}$ mm",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
            color=KIT_RED,
        )

    ax.set_xlabel("Opening radius $r_h$ [mm]")
    ax.set_ylabel("Ring visible [-]")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_ylim(-0.1, 1.1)
    ax.legend(frameon=True)

    save_figure(fig, "fig_1_top_view_threshold")


# -----------------------------------------------------------------------------
# Figure 2: Threshold vs viewing angle
# -----------------------------------------------------------------------------


def combine_view_angle_data() -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    df_base = read_csv_optional(CSV_VIEW_BASELINE)
    if df_base is not None:
        df_base = df_base.copy()
        df_base["source"] = "baseline"
        dfs.append(df_base)

    df_crit = read_csv_optional(CSV_VIEW_CRITICAL)
    if df_crit is not None:
        df_crit = df_crit.copy()
        df_crit["source"] = "critical sweep"
        dfs.append(df_crit)

    if not dfs:
        raise FileNotFoundError("No view-angle CSV files found.")

    df = pd.concat(dfs, ignore_index=True)
    df = numeric(
        df,
        [
            "view_angle_deg",
            "threshold_hole_radius",
            "threshold_hole_diameter",
            "ring_inner_radius",
            "ring_outer_radius",
            "depth_from_opening_plane",
        ],
    )

    # If the same angle appears twice, keep finite thresholds first.
    # If both are finite, keep the first occurrence, which preserves the baseline
    # values before the critical-region refinement.
    df["_finite"] = df["threshold_hole_radius"].notna().astype(int)
    df = df.sort_values(["view_angle_deg", "_finite"], ascending=[True, False])
    df = df.drop_duplicates(subset=["view_angle_deg"], keep="first")
    df = df.drop(columns=["_finite"])
    df = df.sort_values("view_angle_deg")

    return df


def plot_threshold_vs_view_angle() -> None:
    df = combine_view_angle_data()
    finite = finite_threshold_df(df)

    y_infeasible = infeasible_y_value(df, margin=1.0)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(
        finite["view_angle_deg"],
        finite["threshold_hole_radius"],
        marker="o",
        color=KIT_DEEP_BLUE,
        markersize=6,
        linewidth=2.0,
        label="Finite threshold",
    )

    plot_infeasible_points(
        ax,
        df,
        x_col="view_angle_deg",
        y_value=y_infeasible,
        label="Not visible within tested range",
    )

    add_infeasible_axis_margin(ax, y_infeasible)

    ax.set_xlabel("Viewing angle $\\theta$ [deg]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(frameon=True)

    save_figure(fig, "fig_2_threshold_vs_view_angle")


# -----------------------------------------------------------------------------
# Figure 3: Top-view geometry relation
# -----------------------------------------------------------------------------


def plot_top_view_geometry_relation() -> None:
    df = read_csv_required(CSV_TOP_GEOMETRY)
    df = numeric(
        df,
        [
            "ring_inner_radius",
            "radial_width",
            "ring_outer_radius",
            "ring_thickness",
            "threshold_hole_radius",
        ],
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    combo_cols = ["radial_width", "ring_thickness"]
    combos = (
        df[combo_cols].drop_duplicates().sort_values(combo_cols).reset_index(drop=True)
    )

    # Small x offsets keep equal-inner-radius configurations distinguishable.
    offsets = np.linspace(-0.06, 0.06, len(combos)) if len(combos) > 1 else [0.0]

    for i, row in combos.iterrows():
        radial_width = row["radial_width"]
        thickness = row["ring_thickness"]

        group = df[
            (df["radial_width"] == radial_width) & (df["ring_thickness"] == thickness)
        ].sort_values("ring_inner_radius")

        ax.scatter(
            group["ring_inner_radius"] + offsets[i],
            group["threshold_hole_radius"],
            marker=MARKERS[i % len(MARKERS)],
            s=55,
            color=COLORS[i % len(COLORS)],
            label=f"$w={radial_width:g}$ mm, $t={thickness:g}$ mm",
            zorder=3,
        )

    x_min = float(df["ring_inner_radius"].min())
    x_max = float(df["ring_inner_radius"].max())
    xs = np.linspace(x_min, x_max, 100)

    ax.plot(
        xs,
        xs,
        color=KIT_GREY,
        linestyle="--",
        linewidth=1.5,
        label="$r_h^* = r_{inner}$",
    )

    grouped = (
        df.groupby("ring_inner_radius", as_index=False)["threshold_hole_radius"]
        .mean()
        .sort_values("ring_inner_radius")
    )

    ax.plot(
        grouped["ring_inner_radius"],
        grouped["threshold_hole_radius"],
        color=KIT_DEEP_BLUE,
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="Mean threshold",
        zorder=4,
    )

    ax.set_xlabel("Ring inner radius $r_{inner}$ [mm]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(fontsize=8, frameon=True)

    save_figure(fig, "fig_3_top_view_geometry_relation")


# -----------------------------------------------------------------------------
# Figure 4: Oblique regime, outer radius effect
# -----------------------------------------------------------------------------


def plot_oblique_outer_radius_effect() -> None:
    df = read_csv_required(CSV_FIXED_INNER)
    df = numeric(
        df,
        [
            "view_angle_deg",
            "ring_outer_radius",
            "radial_width",
            "threshold_hole_radius",
        ],
    )

    y_infeasible = infeasible_y_value(df, margin=1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    outer_values = sorted(df["ring_outer_radius"].dropna().unique())

    for i, outer_radius in enumerate(outer_values):
        group = df[df["ring_outer_radius"] == outer_radius].sort_values(
            "view_angle_deg"
        )
        finite = group.dropna(subset=["threshold_hole_radius"])

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            finite["view_angle_deg"],
            finite["threshold_hole_radius"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=f"$r_{{outer}}={outer_radius:g}$ mm",
        )

        plot_infeasible_points(
            ax,
            group,
            x_col="view_angle_deg",
            y_value=y_infeasible,
            color=color,
        )

    add_infeasible_axis_margin(ax, y_infeasible)

    ax.set_xlabel("Viewing angle $\\theta$ [deg]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(title="Outer radius", frameon=True)

    save_figure(fig, "fig_4_oblique_outer_radius_effect")


# -----------------------------------------------------------------------------
# Figure 5: Depth effect under oblique views
# -----------------------------------------------------------------------------


def plot_depth_effect_under_oblique_views() -> None:
    df = read_csv_required(CSV_DEPTH_OBLIQUE)
    df = numeric(
        df,
        [
            "view_angle_deg",
            "depth_from_opening_plane",
            "threshold_hole_radius",
            "ring_z",
        ],
    )

    y_infeasible = infeasible_y_value(df, margin=1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    angle_values = sorted(df["view_angle_deg"].dropna().unique())

    for i, angle in enumerate(angle_values):
        group = df[df["view_angle_deg"] == angle].sort_values(
            "depth_from_opening_plane"
        )
        finite = group.dropna(subset=["threshold_hole_radius"])

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            finite["depth_from_opening_plane"],
            finite["threshold_hole_radius"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=f"$\\theta={angle:g}^\\circ$",
        )

        plot_infeasible_points(
            ax,
            group,
            x_col="depth_from_opening_plane",
            y_value=y_infeasible,
            color=color,
        )

    add_infeasible_axis_margin(ax, y_infeasible)

    ax.set_xlabel("Depth from opening plane $d$ [mm]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(title="Viewing angle", frameon=True)

    save_figure(fig, "fig_5_depth_effect_under_oblique_views")


# -----------------------------------------------------------------------------
# Optional Figure 6: Fixed outer radius, varying inner radius
# -----------------------------------------------------------------------------


def plot_fixed_outer_inner_sweep() -> None:
    df = read_csv_required(CSV_FIXED_OUTER)
    df = numeric(
        df,
        [
            "view_angle_deg",
            "ring_inner_radius",
            "ring_outer_radius",
            "threshold_hole_radius",
        ],
    )

    y_infeasible = infeasible_y_value(df, margin=1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    inner_values = sorted(df["ring_inner_radius"].dropna().unique())
    offsets = (
        np.linspace(-0.25, 0.25, len(inner_values)) if len(inner_values) > 1 else [0.0]
    )

    for i, inner_radius in enumerate(inner_values):
        group = df[df["ring_inner_radius"] == inner_radius].sort_values(
            "view_angle_deg"
        )
        finite = group.dropna(subset=["threshold_hole_radius"])

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        x_offset = offsets[i]

        ax.plot(
            finite["view_angle_deg"] + x_offset,
            finite["threshold_hole_radius"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=f"$r_{{inner}}={inner_radius:g}$ mm",
        )

        plot_infeasible_points(
            ax,
            group,
            x_col="view_angle_deg",
            y_value=y_infeasible,
            x_offset=x_offset,
            color=color,
        )

    add_infeasible_axis_margin(ax, y_infeasible)

    ax.set_xlabel("Viewing angle $\\theta$ [deg]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(title="Inner radius", frameon=True)

    save_figure(fig, "fig_6_fixed_outer_inner_sweep")


# -----------------------------------------------------------------------------
# Optional Figure 7: Top-view depth sweep
# -----------------------------------------------------------------------------


def plot_top_view_depth_no_effect() -> None:
    df = read_csv_required(CSV_TOP_DEPTH)
    df = numeric(
        df,
        [
            "ring_z",
            "depth_from_opening_plane",
            "threshold_hole_radius",
        ],
    )

    df = df.sort_values("depth_from_opening_plane")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(
        df["depth_from_opening_plane"],
        df["threshold_hole_radius"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        color=KIT_DEEP_BLUE,
        label="Top-view threshold",
    )

    ax.set_xlabel("Depth from opening plane $d$ [mm]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(frameon=True)

    save_figure(fig, "fig_7_top_view_depth_no_effect")


# -----------------------------------------------------------------------------
# Optional Figure 8: Inner radius sweep with fixed radial width
# -----------------------------------------------------------------------------


def plot_inner_radius_with_fixed_width_sweep() -> None:
    """Support figure for critical-angle sweep with fixed radial width.

    This figure is useful when discussing whether the inner radius or the outer
    radius dominates the oblique visibility limit. It is optional because the
    main text may already be sufficiently supported by Figures 4 and 6.
    """
    df = read_csv_required(CSV_CRITICAL_RING_INNER)
    df = numeric(
        df,
        [
            "view_angle_deg",
            "ring_inner_radius",
            "ring_outer_radius",
            "radial_width",
            "threshold_hole_radius",
        ],
    )

    y_infeasible = infeasible_y_value(df, margin=1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    inner_values = sorted(df["ring_inner_radius"].dropna().unique())
    offsets = (
        np.linspace(-0.25, 0.25, len(inner_values)) if len(inner_values) > 1 else [0.0]
    )

    for i, inner_radius in enumerate(inner_values):
        group = df[df["ring_inner_radius"] == inner_radius].sort_values(
            "view_angle_deg"
        )
        finite = group.dropna(subset=["threshold_hole_radius"])

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        x_offset = offsets[i]

        ax.plot(
            finite["view_angle_deg"] + x_offset,
            finite["threshold_hole_radius"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=f"$r_{{inner}}={inner_radius:g}$ mm",
        )

        plot_infeasible_points(
            ax,
            group,
            x_col="view_angle_deg",
            y_value=y_infeasible,
            x_offset=x_offset,
            color=color,
        )

    add_infeasible_axis_margin(ax, y_infeasible)

    ax.set_xlabel("Viewing angle $\\theta$ [deg]")
    ax.set_ylabel("Threshold opening radius $r_h^*$ [mm]")
    ax.legend(title="Inner radius", frameon=True)

    save_figure(fig, "fig_8_inner_radius_fixed_width_sweep")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_top_view_threshold()
    plot_threshold_vs_view_angle()
    plot_top_view_geometry_relation()
    plot_oblique_outer_radius_effect()
    plot_depth_effect_under_oblique_views()

    # Optional support figures. These can be used in the appendix if they are
    # not needed in the main experiments chapter.
    if CSV_FIXED_OUTER.exists():
        plot_fixed_outer_inner_sweep()

    if CSV_TOP_DEPTH.exists():
        plot_top_view_depth_no_effect()

    if CSV_CRITICAL_RING_INNER.exists():
        plot_inner_radius_with_fixed_width_sweep()


if __name__ == "__main__":
    main()
