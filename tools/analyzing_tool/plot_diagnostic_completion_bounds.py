#!/usr/bin/env python3
from __future__ import annotations

"""
Create a thesis-ready plot for FP/FN diagnostic completion-rate bounds.

This script reads the overlap analysis output:

    ../private/results/diagnostic_overlap_analysis/overlap_bounds.csv

and creates an updated figure with:
    - shorter title
    - FP ∪ FN notation in labels
    - possible+ entries visually separated with hatch
    - compact legend
    - KIT/wbk-style colors
    - PNG and PDF export

Run:

cd <repo>

python tools/analyzing_tool/plot_diagnostic_completion_bounds.py

Optional:

python tools/analyzing_tool/plot_diagnostic_completion_bounds.py \
  --bounds-csv ../private/results/diagnostic_overlap_analysis/overlap_bounds.csv \
  --out-dir ../private/results/diagnostic_overlap_analysis \
  --formats png pdf
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
LIGHT_GRID = "#B8C1C8"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Bounds CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def label_for(raw_label: str) -> str:
    mapping = {
        "Measured baseline": "Measured baseline",
        "FP dependency-pattern upper bound": "FP dependency-pattern bound",
        "FN high-confidence upper bound": "FN high-confidence bound",
        "Combined FP union FN high": r"Combined FP $\cup$ FN high",
        "FN likely+ upper bound": "FN likely+ bound",
        "Combined FP union FN likely": r"Combined FP $\cup$ FN likely+",
        "FN possible+ screening pool": "FN possible+ screening pool",
        "Combined FP union FN possible": r"Combined FP $\cup$ FN possible+",
    }
    return mapping.get(raw_label, raw_label.replace(" union ", r" $\cup$ "))


def style_for(raw_label: str) -> tuple[str, str | None, str]:
    """
    Returns (facecolor, hatch, legend_category).
    """
    if raw_label == "Measured baseline":
        return KIT_DEEP_BLUE, None, "Measured result"

    if raw_label.startswith("Combined"):
        if "possible" in raw_label:
            return KIT_BLUE_GREEN, "///", "Combined upper bound"
        return KIT_BLUE_GREEN, None, "Combined upper bound"

    if raw_label.startswith("FP"):
        return KIT_YELLOW, None, "FP-related bound"

    if raw_label.startswith("FN possible"):
        return KIT_GREY, "///", "Broad screening pool"

    if raw_label.startswith("FN"):
        return KIT_RED, None, "FN-related bound"

    return MID_GREY, None, "Other"


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = [
        "Measured baseline",
        "FP dependency-pattern upper bound",
        "FN high-confidence upper bound",
        "Combined FP union FN high",
        "FN likely+ upper bound",
        "Combined FP union FN likely",
        "FN possible+ screening pool",
        "Combined FP union FN possible",
    ]

    by_label = {row.get("label", ""): row for row in rows}
    selected = [by_label[label] for label in order if label in by_label]

    missing = [label for label in order if label not in by_label]
    if missing:
        print(f"[WARN] Missing expected labels in CSV: {missing}")

    return selected


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------


def plot_diagnostic_bounds(
    *,
    bounds_csv: Path,
    out_dir: Path,
    formats: list[str],
    title: str,
) -> list[Path]:
    rows = select_rows(read_csv_rows(bounds_csv))
    if not rows:
        raise ValueError(f"No matching bound rows found in {bounds_csv}")

    labels = [label_for(row.get("label", "")) for row in rows]
    values = [parse_float(row.get("completion_percent"), 0.0) for row in rows]
    completed_counts = [parse_int(row.get("completed_upper_count"), 0) for row in rows]
    affected_counts = [
        parse_int(row.get("affected_incomplete_cases"), 0) for row in rows
    ]

    facecolors: list[str] = []
    hatches: list[str | None] = []
    legend_categories: list[str] = []

    for row in rows:
        color, hatch, category = style_for(row.get("label", ""))
        facecolors.append(color)
        hatches.append(hatch)
        legend_categories.append(category)

    fig_h = max(4.9, 0.53 * len(rows) + 1.45)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))

    y_positions = list(range(len(rows)))

    bars = ax.barh(
        y_positions,
        values,
        color=facecolors,
        edgecolor=DARK,
        linewidth=0.45,
        height=0.72,
    )

    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Completion rate [%]")
    ax.set_xlim(0, 100)
    ax.set_title(title, fontweight="bold", pad=10)

    ax.grid(axis="x", color=LIGHT_GRID, alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)

    # Use a minimal axis style.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, value, completed, affected in zip(
        y_positions, values, completed_counts, affected_counts
    ):
        label = f"{value:.1f}%"
        if affected > 0:
            label += f" ({completed}/1000)"

        # Keep labels inside the figure even for large possible+ bars.
        x_text = min(value + 1.0, 96.0)
        ax.text(
            x_text,
            yi,
            label,
            va="center",
            ha="left",
            fontsize=9,
            color=DARK,
        )

    # Separator after measured baseline and after conservative likely/high block.
    ax.axhline(0.5, color=LIGHT_GRID, linewidth=0.8, alpha=0.8)
    ax.axhline(5.5, color=LIGHT_GRID, linewidth=0.8, alpha=0.8)

    legend_handles = [
        Patch(facecolor=KIT_DEEP_BLUE, edgecolor=DARK, label="Measured result"),
        Patch(facecolor=KIT_YELLOW, edgecolor=DARK, label="FP-related bound"),
        Patch(facecolor=KIT_RED, edgecolor=DARK, label="FN-related bound"),
        Patch(facecolor=KIT_BLUE_GREEN, edgecolor=DARK, label=r"Combined FP $\cup$ FN"),
        Patch(
            facecolor=KIT_GREY,
            edgecolor=DARK,
            hatch="///",
            label="Broad possible+ pool",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.12),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        borderpad=0.7,
        columnspacing=1.1,
        handlelength=1.8,
    )

    footnote = (
        "Diagnostic upper bounds, not predicted performance. "
        "Combined bars use set union and avoid double-counting overlap."
    )
    fig.text(0.01, 0.012, footnote, fontsize=8, color=MID_GREY)

    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for ext in formats:
        ext = ext.lower().lstrip(".")
        out_path = out_dir / f"diagnostic_completion_bounds_thesis.{ext}"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot thesis-ready diagnostic completion-rate bounds."
    )
    parser.add_argument(
        "--bounds-csv",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("private/results/diagnostic_overlap_analysis/overlap_bounds.csv"),
        help="Path to overlap_bounds.csv produced by analyze_fp_fn_overlap.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("private/results/diagnostic_overlap_analysis"),
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Multi-Assembly Diagnostic Completion Bounds",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = plot_diagnostic_bounds(
        bounds_csv=args.bounds_csv.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        formats=list(args.formats),
        title=args.title,
    )

    print("=" * 80)
    print("[DONE] Wrote diagnostic completion bounds plot:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
