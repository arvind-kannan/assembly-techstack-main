from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath(
        "Bau_Yuan/02_Figures/04_Chapter_6/02_Section_6.2/01_external_completion_analysis_(subsection 6.2.4-6.2.6))/source"
    )
)
RESULTS_ROOT = ROOT

RESULT_DIRS = {
    "joint_assembly": RESULTS_ROOT / "joint_assembly",
    "joint_assembly_rotation": RESULTS_ROOT / "joint_assembly_rotation",
    "multi_assembly": RESULTS_ROOT / "multi_assembly",
}

OUT_DIR = ROOT / "figures" / "external_completion_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
TEXT_COLOR = "#222222"

FULL_SUCCESS_LABEL = "All parts removed"
INCOMPLETE_LABEL = "Not fully disassembled"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------


def parse_bool(value: Any) -> bool:
    """Parse bool-like values from CSV fields."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def read_group_results(group: str, folder: Path) -> pd.DataFrame:
    """
    Read all CSV result files for one group.

    If the same assembly appears in several CSV files, keep the row from
    the newest CSV file. This avoids double counting after repeated runs.
    """
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    frames = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["group"] = group
        df["_source_csv"] = str(csv_path)
        df["_source_mtime"] = csv_path.stat().st_mtime
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    required = ["assembly_name", "success", "completion", "initial_num_parts"]
    for col in required:
        if col not in out.columns:
            raise KeyError(f"Required column '{col}' missing in files from {folder}")

    out["success_bool"] = out["success"].apply(parse_bool)

    out["completion"] = pd.to_numeric(
        out["completion"],
        errors="coerce",
    ).fillna(0.0)
    out["completion"] = out["completion"].clip(lower=0.0, upper=1.0)

    out["initial_num_parts"] = pd.to_numeric(
        out["initial_num_parts"],
        errors="coerce",
    )

    if "initial_num_connections" in out.columns:
        out["initial_num_connections"] = pd.to_numeric(
            out["initial_num_connections"],
            errors="coerce",
        )

    if "partial_observability_observed" in out.columns:
        out["po_bool"] = out["partial_observability_observed"].apply(parse_bool)
    else:
        out["po_bool"] = False

    out = (
        out.sort_values(["assembly_name", "_source_mtime"])
        .drop_duplicates(subset=["assembly_name"], keep="last")
        .reset_index(drop=True)
    )

    return out


def load_all_results() -> dict[str, pd.DataFrame]:
    """Load all external evaluation result groups."""
    return {
        group: read_group_results(group, folder)
        for group, folder in RESULT_DIRS.items()
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG only."""
    png_path = OUT_DIR / f"{name}.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[SAVED] {png_path}")


def add_bar_labels(ax, bars, *, fmt: str = "{:.0f}", dy: float = 2.0) -> None:
    """Add numeric labels above bars."""
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def print_summary(name: str, df: pd.DataFrame) -> None:
    """Print compact result summary."""
    total = len(df)
    success = int(df["success_bool"].sum())
    incomplete = total - success
    mean_completion = float(df["completion"].mean() * 100.0) if total else 0.0

    print(
        f"{name:34s} "
        f"N={total:4d}  "
        f"all_parts_removed={success:4d}  "
        f"not_fully_disassembled={incomplete:4d}  "
        f"full_success_rate={success / total * 100.0 if total else 0.0:6.2f}%  "
        f"mean_removed_part_completion={mean_completion:6.2f}%"
    )


# ---------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------


def removed_part_completion_bin_label(value: float) -> str:
    """
    Removed-part completion bins.

    Removed-part completion means:
        removed target parts / initially present target parts.
    """
    v = float(value)

    if np.isclose(v, 0.0):
        return "0%"
    if np.isclose(v, 1.0):
        return "100%"
    if 0.0 < v <= 0.25:
        return "0–25%"
    if 0.25 < v <= 0.50:
        return "25–50%"
    if 0.50 < v <= 0.75:
        return "50–75%"
    return "75–100%"


def part_count_bin(row: pd.Series) -> str:
    """
    Binning for multi-assembly part-count analysis.

    The bins follow the stratified multi-assembly sampling logic:
        5–9      low part-count region
        10–37    middle region, shown in 4-part intervals
        38–42    high part-count region
    """
    n = row.get("initial_num_parts", np.nan)

    if pd.isna(n):
        return "unknown"

    n = int(n)

    if 5 <= n <= 9:
        return "5–9"
    if 10 <= n <= 13:
        return "10–13"
    if 14 <= n <= 17:
        return "14–17"
    if 18 <= n <= 21:
        return "18–21"
    if 22 <= n <= 25:
        return "22–25"
    if 26 <= n <= 29:
        return "26–29"
    if 30 <= n <= 33:
        return "30–33"
    if 34 <= n <= 37:
        return "34–37"
    if 38 <= n <= 42:
        return "38–42"

    return "other"


PART_BIN_ORDER = [
    "5–9",
    "10–13",
    "14–17",
    "18–21",
    "22–25",
    "26–29",
    "30–33",
    "34–37",
    "38–42",
    "other",
    "unknown",
]


def connection_to_part_ratio_bin(row: pd.Series) -> str:
    """
    Binning for connection-to-part ratio analysis.

    Ratio:
        initial_num_connections / initial_num_parts

    This ratio is used as a normalized proxy for geometric/topological
    constraint density.
    """
    n_conn = row.get("initial_num_connections", np.nan)
    n_parts = row.get("initial_num_parts", np.nan)

    if pd.isna(n_conn) or pd.isna(n_parts):
        return "unknown"

    n_conn = float(n_conn)
    n_parts = float(n_parts)

    if n_parts <= 0.0:
        return "unknown"

    ratio = n_conn / n_parts

    if ratio <= 0.5:
        return "0–0.5"
    if ratio <= 1.0:
        return "0.5–1.0"
    if ratio <= 1.5:
        return "1.0–1.5"
    if ratio <= 2.0:
        return "1.5–2.0"
    if ratio <= 3.0:
        return "2.0–3.0"
    if ratio <= 5.0:
        return "3.0–5.0"
    if ratio <= 8.0:
        return "5.0–8.0"
    return ">8.0"


RATIO_BIN_ORDER = [
    "0–0.5",
    "0.5–1.0",
    "1.0–1.5",
    "1.5–2.0",
    "2.0–3.0",
    "3.0–5.0",
    "5.0–8.0",
    ">8.0",
    "unknown",
]


# ---------------------------------------------------------------------
# Plot 1: external full-disassembly success summary
# ---------------------------------------------------------------------


def plot_external_success_summary(
    *,
    joint: pd.DataFrame,
    rotation: pd.DataFrame,
    multi: pd.DataFrame,
    multi_without_po: pd.DataFrame,
    multi_po: pd.DataFrame,
    all_external: pd.DataFrame,
) -> None:
    """
    Compact 100% stacked horizontal summary of external evaluation results.

    Each row represents one evaluated subset. The full bar corresponds to
    100% of cases in that subset and is split into full disassembly success
    and incomplete rollouts.
    """
    rows = [
        ("Joint assembly", joint),
        ("Rotational joint assembly", rotation),
        ("Multi assembly", multi),
        ("Multi assembly without PO events", multi_without_po),
        ("Multi assembly with PO events", multi_po),
        ("All external cases", all_external),
    ]

    labels = []
    completed_rates = []
    incomplete_rates = []
    completed_counts = []
    incomplete_counts = []
    totals = []

    for name, df in rows:
        total = len(df)
        completed = int(df["success_bool"].sum())
        incomplete = total - completed

        completed_rate = completed / total * 100.0 if total > 0 else 0.0
        incomplete_rate = incomplete / total * 100.0 if total > 0 else 0.0

        labels.append(f"{name}\nN = {total}, all parts removed = {completed_rate:.1f}%")
        completed_rates.append(completed_rate)
        incomplete_rates.append(incomplete_rate)
        completed_counts.append(completed)
        incomplete_counts.append(incomplete)
        totals.append(total)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.8, 5.2))

    ax.barh(
        y,
        completed_rates,
        color=KIT_DEEP_BLUE,
        edgecolor="white",
        linewidth=0.8,
        label=FULL_SUCCESS_LABEL,
    )

    ax.barh(
        y,
        incomplete_rates,
        left=completed_rates,
        color=KIT_RED,
        edgecolor="white",
        linewidth=0.8,
        label=INCOMPLETE_LABEL,
    )

    for i, (
        c_rate,
        f_rate,
        c_count,
        f_count,
        total,
    ) in enumerate(
        zip(
            completed_rates,
            incomplete_rates,
            completed_counts,
            incomplete_counts,
            totals,
        )
    ):
        if c_rate >= 12.0:
            ax.text(
                c_rate / 2.0,
                i,
                f"{c_count}/{total}\n{c_rate:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        else:
            ax.text(
                c_rate + 1.2,
                i,
                f"{c_count}/{total}\n{c_rate:.1f}%",
                ha="left",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

        if f_rate >= 12.0:
            ax.text(
                c_rate + f_rate / 2.0,
                i,
                f"{f_count}/{total}\n{f_rate:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        elif f_rate > 0.0:
            ax.text(
                min(100.0, c_rate + f_rate) + 1.0,
                i,
                f"{f_count}/{total}\n{f_rate:.1f}%",
                ha="left",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of cases [%]")
    ax.set_title("Full Disassembly Success over External Cases")

    ax.grid(axis="x", alpha=0.30)

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0.0, 0.02, 0.86, 1.0))
    save_figure(fig, "external_success_summary_stacked")


# ---------------------------------------------------------------------
# Plot 2: removed-part completion histogram for 1000 multi assemblies
# ---------------------------------------------------------------------


def plot_multi_completion_hist(multi: pd.DataFrame) -> None:
    """
    Plot removed-part completion distribution for multi-assembly cases.

    Removed-part completion means:
        removed target parts / initially present target parts.
    """
    df = multi.copy()
    df["completion_bin"] = df["completion"].apply(removed_part_completion_bin_label)

    order = ["0%", "0–25%", "25–50%", "50–75%", "75–100%", "100%"]
    counts = df["completion_bin"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    colors = [
        KIT_RED,
        "#CF6A4C",
        KIT_YELLOW,
        KIT_BLUE_GREEN,
        "#3C98D9",
        KIT_DEEP_BLUE,
    ]

    bars = ax.bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    add_bar_labels(ax, bars, fmt="{:.0f}", dy=max(counts.max() * 0.015, 1.5))

    ax.set_title("Removed-Part Completion Distribution for Multi-Assembly Cases")
    ax.set_xlabel("Removed-part completion range")
    ax.set_ylabel("Number of multi-assembly cases")
    ax.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 1)
    ax.grid(axis="y", alpha=0.30)

    fig.tight_layout()
    save_figure(fig, "multi_completion_hist")


# ---------------------------------------------------------------------
# Plot 3: full disassembly outcome by part-count bin
# ---------------------------------------------------------------------


def plot_completion_by_part_count_bin(multi: pd.DataFrame) -> None:
    """
    Plot full disassembly outcome by initial part-count bin.

    The blue bars count cases where all parts were removed.
    The red bars count cases that terminated before all parts were removed.
    """
    df = multi.copy()
    df["part_bin"] = df.apply(part_count_bin, axis=1)

    grouped = df.groupby(["part_bin", "success_bool"]).size().unstack(fill_value=0)

    for col in [False, True]:
        if col not in grouped.columns:
            grouped[col] = 0

    ordered_bins = [b for b in PART_BIN_ORDER if b in grouped.index]
    grouped = grouped.loc[ordered_bins]

    completed = grouped[True].to_numpy()
    incomplete = grouped[False].to_numpy()

    x = np.arange(len(grouped.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    bars_completed = ax.bar(
        x - width / 2,
        completed,
        width,
        label=FULL_SUCCESS_LABEL,
        color=KIT_DEEP_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )

    bars_incomplete = ax.bar(
        x + width / 2,
        incomplete,
        width,
        label=INCOMPLETE_LABEL,
        color=KIT_RED,
        edgecolor="white",
        linewidth=0.8,
    )

    ymax = max(
        int(np.max(completed)) if len(completed) else 0,
        int(np.max(incomplete)) if len(incomplete) else 0,
        1,
    )

    add_bar_labels(ax, bars_completed, fmt="{:.0f}", dy=max(ymax * 0.015, 1.0))
    add_bar_labels(ax, bars_incomplete, fmt="{:.0f}", dy=max(ymax * 0.015, 1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("Initial number of parts")
    ax.set_ylabel("Number of multi-assembly cases")
    ax.set_title("Full Disassembly Outcome by Assembly Size")
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.30)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save_figure(fig, "completion_by_part_count_bin")


# ---------------------------------------------------------------------
# Plot 4: full disassembly outcome by connection-to-part ratio
# ---------------------------------------------------------------------


def plot_completion_by_connection_to_part_ratio_bin(multi: pd.DataFrame) -> None:
    """
    Plot full disassembly outcome by connection-to-part ratio.

    The blue bars count cases where all parts were removed.
    The red bars count cases that terminated before all parts were removed.
    """
    df = multi.copy()

    required_cols = ["initial_num_connections", "initial_num_parts"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is required for connection-to-part ratio plot."
            )

    df["connection_to_part_ratio"] = df["initial_num_connections"].astype(float) / df[
        "initial_num_parts"
    ].astype(float)

    df["connection_to_part_ratio_bin"] = df.apply(
        connection_to_part_ratio_bin,
        axis=1,
    )

    grouped = (
        df.groupby(["connection_to_part_ratio_bin", "success_bool"])
        .size()
        .unstack(fill_value=0)
    )

    for col in [False, True]:
        if col not in grouped.columns:
            grouped[col] = 0

    ordered_bins = [b for b in RATIO_BIN_ORDER if b in grouped.index]
    grouped = grouped.loc[ordered_bins]

    completed = grouped[True].to_numpy()
    incomplete = grouped[False].to_numpy()

    x = np.arange(len(grouped.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    bars_completed = ax.bar(
        x - width / 2,
        completed,
        width,
        label=FULL_SUCCESS_LABEL,
        color=KIT_DEEP_BLUE,
        edgecolor="white",
        linewidth=0.8,
    )

    bars_incomplete = ax.bar(
        x + width / 2,
        incomplete,
        width,
        label=INCOMPLETE_LABEL,
        color=KIT_RED,
        edgecolor="white",
        linewidth=0.8,
    )

    ymax = max(
        int(np.max(completed)) if len(completed) else 0,
        int(np.max(incomplete)) if len(incomplete) else 0,
        1,
    )

    add_bar_labels(
        ax,
        bars_completed,
        fmt="{:.0f}",
        dy=max(ymax * 0.015, 1.0),
    )
    add_bar_labels(
        ax,
        bars_incomplete,
        fmt="{:.0f}",
        dy=max(ymax * 0.015, 1.0),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("Connection-to-part ratio")
    ax.set_ylabel("Number of multi-assembly cases")
    ax.set_title("Full Disassembly Outcome by Connection-to-Part Ratio")
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.30)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save_figure(fig, "completion_by_connection_to_part_ratio_bin")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    results = load_all_results()

    joint = results["joint_assembly"]
    rotation = results["joint_assembly_rotation"]
    multi = results["multi_assembly"]

    all_external = pd.concat(
        [joint, rotation, multi],
        ignore_index=True,
    )

    multi_po = multi[multi["po_bool"]].copy()
    multi_without_po = multi[~multi["po_bool"]].copy()

    print("=" * 80)
    print("Loaded result summary")
    print("=" * 80)
    print_summary("joint_assembly", joint)
    print_summary("joint_assembly_rotation", rotation)
    print_summary("multi_assembly", multi)
    print_summary("multi_assembly_without_PO_cases", multi_without_po)
    print_summary("multi_assembly_PO_cases", multi_po)
    print_summary("all_external", all_external)

    plot_external_success_summary(
        joint=joint,
        rotation=rotation,
        multi=multi,
        multi_without_po=multi_without_po,
        multi_po=multi_po,
        all_external=all_external,
    )

    plot_multi_completion_hist(multi)

    plot_completion_by_part_count_bin(multi)

    plot_completion_by_connection_to_part_ratio_bin(multi)


if __name__ == "__main__":
    main()
