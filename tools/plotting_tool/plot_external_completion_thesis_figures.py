from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Default paths
# -----------------------------------------------------------------------------

DEFAULT_RESULTS_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath(
        "Bau_Yuan/02_Figures/04_Chapter_6/02_Section_6.2/01_external_completion_analysis_(subsection 6.2.4-6.2.6))/source"
    )
)

DEFAULT_OUT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath(
        "Bau_Yuan/02_Figures/04_Chapter_6/02_Section_6.2/external_completion_analysis"
    )
)


# -----------------------------------------------------------------------------
# KIT / wbk visual style
# -----------------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838F"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
KIT_GREY = "#D9DEE2"
DARK = "#222222"
MUTED = "#666666"
LIGHT_GRID = "#E5E8EB"

FULL_SUCCESS_LABEL = "All parts removed"
INCOMPLETE_LABEL = "Not fully disassembled"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "figure.titlesize": 9,
        "axes.grid": True,
        "grid.color": LIGHT_GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": DARK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def parse_bool(value: Any) -> bool:
    """Parse bool-like values from CSV/JSON fields."""
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    return str(value).strip().lower() in {"true", "1", "yes", "y", "t", "success"}


def parse_float(value: Any, default: float = np.nan) -> float:
    """Parse a float robustly."""
    if value is None:
        return default

    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass

    return default


def first_existing(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Return the first available field from a list of possible names."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...],
) -> list[Path]:
    """Save a figure in all requested formats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        generated.append(path)

    plt.close(fig)
    return generated


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
            fontsize=7,
            color=DARK,
        )


# -----------------------------------------------------------------------------
# Input folder selection
# -----------------------------------------------------------------------------


def resolve_group_dirs(results_root: Path) -> dict[str, Path]:
    """
    Resolve preferred result folders.

    The supplementary folder may contain both old and new runs. For the thesis
    figures, prefer the cleaned/new result folders where available and avoid
    the reverse-check variants for aggregate completion statistics.
    """
    candidates = {
        "joint_assembly": [
            "joint_assembly",
        ],
        "joint_assembly_rotation": [
            "joint_assembly_rotation",
        ],
        "multi_assembly": [
            "multi_assembly",
        ],
    }

    resolved: dict[str, Path] = {}

    for group, names in candidates.items():
        for name in names:
            folder = results_root / name
            if folder.exists():
                resolved[group] = folder
                break

        if group not in resolved:
            raise FileNotFoundError(
                f"No result folder found for group '{group}' under {results_root}. "
                f"Tried: {names}"
            )

    return resolved


# -----------------------------------------------------------------------------
# Reading CSV / JSONL results
# -----------------------------------------------------------------------------


def normalize_result_row(
    row: dict[str, Any], group: str, source_path: Path
) -> dict[str, Any]:
    """Normalize one raw CSV/JSONL row into the columns used by the plots."""
    assembly_name = str(
        first_existing(
            row,
            ["assembly_name", "case_name", "case", "name"],
            default="",
        )
    )

    completion = parse_float(
        first_existing(
            row,
            [
                "completion",
                "completion_rate",
                "final_completion",
                "removed_part_completion",
            ],
            default=None,
        ),
        default=np.nan,
    )

    success_value = first_existing(
        row,
        [
            "success",
            "success_bool",
            "completed",
            "all_parts_removed",
            "goal_reached",
            "is_success",
        ],
        default=None,
    )

    if success_value is None:
        success_bool = bool(np.isfinite(completion) and completion >= 0.999)
    else:
        success_bool = parse_bool(success_value)

    if not np.isfinite(completion):
        completion = 1.0 if success_bool else 0.0

    initial_num_parts = parse_float(
        first_existing(
            row,
            [
                "initial_num_parts",
                "num_parts",
                "n_parts",
                "initial_parts",
            ],
            default=None,
        ),
        default=np.nan,
    )

    initial_num_connections = parse_float(
        first_existing(
            row,
            [
                "initial_num_connections",
                "num_connections",
                "n_connections",
                "initial_connections",
            ],
            default=None,
        ),
        default=np.nan,
    )

    po_raw = first_existing(
        row,
        [
            "partial_observability_observed",
            "po_observed",
            "po_bool",
            "partial_observability",
        ],
        default=None,
    )

    if po_raw is None:
        po_count = parse_float(
            first_existing(
                row,
                [
                    "partial_observability_event_count",
                    "po_event_count",
                    "num_po_events",
                    "partial_observability_events",
                ],
                default=0,
            ),
            default=0.0,
        )
        po_bool = po_count > 0
    else:
        po_bool = parse_bool(po_raw)

    return {
        "assembly_name": assembly_name,
        "group": group,
        "success_bool": bool(success_bool),
        "completion": float(np.clip(completion, 0.0, 1.0)),
        "initial_num_parts": initial_num_parts,
        "initial_num_connections": initial_num_connections,
        "po_bool": bool(po_bool),
        "_source_file": str(source_path),
        "_source_mtime": source_path.stat().st_mtime,
    }


def read_csv_file(path: Path, group: str) -> list[dict[str, Any]]:
    """Read one CSV result file."""
    df = pd.read_csv(path)
    return [
        normalize_result_row(row.to_dict(), group=group, source_path=path)
        for _, row in df.iterrows()
    ]


def read_jsonl_file(path: Path, group: str) -> list[dict[str, Any]]:
    """Read one JSONL result file."""
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue

            rows.append(normalize_result_row(raw, group=group, source_path=path))

    return rows


def read_group_results(group: str, folder: Path) -> pd.DataFrame:
    """
    Read all CSV and JSONL files for one group.

    If an assembly appears more than once, keep the row from the newest source
    file. This prevents double counting when older and newer runs coexist.
    """
    files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.jsonl")))

    if not files:
        raise FileNotFoundError(f"No CSV or JSONL files found in: {folder}")

    rows: list[dict[str, Any]] = []

    for path in files:
        if path.suffix.lower() == ".csv":
            rows.extend(read_csv_file(path, group))
        elif path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl_file(path, group))

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No result rows could be loaded from: {folder}")

    if "assembly_name" not in df.columns:
        raise KeyError(f"Missing assembly_name after loading {folder}")

    df = df[df["assembly_name"].astype(str).str.len() > 0].copy()

    df = (
        df.sort_values(["assembly_name", "_source_mtime"])
        .drop_duplicates(subset=["assembly_name"], keep="last")
        .reset_index(drop=True)
    )

    return df


def load_all_results(
    results_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load all external result groups."""
    group_dirs = resolve_group_dirs(results_root)

    results = {
        group: read_group_results(group, folder) for group, folder in group_dirs.items()
    }

    return results, group_dirs


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------


def summarize_group(name: str, df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df))
    success = int(df["success_bool"].sum())
    incomplete = total - success
    mean_completion = float(df["completion"].mean()) if total else 0.0

    return {
        "name": name,
        "total": total,
        "all_parts_removed": success,
        "not_fully_disassembled": incomplete,
        "full_success_rate": success / total if total else 0.0,
        "mean_removed_part_completion": mean_completion,
    }


def print_summary(name: str, df: pd.DataFrame) -> None:
    s = summarize_group(name, df)
    print(
        f"{name:38s} "
        f"N={s['total']:4d}  "
        f"all_parts_removed={s['all_parts_removed']:4d}  "
        f"not_fully_disassembled={s['not_fully_disassembled']:4d}  "
        f"full_success_rate={s['full_success_rate'] * 100.0:6.2f}%  "
        f"mean_removed_part_completion={s['mean_removed_part_completion'] * 100.0:6.2f}%"
    )


# -----------------------------------------------------------------------------
# Binning
# -----------------------------------------------------------------------------


def removed_part_completion_bin_label(value: float) -> str:
    """Removed-part completion bin label."""
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
    """Binning for multi-assembly part-count analysis."""
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


def connection_count_bin(row: pd.Series) -> str:
    """Binning for absolute reconstructed connection count."""
    n_conn = row.get("initial_num_connections", np.nan)

    if pd.isna(n_conn):
        return "unknown"

    n_conn = int(n_conn)

    if 0 <= n_conn <= 5:
        return "0–5"
    if 6 <= n_conn <= 10:
        return "6–10"
    if 11 <= n_conn <= 20:
        return "11–20"
    if 21 <= n_conn <= 40:
        return "21–40"
    if 41 <= n_conn <= 60:
        return "41–60"
    if 61 <= n_conn <= 90:
        return "61–90"
    if 91 <= n_conn <= 130:
        return "91–130"
    if n_conn > 130:
        return ">130"

    return "other"


CONNECTION_COUNT_BIN_ORDER = [
    "0–5",
    "6–10",
    "11–20",
    "21–40",
    "41–60",
    "61–90",
    "91–130",
    ">130",
    "other",
    "unknown",
]


def connection_to_part_ratio_bin(row: pd.Series) -> str:
    """Binning for connection-to-part ratio analysis."""
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


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_external_success_summary(
    *,
    joint: pd.DataFrame,
    rotation: pd.DataFrame,
    multi: pd.DataFrame,
    multi_without_po: pd.DataFrame,
    multi_po: pd.DataFrame,
    all_external: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot completion rates across external assembly subsets."""
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

        labels.append(f"{name}\nN = {total}")
        completed_rates.append(completed_rate)
        incomplete_rates.append(incomplete_rate)
        completed_counts.append(completed)
        incomplete_counts.append(incomplete)
        totals.append(total)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

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

    for i, (c_rate, f_rate, c_count, f_count, total) in enumerate(
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
                fontsize=7,
                color="white",
            )
        elif c_rate > 0:
            ax.text(
                c_rate + 1.0,
                i,
                f"{c_count}/{total}\n{c_rate:.1f}%",
                ha="left",
                va="center",
                fontsize=7,
                color=DARK,
            )

        if f_rate >= 12.0:
            ax.text(
                c_rate + f_rate / 2.0,
                i,
                f"{f_count}/{total}\n{f_rate:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
        elif f_rate > 0.0:
            ax.text(
                min(100.0, c_rate + f_rate) + 1.0,
                i,
                f"{f_count}/{total}\n{f_rate:.1f}%",
                ha="left",
                va="center",
                fontsize=7,
                color=DARK,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of cases [%]")
    ax.set_title(
        "Completion rates across external assembly subsets", fontweight="bold", pad=6
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    out_dir = out_root / "completion_rates_external_subsets"
    generated = save_figure(fig, out_dir, "completion_rates_external_subsets", formats)
    return generated


def plot_multi_completion_hist(
    multi: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot removed-part completion distribution for multi-assembly cases."""
    df = multi.copy()
    df["completion_bin"] = df["completion"].apply(removed_part_completion_bin_label)

    order = ["0%", "0–25%", "25–50%", "50–75%", "75–100%", "100%"]
    counts = df["completion_bin"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.8, 3.5))

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

    ax.set_title(
        "Removed-part completion distribution for multi-assembly cases",
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("Removed-part completion range")
    ax.set_ylabel("Number of cases")
    ax.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 1)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    out_dir = out_root / "multi_completion_distribution"
    return save_figure(fig, out_dir, "multi_completion_distribution", formats)


def plot_completion_by_part_count_bin(
    multi: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot full disassembly outcome by initial part-count bin."""
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

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

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
    ax.set_xticklabels(grouped.index, rotation=0)
    ax.set_xlabel("Initial number of parts")
    ax.set_ylabel("Number of cases")
    ax.set_title("Full disassembly outcome by assembly size", fontweight="bold", pad=6)
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    out_dir = out_root / "completion_by_part_count_bin"
    return save_figure(fig, out_dir, "completion_by_part_count_bin", formats)


def plot_completion_by_connection_to_part_ratio_bin(
    multi: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot full disassembly outcome by connection-to-part ratio."""
    df = multi.copy()

    if "initial_num_connections" not in df.columns:
        raise KeyError("Column 'initial_num_connections' is required.")

    df["connection_to_part_ratio"] = df["initial_num_connections"].astype(float) / df[
        "initial_num_parts"
    ].astype(float)
    df["connection_to_part_ratio_bin"] = df.apply(connection_to_part_ratio_bin, axis=1)

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

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

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
    ax.set_xlabel("Connection-to-part ratio")
    ax.set_ylabel("Number of cases")
    ax.set_title(
        "Full disassembly outcome by connection-to-part ratio", fontweight="bold", pad=6
    )
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    out_dir = out_root / "completion_by_connection_to_part_ratio"
    return save_figure(fig, out_dir, "completion_by_connection_to_part_ratio", formats)


def plot_completion_by_connection_count_bin(
    multi: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Plot full disassembly outcome by absolute reconstructed connection count."""
    df = multi.copy()

    if "initial_num_connections" not in df.columns:
        raise KeyError("Column 'initial_num_connections' is required.")

    df["connection_count_bin"] = df.apply(connection_count_bin, axis=1)

    grouped = (
        df.groupby(["connection_count_bin", "success_bool"])
        .size()
        .unstack(fill_value=0)
    )

    for col in [False, True]:
        if col not in grouped.columns:
            grouped[col] = 0

    ordered_bins = [b for b in CONNECTION_COUNT_BIN_ORDER if b in grouped.index]
    grouped = grouped.loc[ordered_bins]

    completed = grouped[True].to_numpy()
    incomplete = grouped[False].to_numpy()

    x = np.arange(len(grouped.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

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
    ax.set_xlabel("Initial number of reconstructed connections")
    ax.set_ylabel("Number of cases")
    ax.set_title(
        "Full disassembly outcome by reconstructed connection count",
        fontweight="bold",
        pad=6,
    )
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    out_dir = out_root / "completion_by_connection_count_bin"
    return save_figure(fig, out_dir, "completion_by_connection_count_bin", formats)


def plot_incomplete_completion_stats(
    df: pd.DataFrame,
    out_root: Path,
    formats: tuple[str, ...],
    stem: str,
    title: str,
) -> list[Path]:
    """
    Plot mean, median, and 90th percentile completion for incomplete rollouts.

    Completion is shown in percent.
    """
    incomplete = df[~df["success_bool"]].copy()

    if incomplete.empty:
        raise ValueError(f"No incomplete cases available for plot: {stem}")

    values = incomplete["completion"].astype(float).to_numpy() * 100.0

    stats = {
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "P90": float(np.percentile(values, 90)),
    }

    labels = list(stats.keys())
    heights = np.asarray(list(stats.values()), dtype=float)

    colors = [KIT_BLUE_GREEN, KIT_DEEP_BLUE, KIT_YELLOW]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    bars = ax.bar(
        labels,
        heights,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, value in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )

    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_ylabel("Completion [%]")
    ax.set_ylim(0, 100)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    out_dir = out_root / stem
    generated = save_figure(fig, out_dir, stem, formats)

    summary = {
        "plot": stem,
        "title": title,
        "num_cases": int(len(df)),
        "num_incomplete_cases": int(len(incomplete)),
        "mean_completion_percent": stats["Mean"],
        "median_completion_percent": stats["Median"],
        "p90_completion_percent": stats["P90"],
    }

    summary_path = out_dir / f"{stem}_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated.append(summary_path)

    return generated


# -----------------------------------------------------------------------------
# Source and summary files
# -----------------------------------------------------------------------------


def write_global_source_summary(
    out_root: Path,
    results_root: Path,
    group_dirs: dict[str, Path],
    results: dict[str, pd.DataFrame],
    summaries: dict[str, Any],
    generated_files: list[Path],
) -> Path:
    """Write a JSON summary for traceability."""
    source_dir = out_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "results_root": str(results_root),
        "group_dirs": {k: str(v) for k, v in group_dirs.items()},
        "group_summaries": summaries,
        "loaded_rows": {group: int(len(df)) for group, df in results.items()},
        "source_files": {
            group: sorted(df["_source_file"].dropna().unique().tolist())
            for group, df in results.items()
        },
        "generated_files": [str(p) for p in generated_files],
    }

    path = source_dir / "external_completion_analysis_source_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_readme(out_root: Path) -> Path:
    """Write a lightweight source README."""
    source_dir = out_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    path = source_dir / "README_source.txt"
    path.write_text(
        "External completion analysis source package\n\n"
        "This folder summarizes how the external completion figures were generated.\n\n"
        "Inputs:\n"
        "- Supplementary experiment result logs from Experiment 6.2\n"
        "- CSV and/or JSONL files in external_cases_results\n\n"
        "Outputs:\n"
        "- PNG for Word insertion\n"
        "- PDF for print-quality export\n"
        "- SVG for editable/vector post-processing\n"
        "- JSON source summary for traceability\n\n"
        "The plotting script deduplicates repeated assemblies by keeping the row\n"
        "from the newest source file.\n",
        encoding="utf-8",
    )
    return path


def copy_script_to_source(out_root: Path, script_path: Path) -> Path:
    """Copy this script into the source folder."""
    source_dir = out_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    dst = source_dir / script_path.name
    shutil.copy2(script_path, dst)
    return dst


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render thesis-style external completion analysis figures."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root folder containing external_cases_results.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root for thesis figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--copy-source",
        action="store_true",
        help="Copy this plotting script into the source folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = args.results_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    formats = tuple(args.formats)

    if not results_root.exists():
        raise FileNotFoundError(f"results-root not found: {results_root}")

    results, group_dirs = load_all_results(results_root)

    joint = results["joint_assembly"]
    rotation = results["joint_assembly_rotation"]
    multi = results["multi_assembly"]

    all_external = pd.concat([joint, rotation, multi], ignore_index=True)
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

    summaries = {
        "joint_assembly": summarize_group("joint_assembly", joint),
        "joint_assembly_rotation": summarize_group("joint_assembly_rotation", rotation),
        "multi_assembly": summarize_group("multi_assembly", multi),
        "multi_assembly_without_PO_cases": summarize_group(
            "multi_assembly_without_PO_cases", multi_without_po
        ),
        "multi_assembly_PO_cases": summarize_group("multi_assembly_PO_cases", multi_po),
        "all_external": summarize_group("all_external", all_external),
    }

    generated: list[Path] = []

    generated.extend(
        plot_external_success_summary(
            joint=joint,
            rotation=rotation,
            multi=multi,
            multi_without_po=multi_without_po,
            multi_po=multi_po,
            all_external=all_external,
            out_root=out_root,
            formats=formats,
        )
    )

    generated.extend(
        plot_multi_completion_hist(
            multi=multi,
            out_root=out_root,
            formats=formats,
        )
    )

    generated.extend(
        plot_completion_by_part_count_bin(
            multi=multi,
            out_root=out_root,
            formats=formats,
        )
    )

    generated.extend(
        plot_completion_by_connection_to_part_ratio_bin(
            multi=multi,
            out_root=out_root,
            formats=formats,
        )
    )
    generated.extend(
        plot_completion_by_connection_count_bin(
            multi=multi,
            out_root=out_root,
            formats=formats,
        )
    )

    generated.extend(
        plot_incomplete_completion_stats(
            df=multi,
            out_root=out_root,
            formats=formats,
            stem="failed_multi_completion_stats",
            title="Incomplete multi-assembly completion",
        )
    )

    generated.extend(
        plot_incomplete_completion_stats(
            df=multi_without_po,
            out_root=out_root,
            formats=formats,
            stem="failed_multi_without_po_completion_stats",
            title="Incomplete multi-assembly completion without PO events",
        )
    )

    generated.extend(
        plot_incomplete_completion_stats(
            df=multi_po,
            out_root=out_root,
            formats=formats,
            stem="failed_multi_po_completion_stats",
            title="Incomplete multi-assembly completion with PO events",
        )
    )

    generated.append(
        write_global_source_summary(
            out_root=out_root,
            results_root=results_root,
            group_dirs=group_dirs,
            results=results,
            summaries=summaries,
            generated_files=generated,
        )
    )
    generated.append(write_readme(out_root))

    if args.copy_source:
        generated.append(copy_script_to_source(out_root, Path(__file__).resolve()))

    print("\nGenerated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
