from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RESULTS_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath(
        "Bau_Yuan/02_Figures/04_Chapter_6/02_Section_6.2/01_external_completion_analysis_(subsection 6.2.4-6.2.6))/source"
    )
)

RESULT_DIRS = {
    "joint_assembly": RESULTS_ROOT / "joint_assembly",
    "joint_assembly_rotation": RESULTS_ROOT / "joint_assembly_rotation",
    "multi_assembly": RESULTS_ROOT / "multi_assembly",
}

OUT_DIR = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath(
        "Bau_Yuan/05_Supplementary_Materials/Code_Quality/source/02_runtime_performance"
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# KIT / wbk style
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_RED = "#B2372C"
KIT_GREY = "#D9DEE2"
DARK = "#222222"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.titlesize": 11,
        "text.color": DARK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_bool(value) -> bool:
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


def read_group_results(group: str, folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    jsonl_files = sorted(folder.glob("*.jsonl"))

    if not csv_files and not jsonl_files:
        raise FileNotFoundError(f"No CSV or JSONL result files found in: {folder}")

    rows = []

    for path in csv_files:
        df = pd.read_csv(path)
        df["_source_file"] = str(path)
        df["_source_mtime"] = path.stat().st_mtime
        rows.append(df)

    for path in jsonl_files:
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        if data:
            df = pd.DataFrame(data)
            df["_source_file"] = str(path)
            df["_source_mtime"] = path.stat().st_mtime
            rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    df["group"] = group

    name_col = None
    for candidate in ["assembly_name", "case_name", "case", "name"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if name_col is None:
        raise KeyError(f"No assembly/case name column found in {folder}")

    df["assembly_name_norm"] = df[name_col].astype(str)

    df = (
        df.sort_values(["assembly_name_norm", "_source_mtime"])
        .drop_duplicates(subset=["assembly_name_norm"], keep="last")
        .reset_index(drop=True)
    )

    return df


def row_has_runtime_or_setup_error(row: pd.Series) -> bool:
    """
    Conservative detection of setup/runtime errors from common result fields.

    If no explicit error field is present, the row is treated as no runtime/setup
    error. This matches the thesis evaluation, where incomplete no_progress
    rollouts are not runtime errors.
    """
    for key in [
        "exception",
        "has_exception",
        "runtime_error",
        "setup_error",
        "error",
    ]:
        if key in row.index and parse_bool(row[key]):
            return True

    for key in [
        "stop_reason",
        "reason",
        "status",
        "failure_reason",
    ]:
        if key in row.index and row[key] is not None:
            value = str(row[key]).strip().lower()
            if any(
                token in value
                for token in ["exception", "runtime_error", "setup_error", "traceback"]
            ):
                return True

    return False


def main() -> None:
    frames = []
    source_files = {}

    for group, folder in RESULT_DIRS.items():
        df = read_group_results(group, folder)
        frames.append(df)
        source_files[group] = sorted(df["_source_file"].dropna().unique().tolist())

    all_results = pd.concat(frames, ignore_index=True)

    total_cases = int(len(all_results))
    error_mask = all_results.apply(row_has_runtime_or_setup_error, axis=1)
    error_cases = int(error_mask.sum())
    stable_cases = total_cases - error_cases

    print(f"Total evaluated cases: {total_cases}")
    print(f"Processed without setup/runtime error: {stable_cases}")
    print(f"Setup/runtime errors: {error_cases}")

    # Matplotlib pie charts cannot show a zero-size wedge clearly.
    # Therefore, the pie shows the stable cases as the full circle and reports
    # the zero-error result explicitly in the center annotation.
    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    wedges, _ = ax.pie(
        [stable_cases],
        colors=[KIT_DEEP_BLUE],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )

    ax.text(
        0,
        0.08,
        f"{stable_cases}/{total_cases}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0,
        -0.12,
        "processed without\nsetup/runtime error",
        ha="center",
        va="center",
        fontsize=9,
        color=DARK,
    )

    ax.set_title("Runtime stability of external evaluation", fontweight="bold", pad=10)
    ax.axis("equal")

    # Small explicit zero-error label outside the pie.
    ax.text(
        0.0,
        -1.28,
        f"Setup/runtime errors: {error_cases}",
        ha="center",
        va="center",
        fontsize=10,
        color=KIT_RED if error_cases else DARK,
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT_DIR / f"runtime_error_zero_pie.{ext}",
            bbox_inches="tight",
            facecolor="white",
            dpi=300,
        )

    plt.close(fig)

    summary = {
        "total_evaluated_cases": total_cases,
        "processed_without_setup_or_runtime_error": stable_cases,
        "setup_or_runtime_errors": error_cases,
        "source_root": str(RESULTS_ROOT),
        "source_files": source_files,
        "interpretation": (
            "Incomplete no_progress rollouts are counted as evaluated cases, "
            "not as setup/runtime errors."
        ),
    }

    (OUT_DIR / "runtime_error_zero_pie_source_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    readme = (
        "Runtime performance / stability pie chart\n\n"
        "This folder contains the source script and generated figure for the "
        "runtime stability summary used in the code-quality supplementary material.\n\n"
        "The pie chart summarizes whether the external evaluation cases were "
        "processed without setup/runtime errors. Incomplete no_progress rollouts "
        "are not counted as runtime errors; they are valid evaluated cases under "
        "the current abstraction.\n\n"
        "Generated files:\n"
        "- runtime_error_zero_pie.png\n"
        "- runtime_error_zero_pie.pdf\n"
        "- runtime_error_zero_pie.svg\n"
        "- runtime_error_zero_pie_source_summary.json\n"
        "- plot_runtime_error_pie.py\n"
    )

    (OUT_DIR / "README_source.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
