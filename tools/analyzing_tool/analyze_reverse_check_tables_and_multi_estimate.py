from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

COMPARE_DIR = (
    Path(__file__)
    .resolve()
    .parents[3]
    .joinpath("private/analysis/external_cases/twopart_reverse_check_compare")
)

DETAIL_CSV = COMPARE_DIR / "three_stage_comparison_details.csv"

MULTI_RESULT_DIR = (
    Path(__file__).resolve().parents[3].joinpath("private/results/multi_assembly")
)

OUT_DIR = COMPARE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def is_report_success(report: dict[str, Any]) -> bool:
    success = report.get("success", None)
    if isinstance(success, bool):
        return success

    try:
        completion = float(report.get("completion", 0.0))
        return completion >= 0.999
    except Exception:
        return False


def load_latest_jsonl_reports(result_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load latest JSONL entry per assembly_name from a result directory.
    """
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)

    jsonl_files = sorted(result_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {result_dir}")

    latest: dict[str, dict[str, Any]] = {}

    for path in jsonl_files:
        mtime = path.stat().st_mtime

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                report = json.loads(line)
                name = str(report.get("assembly_name", ""))
                if not name:
                    continue

                prev = latest.get(name)
                if prev is None or mtime >= float(prev.get("_source_mtime", -1)):
                    report["_source_jsonl"] = str(path)
                    report["_source_mtime"] = mtime
                    latest[name] = report

    return latest


def get_debug(attempt: dict[str, Any]) -> dict[str, Any]:
    debug = attempt.get("debug", {})
    return debug if isinstance(debug, dict) else {}


def get_operation_data(attempt: dict[str, Any]) -> dict[str, Any]:
    debug = get_debug(attempt)
    data = debug.get("operation_data", {})
    return data if isinstance(data, dict) else {}


def attempt_is_failure(attempt: dict[str, Any]) -> bool:
    value = attempt.get("success", False)
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() not in {"true", "1", "yes", "success"}


def attempt_target(attempt: dict[str, Any]) -> str:
    debug = get_debug(attempt)
    return str(debug.get("target_part") or attempt.get("target") or "")


def candidate_failure_has_path_blocking(cand: dict[str, Any]) -> bool:
    if cand.get("path_ray_bundle_reason") == "blocking_hit":
        return True
    if cand.get("feasibility_reason") == "removal_path_projected_footprint_hit":
        return True
    if cand.get("first_blocking_geometry") is not None:
        return True
    return False


def attempt_matches_reverse_check_pattern(attempt: dict[str, Any]) -> bool:
    """
    Match the diagnostic pattern targeted by the reverse-check prototype.

    This is intentionally conservative: it focuses on attempts where a candidate
    direction existed but no feasible full pull-out operation was accepted, often
    because of path-level blockage.
    """
    debug = get_debug(attempt)
    op_data = get_operation_data(attempt)

    exec_reason = str(debug.get("exec_fail_reason", "") or "")
    failure_label = str(debug.get("failure_label", "") or "")
    mask_reason = str(debug.get("mask_reason", "") or "")
    why = str(debug.get("why", "") or "")

    if exec_reason == "no_feasible_dof_candidate":
        return True

    if "no_feasible_dof_candidate" in failure_label:
        return True

    # Optional broader match: mask-level geometric infeasibility.
    # This is weaker than exec-gate failure, so keep it conservative.
    if mask_reason == "geometrically_infeasible" and why in {
        "path_blocked",
        "blocked_path",
    }:
        return True

    cands = op_data.get("candidate_failures", [])
    if isinstance(cands, list):
        for cand in cands:
            if isinstance(cand, dict) and candidate_failure_has_path_blocking(cand):
                return True

    return False


def final_failure_frontier(report: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Approximate final no-progress frontier:
    latest failed attempt for each remaining part.

    If remaining_parts are not available, fall back to all failed attempts.
    """
    attempts = report.get("attempts", [])
    if not isinstance(attempts, list):
        return []

    remaining = report.get("remaining_parts", [])
    remaining_set = (
        {str(x) for x in remaining} if isinstance(remaining, list) else set()
    )

    selected_by_target: dict[str, dict[str, Any]] = {}

    for attempt in attempts:
        if not isinstance(attempt, dict) or not attempt_is_failure(attempt):
            continue

        target = attempt_target(attempt)

        if remaining_set and target not in remaining_set:
            continue

        selected_by_target[target] = attempt

    if selected_by_target:
        return list(selected_by_target.values())

    return [
        attempt
        for attempt in attempts
        if isinstance(attempt, dict) and attempt_is_failure(attempt)
    ]


def report_has_reverse_check_pattern(report: dict[str, Any]) -> bool:
    frontier = final_failure_frontier(report)
    return any(attempt_matches_reverse_check_pattern(a) for a in frontier)


def row_has_reverse_check_pattern(row: dict[str, str], prefix: str = "base") -> bool:
    """
    Use the detailed CSV from the three-stage comparison.
    """
    exec_reason = row.get(f"{prefix}_exec", "")
    reason = row.get(f"{prefix}_reason", "")
    mask = row.get(f"{prefix}_mask", "")

    text = " ".join([exec_reason, reason, mask])

    if "no_feasible_dof_candidate" in text:
        return True
    if "removal_path_projected_footprint_hit" in text:
        return True
    if "blocking_hit" in text:
        return True
    if "path_blocked" in text:
        return True

    return False


def pattern_meaning(pattern: str) -> str:
    meanings = {
        "SSS": "successful in all stages",
        "FFF": "incomplete in all stages",
        "FFS": "solved only after reverse-check variant",
        "FSF": "solved in intermediate stage but failed after reverse check",
        "FSS": "baseline failure solved after implementation changes",
        "SFF": "baseline success regressed after later changes",
        "SFS": "baseline and reverse success, intermediate failure",
        "SSF": "success before reverse check, failed after reverse check",
    }
    return meanings.get(pattern, "other transition pattern")


# ---------------------------------------------------------------------
# Stage and pattern tables
# ---------------------------------------------------------------------


def make_stage_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    stages = [
        ("Baseline", "base_status"),
        ("Intermediate", "mid_status"),
        ("Reverse-check variant", "rev_status"),
    ]

    out = []

    for stage_name, col in stages:
        total = len(rows)
        completed = sum(1 for r in rows if r.get(col) == "S")
        incomplete = total - completed

        out.append(
            {
                "stage": stage_name,
                "total": total,
                "completed": completed,
                "incomplete": incomplete,
                "success_rate": pct(completed / total if total else 0.0),
            }
        )

    return out


def make_pattern_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counter = Counter(r.get("pattern", "") for r in rows)
    total = len(rows)

    out = []
    for pattern, count in sorted(counter.items()):
        out.append(
            {
                "pattern": pattern,
                "meaning": pattern_meaning(pattern),
                "count": count,
                "share": pct(count / total if total else 0.0),
            }
        )

    return out


# ---------------------------------------------------------------------
# Multi-assembly estimate
# ---------------------------------------------------------------------


def make_multi_estimate(rows: list[dict[str, str]]) -> dict[str, Any]:
    """
    Estimate possible reverse-check impact on multi_assembly success rate.

    The transfer factor is derived from joint cases with a matching diagnostic
    pattern in the baseline stage:
        p_reverse = newly_solved_matching_joint_cases / matching_baseline_joint_failures

    This factor is then applied to incomplete multi_assembly cases whose final
    failure frontier contains the same diagnostic pattern.

    This is not a measured multi result. It is a conservative pattern-based
    estimate for discussion.
    """
    base_fail_matching = [
        r
        for r in rows
        if r.get("base_status") == "F" and row_has_reverse_check_pattern(r, "base")
    ]

    newly_solved_matching = [
        r for r in base_fail_matching if r.get("rev_status") == "S"
    ]

    if base_fail_matching:
        transfer_factor = len(newly_solved_matching) / len(base_fail_matching)
    else:
        transfer_factor = 0.0

    multi_reports = load_latest_jsonl_reports(MULTI_RESULT_DIR)

    multi_total = len(multi_reports)
    multi_success = sum(1 for r in multi_reports.values() if is_report_success(r))
    multi_failed = multi_total - multi_success

    matching_failed_names = []

    for name, report in multi_reports.items():
        if is_report_success(report):
            continue

        if report_has_reverse_check_pattern(report):
            matching_failed_names.append(name)

    matching_failed_names = sorted(matching_failed_names)
    matching_failed = len(matching_failed_names)

    estimated_new_successes = transfer_factor * matching_failed
    estimated_success_after = multi_success + estimated_new_successes
    estimated_success_rate_after = (
        estimated_success_after / multi_total if multi_total else 0.0
    )

    # Upper bound: every matching failed multi case becomes solvable.
    upper_success_after = multi_success + matching_failed
    upper_success_rate_after = upper_success_after / multi_total if multi_total else 0.0

    out = {
        "joint_matching_baseline_failures": len(base_fail_matching),
        "joint_matching_failures_solved_by_reverse": len(newly_solved_matching),
        "joint_transfer_factor": transfer_factor,
        "multi_total": multi_total,
        "multi_observed_success": multi_success,
        "multi_observed_failed": multi_failed,
        "multi_observed_success_rate": multi_success / multi_total
        if multi_total
        else 0.0,
        "multi_matching_failed_cases": matching_failed,
        "estimated_new_multi_successes": estimated_new_successes,
        "estimated_multi_success_after_reverse": estimated_success_after,
        "estimated_multi_success_rate_after_reverse": estimated_success_rate_after,
        "upper_bound_success_after_reverse": upper_success_after,
        "upper_bound_success_rate_after_reverse": upper_success_rate_after,
        "matching_failed_names": matching_failed_names,
    }

    return out


def write_markdown_report(
    stage_rows: list[dict[str, Any]],
    pattern_rows: list[dict[str, Any]],
    estimate: dict[str, Any],
    out_path: Path,
) -> None:
    lines = []

    lines.append("# Reverse-check analysis tables\n")

    lines.append("## Table 1. Three-stage comparison\n")
    lines.append("| Stage | Total | Completed | Incomplete | Success rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in stage_rows:
        lines.append(
            f"| {r['stage']} | {r['total']} | {r['completed']} | "
            f"{r['incomplete']} | {r['success_rate']} |"
        )

    lines.append("\n## Table 2. Case-status transition patterns\n")
    lines.append("| Pattern | Meaning | Count | Share |")
    lines.append("|---|---|---:|---:|")
    for r in pattern_rows:
        lines.append(
            f"| {r['pattern']} | {r['meaning']} | {r['count']} | {r['share']} |"
        )

    lines.append("\n## Multi-assembly success-rate estimate\n")
    lines.append(
        f"- Joint baseline failures with matching diagnostic pattern: "
        f"{estimate['joint_matching_baseline_failures']}"
    )
    lines.append(
        f"- Matching joint failures solved by reverse check: "
        f"{estimate['joint_matching_failures_solved_by_reverse']}"
    )
    lines.append(
        f"- Transfer factor: {estimate['joint_transfer_factor']:.3f} "
        f"({pct(estimate['joint_transfer_factor'])})"
    )
    lines.append(
        f"- Observed multi success: {estimate['multi_observed_success']} / "
        f"{estimate['multi_total']} "
        f"({pct(estimate['multi_observed_success_rate'])})"
    )
    lines.append(
        f"- Incomplete multi cases with matching final-frontier diagnostic: "
        f"{estimate['multi_matching_failed_cases']}"
    )
    lines.append(
        f"- Estimated additional multi successes: "
        f"{estimate['estimated_new_multi_successes']:.1f}"
    )
    lines.append(
        f"- Estimated multi success after reverse check: "
        f"{estimate['estimated_multi_success_after_reverse']:.1f} / "
        f"{estimate['multi_total']} "
        f"({pct(estimate['estimated_multi_success_rate_after_reverse'])})"
    )
    lines.append(
        f"- Upper bound if all matching multi cases were solved: "
        f"{estimate['upper_bound_success_after_reverse']} / "
        f"{estimate['multi_total']} "
        f"({pct(estimate['upper_bound_success_rate_after_reverse'])})"
    )

    lines.append(
        "\nNote: The multi-assembly result is an estimate, not a measured "
        "reverse-check evaluation. It applies the joint-case transfer factor "
        "only to incomplete multi cases whose final failure frontier contains "
        "the same diagnostic pattern."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_csv_rows(DETAIL_CSV)

    stage_rows = make_stage_summary(rows)
    pattern_rows = make_pattern_summary(rows)
    estimate = make_multi_estimate(rows)

    stage_csv = OUT_DIR / "reverse_check_stage_summary.csv"
    pattern_csv = OUT_DIR / "reverse_check_pattern_summary.csv"
    estimate_csv = OUT_DIR / "reverse_check_multi_estimate.csv"
    md_path = OUT_DIR / "reverse_check_tables_and_multi_estimate.md"
    matching_names_path = OUT_DIR / "multi_matching_reverse_check_pattern_cases.txt"

    write_csv(
        stage_csv,
        stage_rows,
        ["stage", "total", "completed", "incomplete", "success_rate"],
    )

    write_csv(
        pattern_csv,
        pattern_rows,
        ["pattern", "meaning", "count", "share"],
    )

    estimate_rows = [
        {
            key: (",".join(value) if isinstance(value, list) else value)
            for key, value in estimate.items()
            if key != "matching_failed_names"
        }
    ]

    write_csv(
        estimate_csv,
        estimate_rows,
        [
            "joint_matching_baseline_failures",
            "joint_matching_failures_solved_by_reverse",
            "joint_transfer_factor",
            "multi_total",
            "multi_observed_success",
            "multi_observed_failed",
            "multi_observed_success_rate",
            "multi_matching_failed_cases",
            "estimated_new_multi_successes",
            "estimated_multi_success_after_reverse",
            "estimated_multi_success_rate_after_reverse",
            "upper_bound_success_after_reverse",
            "upper_bound_success_rate_after_reverse",
        ],
    )

    matching_names_path.write_text(
        "\n".join(estimate["matching_failed_names"]) + "\n",
        encoding="utf-8",
    )

    write_markdown_report(stage_rows, pattern_rows, estimate, md_path)

    print(f"[SAVED] {stage_csv}")
    print(f"[SAVED] {pattern_csv}")
    print(f"[SAVED] {estimate_csv}")
    print(f"[SAVED] {matching_names_path}")
    print(f"[SAVED] {md_path}")


if __name__ == "__main__":
    main()
