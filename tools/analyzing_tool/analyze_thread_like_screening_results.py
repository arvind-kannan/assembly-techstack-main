#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze thread-like / rotational-locking screening outputs.

This tool is intended as a companion to screen_thread_like_multi_cases.py.
It reads one or more screening output folders containing files such as:

    thread_like_summary.json
    thread_like_cases.csv
    thread_like_pairs.csv
    thread_like_manifest_for_rendering.csv

and produces a compact analysis package:

    analysis_report.md
    combined_case_summary.csv
    combined_pair_summary.csv
    manifest_sampling_plan.csv
    optional: manual_validation_summary.csv

Typical usage
-------------

Analyze the two joint validation folders together:

cd <repo>

PYTHONPATH=src:$PYTHONPATH \
python tools/analyzing_tool/analyze_thread_like_screening_results.py \
  --input-dirs \
    ../private/results/thread_like_screening_joint/joint_assembly \
    ../private/results/thread_like_screening_joint/joint_assembly_rotation \
  --out ../private/results/thread_like_screening_joint/analysis \
  --sample-per-confidence 30

Analyze the multi-assembly screening folder:

python tools/analyzing_tool/analyze_thread_like_screening_results.py \
  --input-dirs ../private/results/thread_like_screening \
  --out ../private/results/thread_like_screening/analysis \
  --sample-per-confidence 30

Optional manual validation
--------------------------

After visually checking overview images, create a CSV such as:

    assembly_name,target_part,confidence,manual_label
    multi_assembly_00032,part_14,high,TP-clear
    multi_assembly_00070,part_3,likely,FP

Accepted manual labels:
    TP-clear, TP-weak, FP, unclear

Then run:

python tools/analyzing_tool/analyze_thread_like_screening_results.py \
  --input-dirs ... \
  --manual-validation manual_validation.csv \
  --out ...
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONF_RANK = {"none": 0, "possible": 1, "likely": 2, "high": 3}
CONF_ORDER = ["possible", "likely", "high"]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_conf(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"high", "likely", "possible"}:
        return s
    return "none"


def conf_rank(value: Any) -> int:
    return CONF_RANK.get(normalize_conf(value), 0)


def safe_rate(num: int | float, den: int | float) -> float:
    den = float(den)
    if den <= 0:
        return 0.0
    return float(num) / den


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out) + "\n"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    name: str
    input_dir: Path
    summary: dict[str, Any]
    cases: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    manifest: list[dict[str, Any]]


def dataset_name_from_dir(path: Path) -> str:
    name = path.name
    if name in {"joint_assembly", "joint_assembly_rotation", "multi_assembly"}:
        return name
    # Fall back to parent/name for clarity when input dirs are named "analysis" etc.
    return name


def load_bundle(input_dir: Path) -> DatasetBundle:
    input_dir = input_dir.expanduser().resolve()
    summary = read_json(input_dir / "thread_like_summary.json")
    cases = read_csv_rows(input_dir / "thread_like_cases.csv")
    pairs = read_csv_rows(input_dir / "thread_like_pairs.csv")
    manifest = read_csv_rows(input_dir / "thread_like_manifest_for_rendering.csv")

    name = str(
        summary.get("group") or summary.get("dataset") or ""
    ) or dataset_name_from_dir(input_dir)

    for row in cases:
        row["_dataset"] = name
        row["_input_dir"] = str(input_dir)
    for row in pairs:
        row["_dataset"] = name
        row["_input_dir"] = str(input_dir)
    for row in manifest:
        row["_dataset"] = name
        row["_input_dir"] = str(input_dir)

    return DatasetBundle(
        name=name,
        input_dir=input_dir,
        summary=summary,
        cases=cases,
        pairs=pairs,
        manifest=manifest,
    )


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("_dataset", "")), str(row.get("assembly_name", "")))


def target_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("_dataset", "")),
        str(row.get("assembly_name", "")),
        str(row.get("target_part", "")),
    )


def combine_cases(bundles: list[DatasetBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for b in bundles:
        for row in b.cases:
            key = case_key(row)
            if key in seen:
                continue
            seen.add(key)

            out = dict(row)
            out["max_confidence_norm"] = normalize_conf(
                out.get("max_confidence") or out.get("confidence")
            )
            out["confidence_rank"] = conf_rank(out["max_confidence_norm"])
            out["completion_float"] = parse_float(out.get("completion"), None)
            out["failure_relevant_bool"] = parse_bool(
                out.get("failure_relevant_thread_like")
            )
            out["num_detected_pairs_int"] = parse_int(out.get("num_detected_pairs"), 0)
            out["num_high_confidence_pairs_int"] = parse_int(
                out.get("num_high_confidence_pairs"), 0
            )
            out["num_likely_or_high_pairs_int"] = parse_int(
                out.get("num_likely_or_high_pairs"), 0
            )
            rows.append(out)

    return rows


def combine_pairs(bundles: list[DatasetBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b in bundles:
        for row in b.pairs:
            out = dict(row)
            out["confidence_norm"] = normalize_conf(
                out.get("confidence") or out.get("max_confidence")
            )
            out["confidence_rank"] = conf_rank(out["confidence_norm"])
            out["score_float"] = parse_float(
                out.get("score") or out.get("thread_like_score"), 0.0
            )
            rows.append(out)
    return rows


def combine_manifest(bundles: list[DatasetBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for b in bundles:
        for row in b.manifest:
            key = target_key(row)
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["confidence_norm"] = normalize_conf(out.get("confidence"))
            out["confidence_rank"] = conf_rank(out["confidence_norm"])
            out["score_float"] = parse_float(out.get("score"), 0.0)
            rows.append(out)
    return rows


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    by_dataset = Counter(str(r.get("_dataset", "")) for r in cases)
    by_conf = Counter(str(r.get("max_confidence_norm", "none")) for r in cases)

    possible_or_above = sum(1 for r in cases if int(r.get("confidence_rank", 0)) >= 1)
    likely_or_above = sum(1 for r in cases if int(r.get("confidence_rank", 0)) >= 2)
    high = sum(1 for r in cases if int(r.get("confidence_rank", 0)) >= 3)

    failure_relevant_possible = sum(
        1
        for r in cases
        if parse_bool(r.get("failure_relevant_bool"))
        and int(r.get("confidence_rank", 0)) >= 1
    )
    failure_relevant_likely = sum(
        1
        for r in cases
        if parse_bool(r.get("failure_relevant_bool"))
        and int(r.get("confidence_rank", 0)) >= 2
    )
    failure_relevant_high = sum(
        1
        for r in cases
        if parse_bool(r.get("failure_relevant_bool"))
        and int(r.get("confidence_rank", 0)) >= 3
    )

    completions = [
        parse_float(r.get("completion_float"), None)
        for r in cases
        if parse_float(r.get("completion_float"), None) is not None
    ]
    mean_completion = sum(completions) / len(completions) if completions else None

    return {
        "total_cases_in_case_csv": total,
        "cases_by_dataset": dict(by_dataset),
        "cases_by_max_confidence": dict(by_conf),
        "possible_or_above_cases": possible_or_above,
        "likely_or_above_cases": likely_or_above,
        "high_confidence_cases": high,
        "failure_relevant_possible_or_above_cases": failure_relevant_possible,
        "failure_relevant_likely_or_above_cases": failure_relevant_likely,
        "failure_relevant_high_confidence_cases": failure_relevant_high,
        "mean_completion_for_rows_with_completion": mean_completion,
    }


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(r.get("_dataset", "")) for r in pairs)
    by_conf = Counter(str(r.get("confidence_norm", "none")) for r in pairs)
    by_relation = Counter(
        str(r.get("relation", "")) for r in pairs if str(r.get("relation", ""))
    )
    by_relation_conf = Counter(
        (str(r.get("relation", "")), str(r.get("confidence_norm", "none")))
        for r in pairs
    )

    return {
        "total_pair_candidates": len(pairs),
        "pairs_by_dataset": dict(by_dataset),
        "pairs_by_confidence": dict(by_conf),
        "top_relations": by_relation.most_common(20),
        "relation_confidence_counts": [
            {"relation": k[0], "confidence": k[1], "count": v}
            for k, v in by_relation_conf.most_common()
        ],
    }


def make_sampling_plan(
    manifest: list[dict[str, Any]],
    sample_per_confidence: int,
    max_per_case: int,
) -> list[dict[str, Any]]:
    """
    Create a balanced visual-validation sampling plan.

    The plan samples high, likely, and possible candidates separately, prefers
    higher score, and limits the number of targets per case.
    """
    plan: list[dict[str, Any]] = []

    for conf in ["high", "likely", "possible"]:
        subset = [
            r
            for r in manifest
            if normalize_conf(r.get("confidence_norm") or r.get("confidence")) == conf
        ]
        subset.sort(
            key=lambda r: (
                parse_float(r.get("score_float") or r.get("score"), 0.0) or 0.0,
                str(r.get("assembly_name", "")),
                str(r.get("target_part", "")),
            ),
            reverse=True,
        )

        case_counts: dict[tuple[str, str], int] = defaultdict(int)
        added = 0
        for row in subset:
            key = (str(row.get("_dataset", "")), str(row.get("assembly_name", "")))
            if max_per_case > 0 and case_counts[key] >= max_per_case:
                continue

            out = dict(row)
            out["sampling_confidence_group"] = conf
            out["manual_label"] = ""
            out["manual_note"] = ""
            plan.append(out)

            case_counts[key] += 1
            added += 1
            if added >= sample_per_confidence:
                break

    return plan


def top_cases_by_pair_count(
    cases: list[dict[str, Any]], n: int = 20
) -> list[dict[str, Any]]:
    rows = sorted(
        cases,
        key=lambda r: (
            parse_int(
                r.get("num_detected_pairs_int") or r.get("num_detected_pairs"), 0
            ),
            parse_int(
                r.get("num_high_confidence_pairs_int")
                or r.get("num_high_confidence_pairs"),
                0,
            ),
        ),
        reverse=True,
    )
    return rows[:n]


def top_cases_by_confidence(
    cases: list[dict[str, Any]], n: int = 20
) -> list[dict[str, Any]]:
    rows = sorted(
        cases,
        key=lambda r: (
            int(r.get("confidence_rank", 0)),
            parse_float(r.get("max_thread_like_score") or r.get("max_score"), 0.0)
            or 0.0,
            parse_int(
                r.get("num_high_confidence_pairs_int")
                or r.get("num_high_confidence_pairs"),
                0,
            ),
        ),
        reverse=True,
    )
    return rows[:n]


# -----------------------------------------------------------------------------
# Manual validation
# -----------------------------------------------------------------------------


def load_manual_validation(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    return read_csv_rows(path.expanduser().resolve())


def normalize_manual_label(label: Any) -> str:
    s = str(label or "").strip().lower()
    mapping = {
        "tp-clear": "TP-clear",
        "tp_clear": "TP-clear",
        "tp": "TP-clear",
        "true": "TP-clear",
        "yes": "TP-clear",
        "1": "TP-clear",
        "tp-weak": "TP-weak",
        "tp_weak": "TP-weak",
        "weak": "TP-weak",
        "fp": "FP",
        "false": "FP",
        "no": "FP",
        "0": "FP",
        "unclear": "unclear",
        "unknown": "unclear",
        "?": "unclear",
    }
    return mapping.get(s, str(label or "").strip())


def summarize_manual_validation(
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {}, []

    normalized: list[dict[str, Any]] = []
    for r in rows:
        out = dict(r)
        out["confidence_norm"] = normalize_conf(
            r.get("confidence") or r.get("max_confidence")
        )
        out["manual_label_norm"] = normalize_manual_label(
            r.get("manual_label") or r.get("label") or r.get("validation")
        )
        normalized.append(out)

    by_conf: dict[str, Counter] = {c: Counter() for c in CONF_ORDER}
    overall = Counter()

    for r in normalized:
        conf = r["confidence_norm"]
        label = r["manual_label_norm"]
        if conf in by_conf:
            by_conf[conf][label] += 1
        overall[label] += 1

    rows_out: list[dict[str, Any]] = []
    for conf in CONF_ORDER:
        counter = by_conf[conf]
        total_known = sum(counter.values())
        tp_clear = counter.get("TP-clear", 0)
        tp_weak = counter.get("TP-weak", 0)
        fp = counter.get("FP", 0)
        unclear = counter.get("unclear", 0)
        precision_strict = safe_rate(tp_clear, total_known - unclear)
        precision_broad = safe_rate(tp_clear + tp_weak, total_known - unclear)

        rows_out.append(
            {
                "confidence": conf,
                "validated_total": total_known,
                "TP_clear": tp_clear,
                "TP_weak": tp_weak,
                "FP": fp,
                "unclear": unclear,
                "strict_precision_excluding_unclear": precision_strict,
                "broad_precision_excluding_unclear": precision_broad,
            }
        )

    summary = {
        "manual_validation_total_rows": len(normalized),
        "manual_labels_overall": dict(overall),
        "by_confidence": rows_out,
    }
    return summary, rows_out


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def build_report(
    bundles: list[DatasetBundle],
    cases: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    case_summary: dict[str, Any],
    pair_summary: dict[str, Any],
    sampling_plan: list[dict[str, Any]],
    manual_summary: dict[str, Any],
    manual_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []

    lines.append("# Thread-like Screening Analysis Report\n")
    lines.append(
        "This report summarizes geometry-based screening outputs for thread-like, bolt-nut-like, or rotational-locking structures.\n"
    )
    lines.append(
        "The screening result should be interpreted as an estimate of potentially affected cases, not as a predicted performance result of a future screw-aware implementation.\n"
    )

    lines.append("## Input folders\n")
    input_rows = []
    for b in bundles:
        input_rows.append(
            [
                b.name,
                b.input_dir,
                len(b.cases),
                len(b.pairs),
                len(b.manifest),
            ]
        )
    lines.append(
        md_table(
            ["Dataset", "Input directory", "Case rows", "Pair rows", "Manifest rows"],
            input_rows,
        )
    )

    lines.append("## Summary JSON values\n")
    summary_rows = []
    keys_of_interest = [
        "total_latest_multi_assembly_reports",
        "baseline_successful_rollouts",
        "baseline_incomplete_rollouts",
        "baseline_completion_rate",
        "processed_cases",
        "detected_thread_like_cases_possible_or_above",
        "detected_thread_like_cases_likely_or_above",
        "detected_thread_like_cases_high_confidence",
        "failure_relevant_thread_like_cases_possible_or_above",
        "failure_relevant_thread_like_cases_likely_or_above",
        "failure_relevant_thread_like_cases_high_confidence",
        "upper_completion_rate_if_possible_failure_relevant_cases_solved",
        "upper_completion_rate_if_likely_failure_relevant_cases_solved",
        "upper_completion_rate_if_high_confidence_failure_relevant_cases_solved",
        "num_pair_candidates",
    ]
    for b in bundles:
        for key in keys_of_interest:
            if key in b.summary:
                value = b.summary.get(key)
                if isinstance(value, float) and "rate" in key:
                    value = f"{value:.4f} ({pct(value)})"
                summary_rows.append([b.name, key, value])
    lines.append(md_table(["Dataset", "Metric", "Value"], summary_rows))

    lines.append("## Case-level summary\n")
    case_rows = [
        ["Total case rows", case_summary.get("total_cases_in_case_csv", 0)],
        ["Possible or above", case_summary.get("possible_or_above_cases", 0)],
        ["Likely or above", case_summary.get("likely_or_above_cases", 0)],
        ["High confidence", case_summary.get("high_confidence_cases", 0)],
        [
            "Failure-relevant possible or above",
            case_summary.get("failure_relevant_possible_or_above_cases", 0),
        ],
        [
            "Failure-relevant likely or above",
            case_summary.get("failure_relevant_likely_or_above_cases", 0),
        ],
        [
            "Failure-relevant high confidence",
            case_summary.get("failure_relevant_high_confidence_cases", 0),
        ],
    ]
    lines.append(md_table(["Metric", "Value"], case_rows))

    by_dataset_rows = [
        [k, v] for k, v in sorted(case_summary.get("cases_by_dataset", {}).items())
    ]
    lines.append("### Cases by dataset\n")
    lines.append(md_table(["Dataset", "Cases"], by_dataset_rows))

    by_conf_rows = [
        [k, v]
        for k, v in sorted(
            case_summary.get("cases_by_max_confidence", {}).items(),
            key=lambda x: CONF_RANK.get(x[0], 0),
        )
    ]
    lines.append("### Cases by maximum confidence\n")
    lines.append(md_table(["Maximum confidence", "Cases"], by_conf_rows))

    lines.append("## Pair-level summary\n")
    pair_rows = [
        ["Total pair candidates", pair_summary.get("total_pair_candidates", 0)],
    ]
    lines.append(md_table(["Metric", "Value"], pair_rows))

    pair_conf_rows = [
        [k, v]
        for k, v in sorted(
            pair_summary.get("pairs_by_confidence", {}).items(),
            key=lambda x: CONF_RANK.get(x[0], 0),
        )
    ]
    lines.append("### Pairs by confidence\n")
    lines.append(md_table(["Confidence", "Pairs"], pair_conf_rows))

    relation_rows = [
        [rel, count] for rel, count in pair_summary.get("top_relations", [])[:15]
    ]
    lines.append("### Top detected relation types\n")
    lines.append(md_table(["Relation", "Pair count"], relation_rows))

    lines.append("## Top cases for visual inspection\n")
    top_conf = top_cases_by_confidence(cases, 15)
    top_conf_rows = [
        [
            r.get("_dataset", ""),
            r.get("assembly_name", ""),
            r.get("max_confidence_norm", ""),
            r.get("max_thread_like_score", r.get("max_score", "")),
            r.get("num_detected_pairs", ""),
            r.get("num_high_confidence_pairs", ""),
            r.get("failure_relevant_thread_like", ""),
            r.get("completion", ""),
        ]
        for r in top_conf
    ]
    lines.append(
        md_table(
            [
                "Dataset",
                "Assembly",
                "Max confidence",
                "Score",
                "Pairs",
                "High pairs",
                "Failure relevant",
                "Completion",
            ],
            top_conf_rows,
        )
    )

    lines.append("## Balanced manual-validation sampling plan\n")
    lines.append(
        f"The generated sampling plan contains {len(sampling_plan)} rows. "
        "It is balanced by confidence level and limits repeated targets from the same case.\n"
    )
    sample_preview = sampling_plan[:20]
    sample_rows = [
        [
            r.get("_dataset", ""),
            r.get("assembly_name", ""),
            r.get("target_part", ""),
            r.get("confidence", r.get("confidence_norm", "")),
            r.get("score", r.get("score_float", "")),
            r.get("relation", ""),
            r.get("paired_part", ""),
        ]
        for r in sample_preview
    ]
    lines.append(
        md_table(
            [
                "Dataset",
                "Assembly",
                "Target",
                "Confidence",
                "Score",
                "Relation",
                "Paired part",
            ],
            sample_rows,
        )
    )

    if manual_summary:
        lines.append("## Manual validation summary\n")
        mv_rows = []
        for r in manual_rows:
            mv_rows.append(
                [
                    r["confidence"],
                    r["validated_total"],
                    r["TP_clear"],
                    r["TP_weak"],
                    r["FP"],
                    r["unclear"],
                    pct(float(r["strict_precision_excluding_unclear"])),
                    pct(float(r["broad_precision_excluding_unclear"])),
                ]
            )
        lines.append(
            md_table(
                [
                    "Confidence",
                    "Validated",
                    "TP-clear",
                    "TP-weak",
                    "FP",
                    "Unclear",
                    "Strict precision",
                    "Broad precision",
                ],
                mv_rows,
            )
        )
        lines.append(
            "Strict precision counts only `TP-clear` as true positive. "
            "Broad precision counts both `TP-clear` and `TP-weak` as true positives. "
            "Unclear samples are excluded from the denominator.\n"
        )
    else:
        lines.append("## Manual validation summary\n")
        lines.append(
            "No manual validation CSV was provided. After inspecting rendered overview images, "
            "fill `manual_label` in `manifest_sampling_plan.csv` and rerun this tool with "
            "`--manual-validation <csv>`.\n"
        )

    lines.append("## Suggested thesis wording\n")
    lines.append(
        "A geometry-based screening was used to identify thread-like, bolt-nut-like, "
        "or rotational-locking structures that may require future thread-aware, "
        "rotational-release, or dependency-aware capabilities. The screening does not "
        "claim that all detected cases contain physically accurate screw threads. "
        "Instead, it estimates how many cases may be affected by geometric structures "
        "that are not represented by the current translational manipulation abstraction.\n"
    )

    lines.append(
        "For conservative reporting, use the high-confidence case count as the lower-risk "
        "estimate and the likely-or-above count as a broader upper estimate. The possible "
        "category should be treated primarily as a visual-inspection pool, not as a final "
        "claim about thread-like structure frequency.\n"
    )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze outputs from thread-like geometry screening."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more screening output folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output folder for analysis report and combined CSVs.",
    )
    parser.add_argument(
        "--sample-per-confidence",
        type=int,
        default=30,
        help="Number of manifest rows to include per confidence level in the manual-validation sampling plan.",
    )
    parser.add_argument(
        "--max-per-case",
        type=int,
        default=1,
        help="Maximum number of sampling-plan rows per case within each confidence level.",
    )
    parser.add_argument(
        "--manual-validation",
        type=Path,
        default=None,
        help="Optional manually labeled CSV for precision estimation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_bundle(p) for p in args.input_dirs]

    cases = combine_cases(bundles)
    pairs = combine_pairs(bundles)
    manifest = combine_manifest(bundles)

    case_summary = summarize_cases(cases)
    pair_summary = summarize_pairs(pairs)

    sampling_plan = make_sampling_plan(
        manifest=manifest,
        sample_per_confidence=max(args.sample_per_confidence, 0),
        max_per_case=max(args.max_per_case, 0),
    )

    manual_rows_raw = load_manual_validation(args.manual_validation)
    manual_summary, manual_rows = summarize_manual_validation(manual_rows_raw)

    # Write machine-readable summaries.
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "input_dirs": [str(p.expanduser().resolve()) for p in args.input_dirs],
                "case_summary": case_summary,
                "pair_summary": pair_summary,
                "manual_validation": manual_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Combined CSV exports.
    write_csv_rows(out_dir / "combined_case_summary.csv", cases)
    write_csv_rows(out_dir / "combined_pair_summary.csv", pairs)
    write_csv_rows(out_dir / "combined_manifest.csv", manifest)
    write_csv_rows(out_dir / "manifest_sampling_plan.csv", sampling_plan)

    if manual_rows:
        write_csv_rows(out_dir / "manual_validation_summary.csv", manual_rows)

    report = build_report(
        bundles=bundles,
        cases=cases,
        pairs=pairs,
        manifest=manifest,
        case_summary=case_summary,
        pair_summary=pair_summary,
        sampling_plan=sampling_plan,
        manual_summary=manual_summary,
        manual_rows=manual_rows,
    )
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")

    print("=" * 80)
    print(f"[DONE] Analysis written to: {out_dir}")
    print(f"[DONE] Report: {out_dir / 'analysis_report.md'}")
    print(f"[DONE] Summary JSON: {out_dir / 'analysis_summary.json'}")
    print(f"[DONE] Sampling plan: {out_dir / 'manifest_sampling_plan.csv'}")


if __name__ == "__main__":
    main()
