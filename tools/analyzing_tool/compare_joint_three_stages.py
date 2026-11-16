from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------


def discover_files(folder: Path, suffix: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    files = sorted(folder.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {folder}")

    return files


# def normalize_file_args(files: list[Path] | None, folder: Path, suffix: str) -> list[Path]:
#    if files:
#        return [p.expanduser() for p in files]
#    return discover_files(folder.expanduser(), suffix)
def normalize_file_args(
    files: list[Path] | None,
    folders: list[Path],
    suffix: str,
) -> list[Path]:
    if files:
        return [p.expanduser() for p in files]

    out: list[Path] = []
    for folder in folders:
        out.extend(discover_files(folder.expanduser(), suffix))

    return sorted(out)


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------


def get_case_name(row: dict) -> str | None:
    name = (
        row.get("assembly_name")
        or row.get("case")
        or row.get("case_name")
        or row.get("name")
    )
    if not name:
        return None
    return str(name)


def load_csv_summaries(paths: list[Path], stage_name: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found for stage {stage_name}: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = get_case_name(row)
                if not name:
                    continue

                if name in rows:
                    print(
                        f"[warn] duplicate case in {stage_name}: {name} "
                        f"already loaded; overwritten by {path}"
                    )

                row = dict(row)
                row["_source_csv"] = str(path)
                rows[name] = row

    return rows


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def is_success(row: dict | None) -> bool:
    if row is None:
        return False

    status = str(row.get("status", "")).lower()
    result = str(row.get("result", "")).lower()
    failure = str(row.get("failure_reason", "") or row.get("final_reason", "")).lower()

    completed = as_float(
        row.get("completion")
        or row.get("completed")
        or row.get("completion_ratio")
        or row.get("final_completion"),
        default=0.0,
    )

    if status in {"success", "succeeded"} or result in {"success", "succeeded"}:
        return True

    if status in {"fail", "failed", "error"} or result in {"fail", "failed", "error"}:
        return False

    if failure and failure not in {"none", "nan"}:
        return False

    return completed >= 0.999


def status_char(row: dict | None) -> str:
    return "S" if is_success(row) else "F"


# -----------------------------------------------------------------------------
# JSONL loading
# -----------------------------------------------------------------------------


def extract_attempt_info(row: dict) -> dict:
    attempts = row.get("attempts", []) or []
    last_attempt = attempts[-1] if attempts else {}
    debug = last_attempt.get("debug", {}) or {}

    final_reason = (
        row.get("failure_reason")
        or row.get("final_reason")
        or last_attempt.get("reason")
        or debug.get("exec_fail_reason")
        or debug.get("mask_reason")
        or ""
    )

    return {
        "final_reason": final_reason,
        "last_exec_fail_reason": debug.get("exec_fail_reason", ""),
        "last_mask_reason": debug.get("mask_reason", ""),
        "last_target": debug.get("target_part") or last_attempt.get("target") or "",
        "num_attempts": len(attempts),
        "last_action": last_attempt.get("action", ""),
    }


def load_jsonl_infos(paths: list[Path], stage_name: str) -> dict[str, dict]:
    info: dict[str, dict] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found for stage {stage_name}: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                name = str(row.get("assembly_name", ""))
                if not name:
                    continue

                if name in info:
                    print(
                        f"[warn] duplicate JSONL case in {stage_name}: {name} "
                        f"already loaded; overwritten by {path}"
                    )

                data = extract_attempt_info(row)
                data["_source_jsonl"] = str(path)
                info[name] = data

    return info


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------


def stage_counts(rows: dict[str, dict]) -> tuple[int, int, int]:
    total = len(rows)
    succ = sum(1 for row in rows.values() if is_success(row))
    fail = total - succ
    return total, succ, fail


def write_case_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def reason_key(info: dict | None) -> str:
    if not info:
        return ""

    final_reason = str(info.get("final_reason", "") or "")
    exec_reason = str(info.get("last_exec_fail_reason", "") or "")
    mask_reason = str(info.get("last_mask_reason", "") or "")

    if exec_reason:
        return f"exec:{exec_reason}"
    if mask_reason:
        return f"mask:{mask_reason}"
    if final_reason:
        return f"final:{final_reason}"
    return ""


def write_detailed_csv(
    path: Path,
    all_names: list[str],
    base_rows: dict[str, dict],
    mid_rows: dict[str, dict],
    rev_rows: dict[str, dict],
    base_info: dict[str, dict],
    mid_info: dict[str, dict],
    rev_info: dict[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "pattern",
        "base_status",
        "mid_status",
        "rev_status",
        "base_reason",
        "mid_reason",
        "rev_reason",
        "base_exec",
        "mid_exec",
        "rev_exec",
        "base_mask",
        "mid_mask",
        "rev_mask",
        "base_last_target",
        "mid_last_target",
        "rev_last_target",
        "base_attempts",
        "mid_attempts",
        "rev_attempts",
        "base_csv",
        "mid_csv",
        "rev_csv",
        "base_jsonl",
        "mid_jsonl",
        "rev_jsonl",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name in all_names:
            b_row = base_rows.get(name)
            m_row = mid_rows.get(name)
            r_row = rev_rows.get(name)

            b_info = base_info.get(name, {})
            m_info = mid_info.get(name, {})
            r_info = rev_info.get(name, {})

            pattern = f"{status_char(b_row)}{status_char(m_row)}{status_char(r_row)}"

            writer.writerow(
                {
                    "case": name,
                    "pattern": pattern,
                    "base_status": status_char(b_row),
                    "mid_status": status_char(m_row),
                    "rev_status": status_char(r_row),
                    "base_reason": b_info.get("final_reason", ""),
                    "mid_reason": m_info.get("final_reason", ""),
                    "rev_reason": r_info.get("final_reason", ""),
                    "base_exec": b_info.get("last_exec_fail_reason", ""),
                    "mid_exec": m_info.get("last_exec_fail_reason", ""),
                    "rev_exec": r_info.get("last_exec_fail_reason", ""),
                    "base_mask": b_info.get("last_mask_reason", ""),
                    "mid_mask": m_info.get("last_mask_reason", ""),
                    "rev_mask": r_info.get("last_mask_reason", ""),
                    "base_last_target": b_info.get("last_target", ""),
                    "mid_last_target": m_info.get("last_target", ""),
                    "rev_last_target": r_info.get("last_target", ""),
                    "base_attempts": b_info.get("num_attempts", ""),
                    "mid_attempts": m_info.get("num_attempts", ""),
                    "rev_attempts": r_info.get("num_attempts", ""),
                    "base_csv": b_row.get("_source_csv", "") if b_row else "",
                    "mid_csv": m_row.get("_source_csv", "") if m_row else "",
                    "rev_csv": r_row.get("_source_csv", "") if r_row else "",
                    "base_jsonl": b_info.get("_source_jsonl", ""),
                    "mid_jsonl": m_info.get("_source_jsonl", ""),
                    "rev_jsonl": r_info.get("_source_jsonl", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare joint_assembly results across three stages: "
            "base, mid, and reverse-check."
        )
    )

    # parser.add_argument(
    #    "--base-dir",
    #    type=Path,
    #    default=Path(__file__).resolve().parents[2].joinpath("tests/external_cases/results/joint_assembly"),
    #    help="Baseline result directory, e.g. old 10 split files.",
    # )
    # parser.add_argument(
    #    "--mid-dir",
    #    type=Path,
    #    default=Path(__file__).resolve().parents[2].joinpath("tests/external_cases/results/joint_assembly_new"),
    #    help="Middle result directory, e.g. fixed-connection/no-reverse run.",
    # )
    # parser.add_argument(
    #    "--rev-dir",
    #    type=Path,
    #    default=Path(__file__).resolve().parents[2].joinpath("tests/external_cases/results/joint_assembly_new_with_reverse"),
    #    help="Reverse-check result directory.",
    # )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        type=Path,
        default=[
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly"),
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly_rotation"),
        ],
        help="Baseline result directories.",
    )
    parser.add_argument(
        "--mid-dirs",
        nargs="+",
        type=Path,
        default=[
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly_new"),
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly_rotation_new"),
        ],
        help="Middle result directories.",
    )
    parser.add_argument(
        "--rev-dirs",
        nargs="+",
        type=Path,
        default=[
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly_new_with_reverse"),
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("private/results/joint_assembly_rotation_new_with_reverse"),
        ],
        help="Reverse-check result directories.",
    )

    parser.add_argument("--base-csv", nargs="*", type=Path, default=None)
    parser.add_argument("--mid-csv", nargs="*", type=Path, default=None)
    parser.add_argument("--rev-csv", nargs="*", type=Path, default=None)

    parser.add_argument("--base-jsonl", nargs="*", type=Path, default=None)
    parser.add_argument("--mid-jsonl", nargs="*", type=Path, default=None)
    parser.add_argument("--rev-jsonl", nargs="*", type=Path, default=None)

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/joint_three_stage_compare"),
    )

    args = parser.parse_args()

    # base_csvs = normalize_file_args(args.base_csv, args.base_dir, ".csv")
    # mid_csvs = normalize_file_args(args.mid_csv, args.mid_dir, ".csv")
    # rev_csvs = normalize_file_args(args.rev_csv, args.rev_dir, ".csv")
    #
    # base_jsonls = normalize_file_args(args.base_jsonl, args.base_dir, ".jsonl")
    # mid_jsonls = normalize_file_args(args.mid_jsonl, args.mid_dir, ".jsonl")
    # rev_jsonls = normalize_file_args(args.rev_jsonl, args.rev_dir, ".jsonl")
    base_csvs = normalize_file_args(args.base_csv, args.base_dirs, ".csv")
    mid_csvs = normalize_file_args(args.mid_csv, args.mid_dirs, ".csv")
    rev_csvs = normalize_file_args(args.rev_csv, args.rev_dirs, ".csv")

    base_jsonls = normalize_file_args(args.base_jsonl, args.base_dirs, ".jsonl")
    mid_jsonls = normalize_file_args(args.mid_jsonl, args.mid_dirs, ".jsonl")
    rev_jsonls = normalize_file_args(args.rev_jsonl, args.rev_dirs, ".jsonl")

    print("Base CSV files:")
    for p in base_csvs:
        print(f"  {p}")
    print("Mid CSV files:")
    for p in mid_csvs:
        print(f"  {p}")
    print("Rev CSV files:")
    for p in rev_csvs:
        print(f"  {p}")
    print()

    base_rows = load_csv_summaries(base_csvs, "base")
    mid_rows = load_csv_summaries(mid_csvs, "mid")
    rev_rows = load_csv_summaries(rev_csvs, "rev")

    base_info = load_jsonl_infos(base_jsonls, "base")
    mid_info = load_jsonl_infos(mid_jsonls, "mid")
    rev_info = load_jsonl_infos(rev_jsonls, "rev")

    all_names = sorted(set(base_rows) | set(mid_rows) | set(rev_rows))

    base_total, base_succ, base_fail = stage_counts(base_rows)
    mid_total, mid_succ, mid_fail = stage_counts(mid_rows)
    rev_total, rev_succ, rev_fail = stage_counts(rev_rows)

    print("Stage totals:")
    print(f"  base: total={base_total} success={base_succ} fail={base_fail}")
    print(f"  mid:  total={mid_total} success={mid_succ} fail={mid_fail}")
    print(f"  rev:  total={rev_total} success={rev_succ} fail={rev_fail}")
    print()

    missing = {
        "missing_in_base": sorted(set(all_names) - set(base_rows)),
        "missing_in_mid": sorted(set(all_names) - set(mid_rows)),
        "missing_in_rev": sorted(set(all_names) - set(rev_rows)),
    }

    for key, names in missing.items():
        if names:
            print(f"[warn] {key}: {len(names)}")
    if any(missing.values()):
        print()

    patterns: dict[str, list[str]] = defaultdict(list)

    for name in all_names:
        pattern = (
            status_char(base_rows.get(name))
            + status_char(mid_rows.get(name))
            + status_char(rev_rows.get(name))
        )
        patterns[pattern].append(name)

    pattern_counts = Counter({k: len(v) for k, v in patterns.items()})

    print("Pattern counts, S=success, F=fail:")
    for pattern in sorted(pattern_counts):
        print(f"  {pattern}: {pattern_counts[pattern]}")
    print()

    # Pairwise transitions.
    base_success = {n for n, r in base_rows.items() if is_success(r)}
    base_fail_set = set(base_rows) - base_success

    mid_success = {n for n, r in mid_rows.items() if is_success(r)}
    mid_fail_set = set(mid_rows) - mid_success

    rev_success = {n for n, r in rev_rows.items() if is_success(r)}
    rev_fail_set = set(rev_rows) - rev_success

    base_to_mid_new_fail = sorted(base_success & mid_fail_set)
    base_to_mid_new_success = sorted(base_fail_set & mid_success)

    mid_to_rev_new_fail = sorted(mid_success & rev_fail_set)
    mid_to_rev_new_success = sorted(mid_fail_set & rev_success)

    base_to_rev_new_fail = sorted(base_success & rev_fail_set)
    base_to_rev_new_success = sorted(base_fail_set & rev_success)

    print("Pairwise changes:")
    print(f"  base -> mid newly failed:      {len(base_to_mid_new_fail)}")
    print(f"  base -> mid newly successful:  {len(base_to_mid_new_success)}")
    print(f"  mid  -> rev newly failed:      {len(mid_to_rev_new_fail)}")
    print(f"  mid  -> rev newly successful:  {len(mid_to_rev_new_success)}")
    print(f"  base -> rev newly failed:      {len(base_to_rev_new_fail)}")
    print(f"  base -> rev newly successful:  {len(base_to_rev_new_success)}")
    print()

    # Reason summaries for the most important regression sets.
    print("Reason summary for mid -> rev newly failed:")
    reason_counter = Counter(
        reason_key(rev_info.get(name)) for name in mid_to_rev_new_fail
    )
    for reason, count in reason_counter.most_common():
        print(f"  {reason or '<unknown>'}: {count}")
    print()

    print("Reason summary for base -> mid newly failed:")
    reason_counter = Counter(
        reason_key(mid_info.get(name)) for name in base_to_mid_new_fail
    )
    for reason, count in reason_counter.most_common():
        print(f"  {reason or '<unknown>'}: {count}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write lists useful for rendering.
    write_case_list(args.out_dir / "base_to_mid_newly_failed.txt", base_to_mid_new_fail)
    write_case_list(
        args.out_dir / "base_to_mid_newly_successful.txt", base_to_mid_new_success
    )
    write_case_list(args.out_dir / "mid_to_rev_newly_failed.txt", mid_to_rev_new_fail)
    write_case_list(
        args.out_dir / "mid_to_rev_newly_successful.txt", mid_to_rev_new_success
    )
    write_case_list(args.out_dir / "base_to_rev_newly_failed.txt", base_to_rev_new_fail)
    write_case_list(
        args.out_dir / "base_to_rev_newly_successful.txt", base_to_rev_new_success
    )

    # Write pattern-specific lists.
    for pattern, names in sorted(patterns.items()):
        write_case_list(args.out_dir / f"pattern_{pattern}.txt", sorted(names))

    # Detailed table for later inspection.
    detail_csv = args.out_dir / "three_stage_comparison_details.csv"
    write_detailed_csv(
        path=detail_csv,
        all_names=all_names,
        base_rows=base_rows,
        mid_rows=mid_rows,
        rev_rows=rev_rows,
        base_info=base_info,
        mid_info=mid_info,
        rev_info=rev_info,
    )

    print("Wrote outputs:")
    print(f"  {args.out_dir / 'base_to_mid_newly_failed.txt'}")
    print(f"  {args.out_dir / 'mid_to_rev_newly_failed.txt'}")
    print(f"  {args.out_dir / 'base_to_rev_newly_failed.txt'}")
    print(f"  {detail_csv}")


if __name__ == "__main__":
    main()
