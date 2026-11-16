from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import multiprocessing as mp
import random
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part


def discover_obj_parts(case_dir: Path) -> list[Path]:
    """Return all .obj files in a case directory, sorted by numeric stem if possible."""
    obj_files = list(case_dir.glob("*.obj"))

    def sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(obj_files, key=sort_key)


def case_name_from_dir(case_dir: Path, dataset_dir: Path | None = None) -> str:
    """
    Build a stable converted assembly name.

    Examples:
        dataset_dir = .../joint_assembly
        case_dir    = .../joint_assembly/00016
        -> joint_assembly_00016

        dataset_dir = .../joint_assembly_rotation
        case_dir    = .../joint_assembly_rotation/general/00016
        -> joint_assembly_rotation_general_00016
    """
    if dataset_dir is None:
        return f"{case_dir.parent.name}_{case_dir.name}"

    dataset_dir = Path(dataset_dir).resolve()
    case_dir = Path(case_dir).resolve()

    rel = case_dir.relative_to(dataset_dir)
    rel_parts = rel.parts

    safe_parts = [dataset_dir.name] + list(rel_parts)
    return "_".join(safe_parts)


def case_xml_path(
    case_dir: Path, out_root: Path, dataset_dir: Path | None = None
) -> Path:
    """Return the expected XML output path for a converted case."""
    case_name = case_name_from_dir(case_dir, dataset_dir=dataset_dir)
    return out_root / case_name / "xml" / f"{case_name}.xml"


def discover_case_dirs(dataset_dir: Path) -> list[Path]:
    """
    Return all subdirectories that look like assembly cases.

    A valid case directory is defined as a directory that directly contains
    at least one .obj file.

    This supports both:
        assets/joint_assembly/00016/*.obj

    and nested datasets such as:
        assets/joint_assembly_rotation/general/00016/*.obj
        assets/joint_assembly_rotation/puzzle/00031/*.obj
        assets/joint_assembly_rotation/screw/00042/*.obj
    """
    dataset_dir = Path(dataset_dir).resolve()

    case_dirs = []
    for p in dataset_dir.rglob("*"):
        if not p.is_dir():
            continue
        if any(p.glob("*.obj")):
            case_dirs.append(p)

    def sort_key(path: Path):
        rel = path.relative_to(dataset_dir)
        parts = rel.parts

        key = []
        for item in parts:
            if item.isdigit():
                key.append((0, int(item)))
            else:
                key.append((1, item))
        return key

    return sorted(case_dirs, key=sort_key)


def convert_obj_to_stl(obj_path: Path, stl_path: Path) -> None:
    """Load OBJ with Open3D and export as STL."""
    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load OBJ or mesh is empty: {obj_path}")
    if len(mesh.triangles) == 0:
        raise ValueError(f"OBJ has no triangles: {obj_path}")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(stl_path), mesh)
    if not ok:
        raise IOError(f"Failed to write STL: {stl_path}")


def build_parts_from_case(case_dir: Path, mesh_out_dir: Path) -> list[Part]:
    """Convert all OBJ parts to STL and construct Part objects."""
    obj_files = discover_obj_parts(case_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {case_dir}")

    parts: list[Part] = []
    for obj_file in obj_files:
        part_name = f"part_{obj_file.stem}"
        stl_path = mesh_out_dir / f"{part_name}.stl"

        convert_obj_to_stl(obj_file, stl_path)

        part = Part(
            name=part_name,
            mesh_path=str(stl_path),
            density=8000.0,
            material="Generic",
        )
        parts.append(part)

    return parts


def build_assembly(case_name: str, parts: list[Part]) -> Assembly:
    """Construct an Assembly from converted parts."""
    assembly = Assembly(name=case_name)
    for part in parts:
        assembly.add_part(part)
    return assembly


def export_assembly_xml(assembly: Assembly, xml_path: Path) -> None:
    """
    Export assembly in the flat XML format expected by Assembly.load_from_xml():
      <assembly>
        <part ... />
        <part ... />
        <connection ... />
        <connection ... />
      </assembly>
    """
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("assembly")
    root.set("name", assembly.name)

    for part in assembly.parts.values():
        root.append(part.to_xml_element())

    file_prefix = str(xml_path.parent) + "/"
    for conn in assembly.connections:
        try:
            conn_el = conn.to_xml_element(file_prefix=file_prefix)
        except TypeError:
            conn_el = conn.to_xml_element()
        root.append(conn_el)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="\t", level=0)
    except AttributeError:
        pass
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def import_case(
    case_dir: Path, out_root: Path, dataset_dir: Path | None = None
) -> dict:
    """Full pipeline for one Assemble-Them-All case. Returns a summary dict."""
    case_name = case_name_from_dir(case_dir, dataset_dir=dataset_dir)
    case_out_dir = out_root / case_name
    mesh_out_dir = case_out_dir / "meshes"
    xml_out_dir = case_out_dir / "xml"
    xml_path = xml_out_dir / f"{case_name}.xml"

    print(f"[INFO] Importing case: {case_dir}")
    print(f"[INFO] Output dir: {case_out_dir}")

    parts = build_parts_from_case(case_dir, mesh_out_dir)
    print(f"[INFO] Converted {len(parts)} parts to STL")

    assembly = build_assembly(case_name, parts)
    print(f"[INFO] Built assembly with {len(assembly.parts)} parts")

    print("[INFO] Recomputing geometric contacts...")
    assembly.find_contacts()
    print(f"[INFO] Reconstructed {len(assembly.connections)} connections")

    print(f"[INFO] Exporting XML to {xml_path}")
    export_assembly_xml(assembly, xml_path)

    return {
        "source_dataset": case_dir.parent.name,
        "case_id": case_dir.name,
        "assembly_name": case_name,
        "num_parts": len(assembly.parts),
        "num_connections": len(assembly.connections),
        "xml_path": str(xml_path),
        "status": "success",
        "error": "",
    }


def count_obj_parts(case_dir: Path) -> int:
    """Cheap pre-scan: count .obj files without loading meshes."""
    return len(discover_obj_parts(case_dir))


def adaptive_case_timeout_sec(num_parts: int) -> float:
    """
    Adaptive timeout for one assembly import.

    Formula:
        timeout = min(900, max(90, 45 + 0.35 * pair_count))
        pair_count = n_parts * (n_parts - 1) / 2

    This reflects that contact reconstruction scales roughly with pair count.
    """
    n = int(num_parts)
    pair_count = n * (n - 1) / 2.0
    return float(min(900.0, max(90.0, 45.0 + 0.35 * pair_count)))


def _import_case_worker(
    case_dir_str: str,
    out_root_str: str,
    dataset_dir_str: str | None,
    queue,
) -> None:
    """
    Worker process for one case.

    Running each case in a separate process allows the parent process to
    terminate pathological cases that exceed the timeout.
    """
    try:
        case_dir = Path(case_dir_str)
        out_root = Path(out_root_str)
        dataset_dir = Path(dataset_dir_str) if dataset_dir_str is not None else None

        row = import_case(case_dir, out_root, dataset_dir=dataset_dir)
        queue.put(
            {
                "ok": True,
                "row": row,
                "traceback": "",
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "row": None,
                "traceback": traceback.format_exc(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def import_case_with_timeout(
    case_dir: Path,
    out_root: Path,
    *,
    dataset_dir: Path | None,
    timeout_sec: float,
) -> dict:
    """
    Import one case in a child process and terminate it if it exceeds timeout_sec.
    """
    case_name = case_name_from_dir(case_dir, dataset_dir=dataset_dir)
    xml_path = case_xml_path(case_dir, out_root, dataset_dir=dataset_dir)

    num_parts_prescan = count_obj_parts(case_dir)
    t0 = time.time()

    ctx = mp.get_context("spawn")

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_import_case_worker,
        args=(
            str(case_dir),
            str(out_root),
            str(dataset_dir) if dataset_dir is not None else None,
            queue,
        ),
    )

    proc.start()
    proc.join(timeout=float(timeout_sec))

    runtime_sec = time.time() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

        return {
            "source_dataset": dataset_dir.name
            if dataset_dir is not None
            else case_dir.parent.name,
            "case_id": str(case_dir.relative_to(dataset_dir))
            if dataset_dir is not None
            else case_dir.name,
            "assembly_name": case_name,
            "num_parts": num_parts_prescan,
            "num_connections": "",
            "xml_path": str(xml_path),
            "status": "timeout",
            "error": f"Timeout after {timeout_sec:.1f} s",
            "runtime_sec": round(runtime_sec, 3),
            "timeout_sec": round(float(timeout_sec), 3),
        }

    if queue.empty():
        return {
            "source_dataset": dataset_dir.name
            if dataset_dir is not None
            else case_dir.parent.name,
            "case_id": str(case_dir.relative_to(dataset_dir))
            if dataset_dir is not None
            else case_dir.name,
            "assembly_name": case_name,
            "num_parts": num_parts_prescan,
            "num_connections": "",
            "xml_path": str(xml_path),
            "status": "failed",
            "error": f"Worker exited without returning a result. exitcode={proc.exitcode}",
            "runtime_sec": round(runtime_sec, 3),
            "timeout_sec": round(float(timeout_sec), 3),
        }

    result = queue.get()

    if result.get("ok"):
        row = result["row"]
        row["runtime_sec"] = round(runtime_sec, 3)
        row["timeout_sec"] = round(float(timeout_sec), 3)

        # Preserve pre-scan count even if something unusual happens.
        if row.get("num_parts", "") in ("", None):
            row["num_parts"] = num_parts_prescan

        return row

    return {
        "source_dataset": dataset_dir.name
        if dataset_dir is not None
        else case_dir.parent.name,
        "case_id": str(case_dir.relative_to(dataset_dir))
        if dataset_dir is not None
        else case_dir.name,
        "assembly_name": case_name,
        "num_parts": num_parts_prescan,
        "num_connections": "",
        "xml_path": str(xml_path),
        "status": "failed",
        "error": result.get("error", "unknown worker error"),
        "runtime_sec": round(runtime_sec, 3),
        "timeout_sec": round(float(timeout_sec), 3),
    }


def build_multi_part_count_quotas(
    *,
    low_total: int = 50,
    middle_total: int = 900,
    high_total: int = 50,
) -> dict[int, int]:
    """
    Desired part-count distribution for multi_assembly.

    Low:
        5–9 parts, total about 50

    Middle:
        10–37 parts, total about 900
        10–13 get 33 each, 14–37 get 32 each

    High:
        38–42 parts, total about 50

    If the dataset cannot satisfy this exactly, the importer later fills
    missing slots from other available bins.
    """
    quotas: dict[int, int] = {}

    # 5–9: 50 total -> 10 each
    low_counts = list(range(5, 10))
    base = low_total // len(low_counts)
    rem = low_total % len(low_counts)
    for i, n in enumerate(low_counts):
        quotas[n] = base + (1 if i < rem else 0)

    # 10–37: 900 total -> 10–13 get 33, 14–37 get 32
    middle_counts = list(range(10, 38))
    base = middle_total // len(middle_counts)  # 32
    rem = middle_total % len(middle_counts)  # 4
    for i, n in enumerate(middle_counts):
        quotas[n] = base + (1 if i < rem else 0)

    # 38–42: 50 total -> 10 each
    high_counts = list(range(38, 43))
    base = high_total // len(high_counts)
    rem = high_total % len(high_counts)
    for i, n in enumerate(high_counts):
        quotas[n] = base + (1 if i < rem else 0)

    return quotas


def make_stratified_multi_candidate_queue(
    *,
    case_dirs: list[Path],
    dataset_dir: Path,
    out_root: Path,
    target_total: int,
    seed: int,
    skip_existing: bool,
    min_parts: int = 5,
    max_parts: int = 42,
) -> tuple[list[tuple[Path, int]], dict]:
    """
    Build an ordered candidate queue for stratified multi_assembly import.

    Important behavior:
    - Only cases with part count in [min_parts, max_parts] are considered.
    - Existing XMLs are removed from the candidate pool if skip_existing=True.
    - Desired quota is attempted first.
    - If some bins lack enough candidates, remaining slots are filled from
      all other available bins.
    - The returned queue may be longer than target_total so that failed/timeout
      cases can be replaced by later candidates.
    """
    rng = random.Random(seed)

    buckets: dict[int, list[Path]] = defaultdict(list)
    excluded_existing = 0
    excluded_part_count = 0

    for case_dir in case_dirs:
        n_parts = count_obj_parts(case_dir)

        if n_parts < min_parts or n_parts > max_parts:
            excluded_part_count += 1
            continue

        if skip_existing:
            xml_path = case_xml_path(case_dir, out_root, dataset_dir=dataset_dir)
            if xml_path.exists():
                excluded_existing += 1
                continue

        buckets[n_parts].append(case_dir)

    for dirs in buckets.values():
        rng.shuffle(dirs)

    quotas = build_multi_part_count_quotas()
    selected: list[tuple[Path, int]] = []
    selected_set: set[Path] = set()

    # First pass: satisfy desired per-part-count quotas as much as possible.
    for n_parts in sorted(quotas.keys()):
        desired = int(quotas[n_parts])
        available = buckets.get(n_parts, [])

        take = min(desired, len(available))
        for case_dir in available[:take]:
            selected.append((case_dir, n_parts))
            selected_set.add(case_dir)

        buckets[n_parts] = available[take:]

    # Second pass: fill shortages from all remaining valid cases.
    remaining: list[tuple[Path, int]] = []
    for n_parts, dirs in buckets.items():
        for case_dir in dirs:
            if case_dir not in selected_set:
                remaining.append((case_dir, n_parts))

    rng.shuffle(remaining)

    for item in remaining:
        selected.append(item)

    # Do not cut to target_total here.
    # We keep extra candidates so timeout/failed cases can be replaced naturally.
    stats = {
        "target_total": int(target_total),
        "valid_candidate_count": sum(len(v) for v in buckets.values())
        + len(selected_set),
        "queue_length": len(selected),
        "excluded_existing": excluded_existing,
        "excluded_part_count": excluded_part_count,
        "min_parts": min_parts,
        "max_parts": max_parts,
        "desired_quotas": quotas,
        "available_by_part_count": {
            int(k): int(len(v) + sum(1 for _, n in selected if n == k))
            for k, v in sorted(buckets.items())
        },
    }

    return selected, stats


def import_stratified_multi_dataset(
    *,
    dataset_dir: Path,
    out_root: Path,
    target_total: int = 1000,
    seed: int = 42,
    skip_existing: bool = False,
    min_parts: int = 5,
    max_parts: int = 42,
) -> list[dict]:
    """
    Stratified importer for Assemble-Them-All multi_assembly.

    It keeps importing until either:
    - target_total successful conversions are reached, or
    - all valid candidate cases are exhausted.

    Timeout/failed cases are recorded but do not count toward success_total.
    """
    case_dirs = discover_case_dirs(dataset_dir)

    queue, stats = make_stratified_multi_candidate_queue(
        case_dirs=case_dirs,
        dataset_dir=dataset_dir,
        out_root=out_root,
        target_total=target_total,
        seed=seed,
        skip_existing=skip_existing,
        min_parts=min_parts,
        max_parts=max_parts,
    )

    print("=" * 80)
    print("[STRATIFIED MULTI IMPORT]")
    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] discovered case dirs={len(case_dirs)}")
    print(f"[INFO] queue length={len(queue)}")
    print(f"[INFO] target successful imports={target_total}")
    print(f"[INFO] excluded existing={stats['excluded_existing']}")
    print(f"[INFO] excluded by part count={stats['excluded_part_count']}")
    print(f"[INFO] part-count range=[{min_parts}, {max_parts}]")
    print("=" * 80)

    rows: list[dict] = []
    success_count = 0
    attempted_count = 0

    for case_dir, n_parts in queue:
        if success_count >= target_total:
            break

        attempted_count += 1
        case_name = case_name_from_dir(case_dir, dataset_dir=dataset_dir)
        timeout_sec = adaptive_case_timeout_sec(n_parts)

        print("=" * 80)
        print(
            f"[INFO] Attempt {attempted_count}, success {success_count}/{target_total}: "
            f"{case_name} | parts={n_parts} | timeout={timeout_sec:.1f}s"
        )

        row = import_case_with_timeout(
            case_dir,
            out_root,
            dataset_dir=dataset_dir,
            timeout_sec=timeout_sec,
        )

        row["part_count_bin"] = int(n_parts)
        row["target_success_index"] = (
            success_count + 1 if row.get("status") == "success" else ""
        )
        rows.append(row)

        if row.get("status") == "success":
            success_count += 1
            print(
                f"[OK] {case_name}: success_count={success_count}/{target_total}, "
                f"connections={row.get('num_connections')}, runtime={row.get('runtime_sec')}s"
            )
        else:
            print(
                f"[SKIP] {case_name}: status={row.get('status')}, "
                f"error={row.get('error')}, runtime={row.get('runtime_sec')}s"
            )

    print("=" * 80)
    print(f"[DONE] attempted={attempted_count}")
    print(f"[DONE] successful={success_count}/{target_total}")
    print(f"[DONE] rows recorded={len(rows)}")

    if success_count < target_total:
        print(
            f"[WARN] Only {success_count}/{target_total} cases were successfully imported. "
            "Candidate pool was exhausted or too many cases failed/timed out."
        )

    return rows


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    """Write batch import summary to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_dataset",
        "case_id",
        "assembly_name",
        "part_count_bin",
        "target_success_index",
        "num_parts",
        "num_connections",
        "runtime_sec",
        "timeout_sec",
        "xml_path",
        "status",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_dataset(
    dataset_dir: Path,
    out_root: Path,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
    case_ids: set[str] | None = None,
    skip_existing: bool = False,
) -> list[dict]:
    """
    Batch import all or selected cases from a dataset directory.

    Important behavior:
    - If skip_existing=True, already converted cases are removed from the candidate
      pool BEFORE random sampling.
    - Therefore repeated calls with --sample 100 --skip-existing naturally produce
      non-overlapping batches, as long as out_root is the same.
    """
    case_dirs = discover_case_dirs(dataset_dir)

    if case_ids:
        case_dirs = [p for p in case_dirs if p.name in case_ids]

    total_before_existing_filter = len(case_dirs)

    # IMPORTANT: filter existing cases BEFORE sampling.
    # This ensures the next sample does not waste slots on already converted cases.
    if skip_existing:
        remaining_case_dirs = []
        skipped_existing_count = 0

        for p in case_dirs:
            xml_path = case_xml_path(p, out_root, dataset_dir=dataset_dir)
            if xml_path.exists():
                skipped_existing_count += 1
            else:
                remaining_case_dirs.append(p)

        case_dirs = remaining_case_dirs

        print(
            f"[INFO] skip_existing=True: excluded {skipped_existing_count} already converted cases "
            f"from {total_before_existing_filter} candidates"
        )

    if sample is not None:
        if sample > len(case_dirs):
            raise ValueError(
                f"Requested sample size {sample}, but only {len(case_dirs)} non-existing candidate cases "
                f"are available in {dataset_dir}. "
                f"Try a smaller --sample or check --out-root / --skip-existing."
            )

        rng = random.Random(seed)
        case_dirs = rng.sample(case_dirs, sample)

        # Keep output order deterministic and readable after sampling.
        def sort_key(path: Path):
            name = path.name
            return (0, int(name)) if name.isdigit() else (1, name)

        case_dirs = sorted(case_dirs, key=sort_key)

    if limit is not None:
        case_dirs = case_dirs[:limit]

    print(f"[INFO] Selected {len(case_dirs)} candidate cases in {dataset_dir}")

    rows: list[dict] = []

    for idx, case_dir in enumerate(case_dirs, start=1):
        case_name = case_name_from_dir(case_dir, dataset_dir=dataset_dir)
        xml_path = case_xml_path(case_dir, out_root, dataset_dir=dataset_dir)

        print("=" * 80)
        print(f"[INFO] [{idx}/{len(case_dirs)}] Processing {case_dir.name}")

        # This is now mostly a safety guard, because existing cases were already
        # filtered out before sampling.
        if skip_existing and xml_path.exists():
            print(f"[INFO] Skipping existing case: {xml_path}")
            rows.append(
                {
                    "source_dataset": case_dir.parent.name,
                    "case_id": case_dir.name,
                    "assembly_name": case_name,
                    "num_parts": "",
                    "num_connections": "",
                    "xml_path": str(xml_path),
                    "status": "skipped_existing",
                    "error": "",
                }
            )
            continue

        try:
            row = import_case(case_dir, out_root, dataset_dir=dataset_dir)
            rows.append(row)
        except Exception as exc:
            print(f"[ERROR] Failed to import case {case_dir}")
            print(traceback.format_exc())

            rows.append(
                {
                    "source_dataset": case_dir.parent.name,
                    "case_id": case_dir.name,
                    "assembly_name": case_name,
                    "num_parts": "",
                    "num_connections": "",
                    "xml_path": str(xml_path),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return rows


def parse_case_ids(case_ids_raw: str | None) -> set[str] | None:
    """Parse comma-separated case ids like '00016,00031'."""
    if not case_ids_raw:
        return None
    return {token.strip() for token in case_ids_raw.split(",") if token.strip()}


def make_batch_range_label(
    *,
    batch_start: int | None,
    batch_size: int | None,
    num_rows: int,
) -> str:
    """
    Build a readable batch range label.

    Examples:
        batch_start=1, batch_size=100  -> "1-100"
        batch_start=101, batch_size=100 -> "101-200"
        no batch info -> "100"
    """
    if batch_start is None:
        return str(num_rows)

    if batch_start < 1:
        raise ValueError("batch_start must be >= 1")

    if batch_size is None:
        # Use actual number of rows as the range length.
        batch_end = batch_start + max(num_rows, 1) - 1
    else:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        batch_end = batch_start + batch_size - 1

    return f"{batch_start}-{batch_end}"


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case-dir", help="Path to a single assembly case directory")
    mode.add_argument(
        "--dataset-dir",
        help="Path to a dataset directory containing many case subdirectories",
    )

    parser.add_argument(
        "--out-root",
        default=str(
            Path(__file__).resolve().parents[3] / "external" / "test_cases_external"
        ),
        help="Directory where converted cases will be written",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N cases"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N cases without replacement",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=None,
        help="Optional 1-based batch start label for summary naming, e.g. 1 or 101",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size label for summary naming, e.g. 100",
    )
    parser.add_argument(
        "--case-ids", default=None, help="Comma-separated case ids, e.g. 00016,00031"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose XML already exists",
    )
    parser.add_argument(
        "--summary-csv", default=None, help="Optional CSV path for batch summary"
    )
    parser.add_argument(
        "--stratified-multi",
        action="store_true",
        help="Use stratified multi_assembly import with adaptive per-case timeout and replacement after timeout/failure",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1000,
        help="Target number of successful imports for --stratified-multi",
    )
    parser.add_argument(
        "--min-parts",
        type=int,
        default=5,
        help="Minimum number of OBJ parts for --stratified-multi",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=42,
        help="Maximum number of OBJ parts for --stratified-multi",
    )

    args = parser.parse_args()

    if args.limit is not None and args.sample is not None:
        parser.error("--limit and --sample should not be used together")

    if args.batch_start is not None and args.batch_start < 1:
        parser.error("--batch-start must be >= 1")

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    out_root = Path(args.out_root).resolve()

    if args.case_dir:
        case_dir = Path(args.case_dir).resolve()
        row = import_case(case_dir, out_root)
        print(f"[DONE] XML written to: {row['xml_path']}")
        return

    dataset_dir = Path(args.dataset_dir).resolve()
    case_ids = parse_case_ids(args.case_ids)

    if args.stratified_multi:
        if args.limit is not None or args.sample is not None or case_ids is not None:
            parser.error(
                "--stratified-multi should not be combined with --limit, --sample, or --case-ids"
            )

        rows = import_stratified_multi_dataset(
            dataset_dir=dataset_dir,
            out_root=out_root,
            target_total=args.target_total,
            seed=args.seed,
            skip_existing=args.skip_existing,
            min_parts=args.min_parts,
            max_parts=args.max_parts,
        )

        if args.summary_csv:
            csv_path = Path(args.summary_csv).resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = (
                out_root
                / f"{dataset_dir.name}_import_stratified_{args.target_total}_{timestamp}.csv"
            )

        write_summary_csv(rows, csv_path)
        print(f"[DONE] Summary written to: {csv_path}")
        return

    rows = import_dataset(
        dataset_dir=dataset_dir,
        out_root=out_root,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        case_ids=case_ids,
        skip_existing=args.skip_existing,
    )

    if args.summary_csv:
        csv_path = Path(args.summary_csv).resolve()
    else:
        range_label = make_batch_range_label(
            batch_start=args.batch_start,
            batch_size=args.batch_size,
            num_rows=len(rows),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_root / f"{dataset_dir.name}_import_{range_label}_{timestamp}.csv"

    write_summary_csv(rows, csv_path)
    print(f"[DONE] Summary written to: {csv_path}")


if __name__ == "__main__":
    main()
