from __future__ import annotations

"""
Render thesis-style assembly overview figures for external 2-part failure cases.

This script scans rollout JSONL files under:

    ../private/results/joint_assembly
    ../private/results/joint_assembly_rotation

It selects assemblies that contain at least one failed attempt by default, reloads the
corresponding converted XML assembly, and renders compact overview figures with:

    - target part: KIT yellow
    - other parts: KIT deep blue, transparent
    - mesh edge overlay
    - bottom legend: target part / other part

The script is intentionally simpler than the HTML gallery renderer. It produces
standalone overview figures suitable for direct use in the thesis.

Recommended run:

cd <repo>

PYTHONPATH=src:$PYTHONPATH \
python tools/rendering_tool/render_failure_case_overviews.py \
  --results-root ../private/results \
  --groups joint_assembly joint_assembly_rotation \
  --cases-root ../external/test_cases_external \
  --out ../private/figures/failure_cases_joint_assembly_cases \
  --formats png pdf \
  --target-mode first-failed

For a quick smoke test:

python tools/rendering_tool/render_failure_case_overviews.py --max-cases 10 --formats png
"""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------------------------------------------------------
# Visual style: KIT / wbk
# -----------------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838f"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
DARK = "#222222"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OverviewRequest:
    group: str
    assembly_name: str
    xml_path: Path
    target_part: str | None
    attempt_index: int | None
    failed_attempt_count: int
    report_success: bool
    stop_reason: str | None
    completion: float | None
    source_jsonl: Path


# -----------------------------------------------------------------------------
# Project imports and path handling
# -----------------------------------------------------------------------------


def add_project_paths(project_root: Path | None) -> None:
    """Add project root and src directory to sys.path for local package imports."""
    candidates: list[Path] = []

    if project_root is not None:
        candidates.append(project_root)
        candidates.append(project_root / "src")

    cwd = Path.cwd()
    candidates.extend([cwd, cwd / "src", cwd.parent, cwd.parent / "src"])

    for p in candidates:
        if p.exists():
            sp = str(p.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)


def resolve_existing_file(path_str: str, xml_path: Path) -> str:
    """Resolve mesh and npy paths stored in converted XML files."""
    p = Path(path_str)
    if p.exists():
        return str(p)

    name = Path(path_str).name
    candidates = [
        xml_path.parent / path_str,
        xml_path.parent.parent / path_str,
        xml_path.parent / name,
        xml_path.parent.parent / name,
        xml_path.parent / "stl" / name,
        xml_path.parent.parent / "stl" / name,
        xml_path.parent / "obj" / name,
        xml_path.parent.parent / "obj" / name,
        xml_path.parent / f"{xml_path.stem}_data" / name,
        xml_path.parent.parent / f"{xml_path.stem}_data" / name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(p)


def resolve_xml_path(raw_xml: str | None, assembly_name: str, cases_root: Path) -> Path:
    """Resolve XML path from report value, falling back to the converted-case layout."""
    if raw_xml:
        p = Path(raw_xml).expanduser()
        if p.exists():
            return p.resolve()

        # If the report was created on a different path, preserve the tail from
        # the assembly folder onward and attach it to the current cases_root.
        parts = list(p.parts)
        if assembly_name in parts:
            idx = parts.index(assembly_name)
            tail = Path(*parts[idx:])
            candidate = cases_root / tail
            if candidate.exists():
                return candidate.resolve()

    return (cases_root / assembly_name / "xml" / f"{assembly_name}.xml").resolve()


def load_assembly_with_resolved_paths(xml_path: Path):
    """Load an Assembly while resolving relative mesh/contact-data paths robustly."""
    from diss_assembly_dataclasses.assembly import Assembly
    from diss_assembly_dataclasses.connection import (
        Connection,
        Fixed_Connection,
        Geometric_Connection,
        Safety_Ring_Connection,
        Screw_Connection,
    )
    from diss_assembly_dataclasses.part import Part

    xml_path = xml_path.expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    assembly = Assembly(root.attrib.get("name", xml_path.stem))

    for part_element in root.findall("part"):
        elem = ET.Element("part", dict(part_element.attrib))
        if "mesh_path" in elem.attrib:
            elem.set(
                "mesh_path", resolve_existing_file(elem.attrib["mesh_path"], xml_path)
            )
        part = Part.from_xml_element(elem)
        assembly.add_part(part)

    for conn_element in root.findall("connection"):
        elem = ET.Element("connection", dict(conn_element.attrib))
        for key in ("points_filename", "normals_filename", "distances_filename"):
            if key in elem.attrib:
                elem.set(key, resolve_existing_file(elem.attrib[key], xml_path))

        conn_type = elem.attrib.get("type", "generic")
        if conn_type == "geometric":
            conn = Geometric_Connection.from_xml_element(elem)
        elif conn_type == "screw":
            conn = Screw_Connection.from_xml_element(elem)
        elif conn_type == "safety_ring":
            conn = Safety_Ring_Connection.from_xml_element(elem)
        elif conn_type == "fixed":
            conn = Fixed_Connection.from_xml_element(elem)
        else:
            conn = Connection.from_xml_element(elem)

        assembly.add_connection(conn)

    assembly.xml_path = str(xml_path)
    return assembly


# -----------------------------------------------------------------------------
# JSONL parsing and selection
# -----------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                print(f"[WARN] Could not parse {path}:{line_idx}: {exc}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def attempt_success(attempt: dict[str, Any]) -> bool:
    return parse_bool(attempt.get("success", False))


def attempt_failed(attempt: dict[str, Any]) -> bool:
    return not attempt_success(attempt)


def get_attempt_target(attempt: dict[str, Any]) -> str | None:
    debug = attempt.get("debug", {}) or {}
    for key in ("target_part", "target", "part_name"):
        value = debug.get(key)
        if isinstance(value, str) and value:
            return value

    value = attempt.get("target")
    if isinstance(value, str) and value:
        return value

    # Some logs store target in operation data.
    op_data = debug.get("operation_data", {}) or {}
    for key in ("target_part", "target", "part_name"):
        value = op_data.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def find_result_jsonl_files(
    results_root: Path, groups: list[str]
) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    for group in groups:
        folder = results_root / group
        if not folder.exists():
            print(f"[WARN] Result folder does not exist: {folder}")
            continue

        for path in sorted(folder.glob("*.jsonl")):
            files.append((group, path))

    return files


def load_latest_reports(jsonl_files: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Load reports and keep newest record per assembly name."""
    latest: dict[str, dict[str, Any]] = {}

    for group, path in jsonl_files:
        mtime = path.stat().st_mtime
        for row in iter_jsonl(path):
            assembly_name = str(row.get("assembly_name", ""))
            if not assembly_name:
                continue

            row["group"] = row.get("group", group)
            row["_source_jsonl"] = str(path)
            row["_source_mtime"] = float(mtime)

            prev = latest.get(assembly_name)
            if prev is None or float(mtime) >= float(prev.get("_source_mtime", -1.0)):
                latest[assembly_name] = row

    reports = list(latest.values())
    reports.sort(
        key=lambda r: (str(r.get("group", "")), str(r.get("assembly_name", "")))
    )
    return reports


def build_overview_requests(
    reports: list[dict[str, Any]],
    cases_root: Path,
    target_mode: str,
    final_failed_only: bool,
) -> list[OverviewRequest]:
    """
    Create overview render requests.

    target_mode:
        first-failed    -> one overview per case, using the first failed target
        failed-targets  -> one overview per unique failed target in each case
        all-targets     -> one overview per part later, target chosen after XML load
    """
    requests: list[OverviewRequest] = []

    for row in reports:
        assembly_name = str(row.get("assembly_name", ""))
        if not assembly_name:
            continue

        report_success = parse_bool(row.get("success", False))
        if final_failed_only and report_success:
            continue

        attempts = row.get("attempts", [])
        failed_attempts = [
            a for a in attempts if isinstance(a, dict) and attempt_failed(a)
        ]

        # "failure cases" means cases that either ended incomplete or contain at
        # least one failed attempt. This includes informative intermediate failures.
        if report_success and not failed_attempts:
            continue
        if not report_success and not failed_attempts:
            # Still render the final failed rollout once if attempts are missing.
            failed_attempts = []

        xml_path = resolve_xml_path(row.get("xml_path"), assembly_name, cases_root)
        source_jsonl = Path(str(row.get("_source_jsonl", "")))
        failed_count = len(failed_attempts)
        stop_reason = row.get("stop_reason")
        stop_reason_s = str(stop_reason) if stop_reason is not None else None
        completion = parse_float_or_none(row.get("completion"))
        group = str(row.get("group", "unknown"))

        if target_mode == "all-targets":
            requests.append(
                OverviewRequest(
                    group=group,
                    assembly_name=assembly_name,
                    xml_path=xml_path,
                    target_part=None,
                    attempt_index=None,
                    failed_attempt_count=failed_count,
                    report_success=report_success,
                    stop_reason=stop_reason_s,
                    completion=completion,
                    source_jsonl=source_jsonl,
                )
            )
            continue

        target_attempt_pairs: list[tuple[str | None, int | None]] = []
        for idx, attempt in enumerate(failed_attempts):
            target_attempt_pairs.append((get_attempt_target(attempt), idx))

        if not target_attempt_pairs:
            target_attempt_pairs = [(None, None)]

        if target_mode == "first-failed":
            target_attempt_pairs = target_attempt_pairs[:1]
        elif target_mode == "failed-targets":
            unique_pairs: list[tuple[str | None, int | None]] = []
            seen: set[str | None] = set()
            for target, idx in target_attempt_pairs:
                if target in seen:
                    continue
                seen.add(target)
                unique_pairs.append((target, idx))
            target_attempt_pairs = unique_pairs
        else:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

        for target, idx in target_attempt_pairs:
            requests.append(
                OverviewRequest(
                    group=group,
                    assembly_name=assembly_name,
                    xml_path=xml_path,
                    target_part=target,
                    attempt_index=idx,
                    failed_attempt_count=failed_count,
                    report_success=report_success,
                    stop_reason=stop_reason_s,
                    completion=completion,
                    source_jsonl=source_jsonl,
                )
            )

    return requests


def infer_group_from_assembly_name(assembly_name: str) -> str:
    """Infer result group from an assembly name when a manifest has no group column."""
    if assembly_name.startswith("joint_assembly_rotation"):
        return "joint_assembly_rotation"
    if assembly_name.startswith("joint_assembly"):
        return "joint_assembly"
    if assembly_name.startswith("multi_assembly"):
        return "multi_assembly"
    return "manifest"


def confidence_rank(value: str | None) -> int:
    order = {"possible": 1, "likely": 2, "high": 3}
    return order.get(str(value or "").strip().lower(), 0)


def load_overview_requests_from_manifest(
    manifest_path: Path,
    cases_root: Path,
    min_confidence: str | None = None,
) -> list[OverviewRequest]:
    """
    Load render requests directly from a CSV manifest.

    Supported inputs:
    1) thread_like_manifest_for_rendering.csv from screen_thread_like_multi_cases.py
       columns: assembly_name,target_part,confidence,score,relation,paired_part,xml_path,source_jsonl
    2) overview_manifest.csv produced by this renderer
       columns: group,assembly_name,target_part,...,xml_path,source_jsonl
    """
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    min_rank = confidence_rank(min_confidence)
    requests: list[OverviewRequest] = []
    seen: set[tuple[str, str | None]] = set()

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            assembly_name = str(row.get("assembly_name", "")).strip()
            if not assembly_name:
                print(f"[WARN] Manifest row {row_idx} has no assembly_name; skipped.")
                continue

            confidence = row.get("confidence")
            if min_rank and confidence_rank(confidence) < min_rank:
                continue

            target_part_raw = row.get("target_part")
            target_part = (
                str(target_part_raw).strip()
                if target_part_raw not in (None, "")
                else None
            )

            key = (assembly_name, target_part)
            if key in seen:
                continue
            seen.add(key)

            xml_raw = row.get("xml_path") or None
            xml_path = resolve_xml_path(xml_raw, assembly_name, cases_root)

            source_raw = row.get("source_jsonl") or ""
            source_jsonl = Path(source_raw) if source_raw else manifest_path

            group = str(
                row.get("group") or infer_group_from_assembly_name(assembly_name)
            )
            completion = parse_float_or_none(row.get("completion"))
            stop_reason_raw = row.get("stop_reason")
            stop_reason = (
                str(stop_reason_raw) if stop_reason_raw not in (None, "") else None
            )
            report_success = parse_bool(row.get("report_success", False))

            # attempt_index may be missing or non-numeric in thread-like manifests.
            attempt_index = None
            try:
                raw_attempt = row.get("attempt_index")
                if raw_attempt not in (None, ""):
                    attempt_index = int(float(raw_attempt))
            except Exception:
                attempt_index = None

            failed_count = 1
            try:
                raw_count = row.get("failed_attempt_count")
                if raw_count not in (None, ""):
                    failed_count = int(float(raw_count))
            except Exception:
                failed_count = 1

            requests.append(
                OverviewRequest(
                    group=group,
                    assembly_name=assembly_name,
                    xml_path=xml_path,
                    target_part=target_part,
                    attempt_index=attempt_index,
                    failed_attempt_count=failed_count,
                    report_success=report_success,
                    stop_reason=stop_reason,
                    completion=completion,
                    source_jsonl=source_jsonl,
                )
            )

    return requests


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------


def safe_filename(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def mesh_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    return np.asarray(mesh.vertices, dtype=float)


def mesh_triangles(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    return np.asarray(mesh.triangles, dtype=int)


def setup_3d(ax, title: str) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.45, 1.0, 0.78))
    ax.view_init(elev=23, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, pad=8, fontweight="bold", fontsize=12, color=DARK)


def add_mesh_to_axis(
    ax,
    mesh: o3d.geometry.TriangleMesh,
    color: str,
    alpha: float,
    edgecolor: str,
    linewidth: float,
    max_faces: int,
) -> np.ndarray:
    mesh.compute_vertex_normals()
    verts = mesh_vertices(mesh)
    tris = mesh_triangles(mesh)

    if tris.shape[0] > max_faces:
        idx = np.linspace(0, tris.shape[0] - 1, max_faces).round().astype(int)
        tris_draw = tris[idx]
    else:
        tris_draw = tris

    faces = [verts[t] for t in tris_draw]
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly)
    return verts


def set_equal_bounds(ax, points: np.ndarray, margin: float = 0.22) -> None:
    if points.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    span = hi - lo
    radius = 0.5 * max(float(np.max(span)), 1.0) * (1.0 + margin)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def choose_target_name(assembly, requested_target: str | None) -> str:
    if requested_target and assembly.get_part_by_name(requested_target) is not None:
        return requested_target
    if assembly.get_part_by_name("part_0") is not None:
        return "part_0"
    return next(iter(assembly.parts.keys()))


def render_assembly_overview(
    assembly,
    target_name: str,
    out_path_stem: Path,
    formats: tuple[str, ...],
    max_faces: int,
    title: str,
) -> list[Path]:
    out_path_stem.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.6, 6.9))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, title)

    all_points: list[np.ndarray] = []

    for part in assembly.parts.values():
        if part.name == target_name:
            color = KIT_YELLOW
            alpha = 0.72
            edge = "#8A6D00"
            lw = 0.12
        else:
            color = KIT_DEEP_BLUE
            alpha = 0.28
            edge = KIT_DEEP_BLUE
            lw = 0.08

        try:
            verts = add_mesh_to_axis(
                ax=ax,
                mesh=part.mesh,
                color=color,
                alpha=alpha,
                edgecolor=edge,
                linewidth=lw,
                max_faces=max_faces,
            )
            all_points.append(verts)
        except Exception as exc:
            print(f"[WARN] Could not render part {part.name}: {exc}")

    if all_points:
        set_equal_bounds(ax, np.vstack(all_points))

    ax.legend(
        handles=[
            Patch(
                facecolor=KIT_YELLOW,
                edgecolor="#8A6D00",
                alpha=0.72,
                label="target part",
            ),
            Patch(
                facecolor=KIT_DEEP_BLUE,
                edgecolor=KIT_DEEP_BLUE,
                alpha=0.28,
                label="other part",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    generated: list[Path] = []
    for fmt in formats:
        out_path = out_path_stem.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, facecolor=WHITE)
        generated.append(out_path)

    plt.close(fig)
    return generated


def write_manifest(rows: list[dict[str, Any]], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "assembly_name",
        "target_part",
        "attempt_index",
        "failed_attempt_count",
        "report_success",
        "stop_reason",
        "completion",
        "xml_path",
        "source_jsonl",
        "output_stem",
        "status",
        "error",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def limit_requests_per_case(
    requests: list[OverviewRequest],
    max_requests_per_case: int | None,
    prefer_unique_cases: bool,
) -> list[OverviewRequest]:
    if max_requests_per_case is None or max_requests_per_case <= 0:
        if not prefer_unique_cases:
            return requests
        max_requests_per_case = 1

    out: list[OverviewRequest] = []
    counts: dict[str, int] = {}

    for r in requests:
        count = counts.get(r.assembly_name, 0)
        if count >= max_requests_per_case:
            continue
        out.append(r)
        counts[r.assembly_name] = count + 1

    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render standalone assembly overview figures for external 2-part failure cases."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[3].joinpath("private/results"),
        help="Root folder containing result JSONL group folders.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["joint_assembly", "joint_assembly_rotation"],
        help="Result groups to scan under --results-root.",
    )
    parser.add_argument(
        "--jsonl",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit JSONL files. If provided, --results-root/--groups are ignored.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV manifest to render directly. Supports either the "
            "thread-like screening manifest columns (assembly_name,target_part,xml_path,source_jsonl,confidence,score,relation,paired_part) "
            "or this renderer's overview_manifest.csv columns. If provided, JSONL selection is skipped."
        ),
    )
    parser.add_argument(
        "--manifest-min-confidence",
        choices=["possible", "likely", "high"],
        default=None,
        help="Optional confidence filter for thread-like screening manifests.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("external/test_cases_external"),
        help="Root folder containing converted external test case folders.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root to add to sys.path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("private/figures/failure_cases_joint_assembly_cases"),
        help="Output root directory for rendered overview figures.",
    )

    parser.add_argument(
        "--target-mode",
        choices=["first-failed", "failed-targets", "all-targets"],
        default="first-failed",
        help=(
            "first-failed: one overview per case using first failed target; "
            "failed-targets: one overview per unique failed target; "
            "all-targets: render all parts as targets for selected failure cases."
        ),
    )
    parser.add_argument(
        "--final-failed-only",
        action="store_true",
        help="Render only final incomplete rollouts. By default, cases with intermediate failed attempts are also included.",
    )
    parser.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="Optional assembly names to render after selection.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of overview requests to render.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=3500,
        help="Maximum rendered faces per part for readability and speed.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save all figures directly under --out instead of grouped subfolders.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Assembly overview",
        help="Figure title used in each rendered overview.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List selected overview requests without rendering.",
    )
    parser.add_argument(
        "--max-requests-per-case",
        type=int,
        default=None,
        help="Limit how many overview requests are rendered per assembly/case.",
    )

    parser.add_argument(
        "--prefer-unique-cases",
        action="store_true",
        help="Shortcut for rendering at most one overview request per assembly/case.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_project_paths(args.project_root)

    results_root = args.results_root.expanduser().resolve()
    cases_root = args.cases_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    formats = tuple(args.formats)

    if not cases_root.exists():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")

    if args.manifest is not None:
        requests = load_overview_requests_from_manifest(
            manifest_path=args.manifest,
            cases_root=cases_root,
            min_confidence=args.manifest_min_confidence,
        )
        jsonl_files = []
        reports = []
        print(f"[INFO] Manifest: {args.manifest.expanduser().resolve()}")
        if args.manifest_min_confidence:
            print(
                f"[INFO] Manifest confidence filter: >= {args.manifest_min_confidence}"
            )
    else:
        if args.jsonl:
            jsonl_files = [("explicit", p.expanduser().resolve()) for p in args.jsonl]
        else:
            jsonl_files = find_result_jsonl_files(results_root, list(args.groups))

        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL result files found. results_root={results_root}, groups={args.groups}"
            )

        reports = load_latest_reports(jsonl_files)
        requests = build_overview_requests(
            reports=reports,
            cases_root=cases_root,
            target_mode=args.target_mode,
            final_failed_only=bool(args.final_failed_only),
        )

    if args.case:
        allowed = set(args.case)
        requests = [r for r in requests if r.assembly_name in allowed]

    requests = limit_requests_per_case(
        requests=requests,
        max_requests_per_case=args.max_requests_per_case,
        prefer_unique_cases=bool(args.prefer_unique_cases),
    )

    if args.max_cases is not None:
        requests = requests[: max(args.max_cases, 0)]

    if args.manifest is None:
        print(f"[INFO] JSONL files: {len(jsonl_files)}")
        print(f"[INFO] Latest reports: {len(reports)}")
    print(f"[INFO] Overview requests selected: {len(requests)}")

    if args.list_only:
        for r in requests:
            print(
                f"group={r.group} case={r.assembly_name} target={r.target_part} "
                f"failed_attempts={r.failed_attempt_count} success={r.report_success} "
                f"stop={r.stop_reason} completion={r.completion} xml={r.xml_path}"
            )
        return

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    generated_count = 0

    for idx, req in enumerate(requests, start=1):
        print(
            f"[RENDER] {idx}/{len(requests)} {req.group}/{req.assembly_name} "
            f"target={req.target_part or '[auto]'}"
        )

        row = {
            "group": req.group,
            "assembly_name": req.assembly_name,
            "target_part": req.target_part or "",
            "attempt_index": req.attempt_index if req.attempt_index is not None else "",
            "failed_attempt_count": req.failed_attempt_count,
            "report_success": req.report_success,
            "stop_reason": req.stop_reason or "",
            "completion": req.completion if req.completion is not None else "",
            "xml_path": str(req.xml_path),
            "source_jsonl": str(req.source_jsonl),
            "output_stem": "",
            "status": "",
            "error": "",
        }

        try:
            assembly = load_assembly_with_resolved_paths(req.xml_path)

            if args.target_mode == "all-targets":
                target_names = list(assembly.parts.keys())
            else:
                target_names = [choose_target_name(assembly, req.target_part)]

            for target_name in target_names:
                if args.flat:
                    stem = (
                        out_root
                        / f"{safe_filename(req.group)}__{safe_filename(req.assembly_name)}__target_{safe_filename(target_name)}__overview"
                    )
                else:
                    stem = (
                        out_root
                        / safe_filename(req.group)
                        / safe_filename(req.assembly_name)
                        / f"target_{safe_filename(target_name)}_overview"
                        / "00_assembly_overview"
                    )

                render_assembly_overview(
                    assembly=assembly,
                    target_name=target_name,
                    out_path_stem=stem,
                    formats=formats,
                    max_faces=args.max_faces,
                    title=args.title,
                )

                generated_count += len(formats)
                row_for_target = dict(row)
                row_for_target["target_part"] = target_name
                row_for_target["output_stem"] = str(stem)
                row_for_target["status"] = "ok"
                manifest_rows.append(row_for_target)

        except Exception as exc:
            print(f"[ERROR] {req.assembly_name}: {type(exc).__name__}: {exc}")
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            manifest_rows.append(row)

    manifest_path = out_root / "overview_manifest.csv"
    write_manifest(manifest_rows, manifest_path)

    print("=" * 80)
    print(f"[DONE] Rendered files: {generated_count}")
    print(f"[DONE] Output root: {out_root}")
    print(f"[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
