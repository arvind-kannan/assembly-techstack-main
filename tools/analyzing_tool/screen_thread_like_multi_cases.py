from __future__ import annotations

"""
Screen multi-assembly evaluation cases for thread-like / bolt-nut-like locking structures.

Purpose
-------
This tool is designed for thesis-scale post-hoc analysis of the large-scale
multi_assembly evaluation. It searches converted XML/STL assemblies, together
with rollout JSONL reports, for geometry patterns that may indicate missing
thread-aware or rotational-release capabilities.

It implements a conservative MVP screening strategy:

1. Focus on failed / incomplete multi-assembly cases by default.
2. Detect shaft-like parts and ring/nut/carrier-like hollow parts from STL mesh
   geometry using simple PCA and ray-cast descriptors.
3. Mark a detected pair as failure-relevant only when it appears in the final
   residual assembly or final failure frontier.

The output is not a proof that a case contains a true screw thread. It is a
screening estimate for thread-like or rotational-locking structures that may be
underestimated by a purely translational gripping abstraction.

Default paths match the current Masterarbeit project layout:

    Results:
        ../private/results/multi_assembly
    Converted cases:
        ../external/test_cases_external
    Output:
        ../private/results/thread_like_screening

Example
-------
cd <repo>

PYTHONPATH=src:$PYTHONPATH \
python tools/analyzing_tool/screen_thread_like_multi_cases.py

For a quick test:

python tools/analyzing_tool/screen_thread_like_multi_cases.py --max-cases 20

To scan all 1000 cases instead of incomplete cases only:

python tools/analyzing_tool/screen_thread_like_multi_cases.py --all-cases
"""

import argparse
import json
import math
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PartRecord:
    name: str
    mesh_path: Path


@dataclass
class AssemblyRecord:
    assembly_name: str
    xml_path: Path
    parts: dict[str, PartRecord]
    connections: list[tuple[str, str, str]]


@dataclass
class PartDescriptor:
    name: str
    mesh_path: str
    n_vertices: int
    center: list[float]
    axes: list[list[float]]
    extents: list[float]

    main_axis: list[float]
    short_axis: list[float]

    length_main: float
    radius_p50_main: float
    radius_p90_main: float
    radius_p95_main: float
    slenderness: float
    radial_circularity: float
    head_ratio: float
    shaft_score: float
    is_shaft_like: bool

    ring_axis: list[float]
    ring_outer_radius: float
    ring_inner_radius_est: float | None
    ring_hole_ratio: float | None
    ring_thickness_ratio: float
    ring_center_occupancy: float | None
    ring_score: float
    is_ring_like: bool


@dataclass
class PairCandidate:
    assembly_name: str
    success: bool
    completion: float | None
    stop_reason: str | None
    part_a: str
    part_b: str
    shaft_part: str
    ring_or_carrier_part: str
    connection_type: str

    score: float
    confidence: str
    failure_relevant: bool
    relation: str

    axis_alignment: float
    axis_distance: float
    axis_distance_norm: float
    shaft_slenderness: float
    shaft_radius: float
    ring_outer_radius: float
    ring_inner_radius_est: float | None
    radius_fit_score: float
    shaft_score: float
    ring_score: float
    shaft_head_ratio: float

    both_remaining: bool
    intersects_final_failure_frontier: bool
    remaining_parts_count: int | None
    final_failure_targets: str
    source_jsonl: str | None
    xml_path: str


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def add_project_paths(project_root: Path | None) -> None:
    candidates = []
    if project_root is not None:
        candidates.append(project_root)
        candidates.append(project_root / "src")
    candidates.append(Path.cwd())
    candidates.append(Path.cwd() / "src")

    for p in candidates:
        if p.exists():
            sp = str(p.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        return None
    return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                print(f"[WARN] Could not parse {path}:{line_idx}: {exc}")


def result_jsonl_files(results_root: Path, group: str) -> list[Path]:
    group_dir = results_root / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Result group folder not found: {group_dir}")
    return sorted(group_dir.glob("*.jsonl"))


def load_latest_reports(jsonl_files: list[Path]) -> list[dict]:
    """Load reports and keep newest row per assembly by JSONL modification time."""
    latest: dict[str, dict] = {}

    for path in jsonl_files:
        mtime = path.stat().st_mtime
        for row in iter_jsonl(path):
            assembly_name = str(row.get("assembly_name", ""))
            if not assembly_name:
                continue
            row["_source_jsonl"] = str(path)
            row["_source_mtime"] = mtime
            prev = latest.get(assembly_name)
            if prev is None or float(mtime) >= float(prev.get("_source_mtime", -1.0)):
                latest[assembly_name] = row

    rows = list(latest.values())
    rows.sort(key=lambda r: str(r.get("assembly_name", "")))
    return rows


def resolve_xml_path(raw_xml: str | None, assembly_name: str, cases_root: Path) -> Path:
    if raw_xml:
        p = Path(raw_xml)
        if p.exists():
            return p.resolve()

        parts = list(p.parts)
        if assembly_name in parts:
            idx = parts.index(assembly_name)
            tail = Path(*parts[idx:])
            candidate = cases_root / tail
            if candidate.exists():
                return candidate.resolve()

    return (cases_root / assembly_name / "xml" / f"{assembly_name}.xml").resolve()


def resolve_existing_file(path_str: str, xml_path: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()

    candidates = [
        xml_path.parent / path_str,
        xml_path.parent.parent / path_str,
        xml_path.parent / "stl" / Path(path_str).name,
        xml_path.parent.parent / "stl" / Path(path_str).name,
        xml_path.parent / "obj" / Path(path_str).name,
        xml_path.parent.parent / "obj" / Path(path_str).name,
        xml_path.parent / f"{xml_path.stem}_data" / Path(path_str).name,
        xml_path.parent.parent / f"{xml_path.stem}_data" / Path(path_str).name,
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return p


# ---------------------------------------------------------------------------
# Assembly / mesh loading
# ---------------------------------------------------------------------------


def load_assembly_record(xml_path: Path) -> AssemblyRecord:
    xml_path = xml_path.resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    root = ET.parse(str(xml_path)).getroot()
    assembly_name = root.attrib.get("name", xml_path.stem)

    parts: dict[str, PartRecord] = {}
    for part_elem in root.findall("part"):
        name = part_elem.attrib.get("name")
        mesh_raw = part_elem.attrib.get("mesh_path")
        if not name or not mesh_raw:
            continue
        parts[name] = PartRecord(
            name=name,
            mesh_path=resolve_existing_file(mesh_raw, xml_path),
        )

    connections: list[tuple[str, str, str]] = []
    for conn_elem in root.findall("connection"):
        p1 = conn_elem.attrib.get("part_name1")
        p2 = conn_elem.attrib.get("part_name2")
        ctype = conn_elem.attrib.get("type", "generic")
        if p1 and p2:
            connections.append((p1, p2, ctype))

    return AssemblyRecord(
        assembly_name=assembly_name,
        xml_path=xml_path,
        parts=parts,
        connections=connections,
    )


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {path}")
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def sample_vertices(vertices: np.ndarray, max_vertices: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
    if vertices.shape[0] <= max_vertices:
        return vertices
    idx = np.linspace(0, vertices.shape[0] - 1, max_vertices).round().astype(int)
    return vertices[idx]


# ---------------------------------------------------------------------------
# Geometry descriptors
# ---------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-12 or not np.isfinite(n):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    for i in range(3):
        vecs[:, i] = normalize(vecs[:, i])
    return center, vals, vecs


def extents_along_axes(
    points: np.ndarray, center: np.ndarray, axes: np.ndarray
) -> np.ndarray:
    local = (points - center.reshape(1, 3)) @ axes
    return local.max(axis=0) - local.min(axis=0)


def radial_distances_to_axis(
    points: np.ndarray, center: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    axis = normalize(axis)
    rel = points - center.reshape(1, 3)
    axial = rel @ axis
    closest = center.reshape(1, 3) + axial.reshape(-1, 1) * axis.reshape(1, 3)
    return np.linalg.norm(points - closest, axis=1)


def radius_profile_head_ratio(
    points: np.ndarray, center: np.ndarray, axis: np.ndarray, bins: int = 10
) -> float:
    axis = normalize(axis)
    rel = points - center.reshape(1, 3)
    x = rel @ axis
    if x.max() - x.min() <= 1e-9:
        return 1.0

    radii = radial_distances_to_axis(points, center, axis)
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    per_bin: list[float] = []
    for i in range(bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1])
        if np.count_nonzero(mask) < 3:
            per_bin.append(float("nan"))
        else:
            per_bin.append(float(np.percentile(radii[mask], 90)))

    arr = np.asarray(per_bin, dtype=float)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) < 4:
        return 1.0

    mid = arr[bins // 3 : 2 * bins // 3]
    mid = mid[np.isfinite(mid)]
    end = np.r_[arr[: max(1, bins // 5)], arr[-max(1, bins // 5) :]]
    end = end[np.isfinite(end)]
    if mid.size == 0 or end.size == 0:
        return 1.0

    mid_radius = float(np.median(mid))
    end_radius = float(np.max(end))
    if mid_radius <= 1e-9:
        return 1.0
    return float(end_radius / mid_radius)


def make_raycast_scene(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def estimate_center_occupancy(
    scene: o3d.t.geometry.RaycastingScene, center: np.ndarray
) -> float | None:
    try:
        query = o3d.core.Tensor(
            np.asarray(center, dtype=np.float32).reshape(1, 3),
            dtype=o3d.core.Dtype.Float32,
        )
        occ = scene.compute_occupancy(query).numpy()
        return float(occ.reshape(-1)[0])
    except Exception:
        return None


def estimate_hole_radius_by_raycast(
    scene: o3d.t.geometry.RaycastingScene,
    center: np.ndarray,
    axis: np.ndarray,
    outer_radius: float,
    n_dirs: int = 16,
) -> tuple[float | None, float | None]:
    """
    Estimate distance from part center to first surface hit in directions
    perpendicular to `axis`. For hollow/ring-like parts this approximates an
    inner radius; for solid parts it approximates outer radius, so occupancy is
    also used by the descriptor.
    """
    if outer_radius <= 1e-9 or not np.isfinite(outer_radius):
        return None, None

    axis = normalize(axis)
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, [1, 0, 0]))) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = normalize(np.cross(axis, helper))
    v = normalize(np.cross(axis, u))

    origins = []
    dirs = []
    for i in range(n_dirs):
        theta = 2.0 * math.pi * i / n_dirs
        d = math.cos(theta) * u + math.sin(theta) * v
        origins.append(center)
        dirs.append(normalize(d))

    rays = np.hstack(
        [
            np.asarray(origins, dtype=np.float32),
            np.asarray(dirs, dtype=np.float32),
        ]
    )
    try:
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t = ans["t_hit"].numpy().astype(float)
        t = t[np.isfinite(t)]
        t = t[(t > 1e-6) & (t < outer_radius * 2.5)]
        if t.size == 0:
            return None, None
        inner = float(np.percentile(t, 20))
        ratio = float(inner / outer_radius) if outer_radius > 1e-9 else None
        return inner, ratio
    except Exception:
        return None, None


def score_clip(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def describe_part(part: PartRecord, max_vertices: int = 8000) -> PartDescriptor:
    mesh = load_mesh(part.mesh_path)
    vertices_full = np.asarray(mesh.vertices, dtype=float)
    vertices = sample_vertices(vertices_full, max_vertices)

    center, _, axes = pca_axes(vertices)
    extents = extents_along_axes(vertices, center, axes)

    main_axis = axes[:, 0]
    short_axis = axes[:, 2]

    r = radial_distances_to_axis(vertices, center, main_axis)
    r50 = float(np.percentile(r, 50))
    r90 = float(np.percentile(r, 90))
    r95 = float(np.percentile(r, 95))
    length = float(extents[0])
    radial_extent_a = float(extents[1])
    radial_extent_b = float(extents[2])
    radial_circ = float(
        min(radial_extent_a, radial_extent_b)
        / max(radial_extent_a, radial_extent_b, 1e-9)
    )
    slenderness = float(length / max(2.0 * r95, 1e-9))
    head_ratio = radius_profile_head_ratio(vertices, center, main_axis)

    # Shaft score: conservative but not too brittle.
    shaft_score = (
        0.45 * score_clip(slenderness, 2.2, 5.0)
        + 0.25 * score_clip(radial_circ, 0.45, 0.90)
        + 0.20 * score_clip(head_ratio, 1.10, 1.80)
        + 0.10 * score_clip(length, 5.0, 60.0)
    )
    is_shaft_like = bool(
        slenderness >= 2.6 and radial_circ >= 0.35 and shaft_score >= 0.38
    )

    # Ring/nut-like descriptor around the shortest PCA axis.
    ring_axis = short_axis
    rel = vertices - center.reshape(1, 3)
    axial_short = rel @ ring_axis
    radial_ring = np.linalg.norm(
        rel - axial_short.reshape(-1, 1) * ring_axis.reshape(1, 3), axis=1
    )
    outer_radius = float(np.percentile(radial_ring, 95))
    thickness = float(extents[2])
    thickness_ratio = float(thickness / max(2.0 * outer_radius, 1e-9))

    inner_est = None
    hole_ratio = None
    occupancy = None
    try:
        scene = make_raycast_scene(mesh)
        occupancy = estimate_center_occupancy(scene, center)
        inner_est, hole_ratio = estimate_hole_radius_by_raycast(
            scene, center, ring_axis, outer_radius
        )
    except Exception:
        pass

    hole_score = 0.0
    if hole_ratio is not None:
        # A non-trivial inner radius, but not almost equal to outer radius.
        hole_score = max(
            hole_score,
            score_clip(hole_ratio, 0.08, 0.45)
            * (1.0 - 0.35 * score_clip(hole_ratio, 0.75, 1.00)),
        )
    if occupancy is not None and occupancy < 0.5:
        # Center is likely empty. This is strong evidence for a ring/hole.
        hole_score = max(hole_score, 0.75)

    planar_score = score_clip(float(extents[0] / max(extents[1], 1e-9)), 0.65, 1.25)
    thickness_score = 1.0 - score_clip(thickness_ratio, 0.70, 1.60)
    ring_score = (
        0.45 * hole_score
        + 0.25 * thickness_score
        + 0.15 * planar_score
        + 0.15 * score_clip(outer_radius, 1.0, 25.0)
    )
    is_ring_like = bool(ring_score >= 0.42 and outer_radius > 1e-6)

    return PartDescriptor(
        name=part.name,
        mesh_path=str(part.mesh_path),
        n_vertices=int(vertices_full.shape[0]),
        center=center.tolist(),
        axes=axes.T.tolist(),
        extents=extents.tolist(),
        main_axis=main_axis.tolist(),
        short_axis=short_axis.tolist(),
        length_main=length,
        radius_p50_main=r50,
        radius_p90_main=r90,
        radius_p95_main=r95,
        slenderness=slenderness,
        radial_circularity=radial_circ,
        head_ratio=head_ratio,
        shaft_score=float(shaft_score),
        is_shaft_like=is_shaft_like,
        ring_axis=ring_axis.tolist(),
        ring_outer_radius=outer_radius,
        ring_inner_radius_est=inner_est,
        ring_hole_ratio=hole_ratio,
        ring_thickness_ratio=thickness_ratio,
        ring_center_occupancy=occupancy,
        ring_score=float(ring_score),
        is_ring_like=is_ring_like,
    )


def axis_distance(
    c1: np.ndarray, a1: np.ndarray, c2: np.ndarray, a2: np.ndarray
) -> float:
    a1 = normalize(a1)
    a2 = normalize(a2)
    delta = c2 - c1
    cross = np.cross(a1, a2)
    n = float(np.linalg.norm(cross))
    if n < 1e-9:
        return float(np.linalg.norm(np.cross(delta, a1)))
    return float(abs(np.dot(delta, cross)) / n)


def classify_pair(
    assembly_name: str,
    row: dict,
    asm: AssemblyRecord,
    desc_a: PartDescriptor,
    desc_b: PartDescriptor,
    connection_type: str,
    remaining_parts: set[str] | None,
    final_failure_targets: set[str],
) -> PairCandidate | None:
    """Return a pair candidate if the pair is at least possibly thread-like."""
    candidates = []

    for shaft, ring in [(desc_a, desc_b), (desc_b, desc_a)]:
        if shaft.shaft_score < 0.30 or ring.ring_score < 0.25:
            continue

        c_shaft = np.asarray(shaft.center, dtype=float)
        c_ring = np.asarray(ring.center, dtype=float)
        a_shaft = np.asarray(shaft.main_axis, dtype=float)
        a_ring = np.asarray(ring.ring_axis, dtype=float)

        align = abs(float(np.dot(normalize(a_shaft), normalize(a_ring))))
        dist = axis_distance(c_shaft, a_shaft, c_ring, a_ring)

        inner = ring.ring_inner_radius_est
        outer = ring.ring_outer_radius
        shaft_r = shaft.radius_p90_main

        # Distance normalized by the plausible available hole radius or ring size.
        ref_radius = max(
            shaft_r * 2.0,
            (inner if inner is not None else 0.0) * 0.75,
            outer * 0.25,
            1e-6,
        )
        dist_norm = float(dist / ref_radius)
        dist_score = 1.0 - score_clip(dist_norm, 0.6, 1.8)

        if inner is not None and inner > 1e-9:
            fit_ratio = float(shaft_r / inner)
            # Good when shaft radius is smaller than estimated hole radius,
            # but not tiny. Conservative tolerance allows noisy meshes.
            radius_fit_score = 1.0 - score_clip(fit_ratio, 0.90, 1.45)
        else:
            fit_ratio = None
            radius_fit_score = 1.0 - score_clip(shaft_r / max(outer, 1e-9), 0.22, 0.55)

        align_score = score_clip(align, 0.65, 0.95)

        score = (
            0.28 * shaft.shaft_score
            + 0.25 * ring.ring_score
            + 0.20 * align_score
            + 0.17 * dist_score
            + 0.10 * radius_fit_score
        )

        # Relation label.
        if ring.is_ring_like and inner is not None:
            relation = "shaft_through_ring_or_nut"
        elif ring.is_ring_like:
            relation = "shaft_aligned_with_hollow_or_ring_like_part"
        else:
            relation = "possible_coaxial_rotational_lock"

        candidates.append(
            (score, relation, shaft, ring, align, dist, dist_norm, radius_fit_score)
        )

    if not candidates:
        return None

    score, relation, shaft, ring, align, dist, dist_norm, radius_fit_score = max(
        candidates, key=lambda x: x[0]
    )

    # Conservative confidence levels.
    key_geometry_ok = (
        shaft.is_shaft_like
        and align >= 0.72
        and dist_norm <= 1.35
        and radius_fit_score >= 0.25
    )

    if score >= 0.72 and key_geometry_ok and ring.ring_score >= 0.42:
        confidence = "high"
    elif score >= 0.55 and key_geometry_ok:
        confidence = "likely"
    elif score >= 0.38:
        confidence = "possible"
    else:
        return None

    success = parse_bool(row.get("success", False))
    completion = safe_float(row.get("completion"))
    stop_reason = row.get("stop_reason")

    if remaining_parts is None:
        both_remaining = False
        remaining_count = None
    else:
        both_remaining = shaft.name in remaining_parts and ring.name in remaining_parts
        remaining_count = len(remaining_parts)

    intersects_frontier = bool(
        {shaft.name, ring.name}.intersection(final_failure_targets)
    )
    failure_relevant = bool(
        (not success)
        and both_remaining
        and (intersects_frontier or not final_failure_targets)
    )

    return PairCandidate(
        assembly_name=assembly_name,
        success=success,
        completion=completion,
        stop_reason=str(stop_reason) if stop_reason is not None else None,
        part_a=desc_a.name,
        part_b=desc_b.name,
        shaft_part=shaft.name,
        ring_or_carrier_part=ring.name,
        connection_type=connection_type,
        score=float(score),
        confidence=confidence,
        failure_relevant=failure_relevant,
        relation=relation,
        axis_alignment=float(align),
        axis_distance=float(dist),
        axis_distance_norm=float(dist_norm),
        shaft_slenderness=float(shaft.slenderness),
        shaft_radius=float(shaft.radius_p90_main),
        ring_outer_radius=float(ring.ring_outer_radius),
        ring_inner_radius_est=ring.ring_inner_radius_est,
        radius_fit_score=float(radius_fit_score),
        shaft_score=float(shaft.shaft_score),
        ring_score=float(ring.ring_score),
        shaft_head_ratio=float(shaft.head_ratio),
        both_remaining=both_remaining,
        intersects_final_failure_frontier=intersects_frontier,
        remaining_parts_count=remaining_count,
        final_failure_targets=",".join(sorted(final_failure_targets)),
        source_jsonl=row.get("_source_jsonl"),
        xml_path=str(asm.xml_path),
    )


# ---------------------------------------------------------------------------
# Rollout report parsing
# ---------------------------------------------------------------------------


def get_remaining_parts(row: dict) -> set[str] | None:
    for key in ["remaining_parts", "final_remaining_parts"]:
        value = row.get(key)
        if isinstance(value, list):
            return {str(v) for v in value}
    return None


def attempt_success(attempt: dict) -> bool:
    return parse_bool(attempt.get("success", False))


def attempt_target(attempt: dict) -> str | None:
    debug = attempt.get("debug", {}) or {}
    for key in ["target_part", "target"]:
        value = debug.get(key)
        if value:
            return str(value)
    value = attempt.get("target")
    return str(value) if value else None


def final_failure_targets(row: dict) -> set[str]:
    attempts = row.get("attempts", [])
    if not isinstance(attempts, list):
        return set()

    last_success_idx = -1
    for i, attempt in enumerate(attempts):
        if attempt_success(attempt):
            last_success_idx = i

    targets: set[str] = set()
    for attempt in attempts[last_success_idx + 1 :]:
        if not attempt_success(attempt):
            t = attempt_target(attempt)
            if t:
                targets.add(t)

    # Fallback: all failed targets.
    if not targets:
        for attempt in attempts:
            if not attempt_success(attempt):
                t = attempt_target(attempt)
                if t:
                    targets.add(t)

    return targets


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def connected_pair_types(asm: AssemblyRecord) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for p1, p2, ctype in asm.connections:
        out[pair_key(p1, p2)] = ctype
    return out


def pairs_to_scan(
    asm: AssemblyRecord,
    part_names: list[str],
    descriptors: dict[str, PartDescriptor],
    include_all_candidate_pairs: bool,
) -> list[tuple[str, str, str]]:
    conn_types = connected_pair_types(asm)
    selected = set(part_names)
    pairs: dict[tuple[str, str], str] = {}

    # XML reconstructed connections are the primary source.
    for key, ctype in conn_types.items():
        if key[0] in selected and key[1] in selected:
            pairs[key] = ctype

    # Add shaft-like vs ring-like candidate pairs, because a semantic screw/nut
    # relation can have a small clearance and may not always be reconstructed as
    # contact in the XML.
    if include_all_candidate_pairs:
        shaft_names = [n for n in part_names if descriptors[n].shaft_score >= 0.30]
        ring_names = [n for n in part_names if descriptors[n].ring_score >= 0.25]
        for s in shaft_names:
            for r in ring_names:
                if s == r:
                    continue
                key = pair_key(s, r)
                if key not in pairs:
                    pairs[key] = "not_xml_connected_candidate"

    return [(a, b, ctype) for (a, b), ctype in sorted(pairs.items())]


# ---------------------------------------------------------------------------
# Case analysis
# ---------------------------------------------------------------------------


def analyze_case(
    row: dict,
    cases_root: Path,
    residual_only: bool,
    include_all_candidate_pairs: bool,
    max_vertices: int,
) -> tuple[dict, list[PairCandidate], list[dict]]:
    assembly_name = str(row.get("assembly_name", ""))
    if not assembly_name:
        raise ValueError("row has no assembly_name")

    xml_path = resolve_xml_path(row.get("xml_path"), assembly_name, cases_root)
    asm = load_assembly_record(xml_path)

    remaining = get_remaining_parts(row)
    frontier_targets = final_failure_targets(row)

    if residual_only and remaining:
        part_names = [n for n in sorted(remaining) if n in asm.parts]
    else:
        part_names = sorted(asm.parts.keys())

    if len(part_names) < 2:
        return make_case_summary(row, asm, 0, None, remaining, frontier_targets), [], []

    descriptors: dict[str, PartDescriptor] = {}
    descriptor_rows: list[dict] = []

    for name in part_names:
        desc = describe_part(asm.parts[name], max_vertices=max_vertices)
        descriptors[name] = desc
        descriptor_rows.append(asdict(desc))

    pair_candidates: list[PairCandidate] = []
    for a, b, ctype in pairs_to_scan(
        asm, part_names, descriptors, include_all_candidate_pairs
    ):
        cand = classify_pair(
            assembly_name=assembly_name,
            row=row,
            asm=asm,
            desc_a=descriptors[a],
            desc_b=descriptors[b],
            connection_type=ctype,
            remaining_parts=remaining,
            final_failure_targets=frontier_targets,
        )
        if cand is not None:
            pair_candidates.append(cand)

    max_pair = max(pair_candidates, key=lambda p: p.score) if pair_candidates else None
    summary = make_case_summary(
        row, asm, len(part_names), max_pair, remaining, frontier_targets
    )
    summary["num_detected_pairs"] = len(pair_candidates)
    summary["num_failure_relevant_pairs"] = sum(
        1 for p in pair_candidates if p.failure_relevant
    )
    summary["num_high_confidence_pairs"] = sum(
        1 for p in pair_candidates if p.confidence == "high"
    )
    summary["num_likely_or_high_pairs"] = sum(
        1 for p in pair_candidates if p.confidence in {"likely", "high"}
    )

    return summary, pair_candidates, descriptor_rows


def make_case_summary(
    row: dict,
    asm: AssemblyRecord,
    scanned_part_count: int,
    max_pair: PairCandidate | None,
    remaining: set[str] | None,
    frontier_targets: set[str],
) -> dict:
    success = parse_bool(row.get("success", False))
    completion = safe_float(row.get("completion"))

    if max_pair is None:
        max_score = 0.0
        max_conf = "none"
        failure_relevant = False
        relation = "none"
        shaft = ""
        ring = ""
    else:
        max_score = max_pair.score
        max_conf = max_pair.confidence
        failure_relevant = max_pair.failure_relevant
        relation = max_pair.relation
        shaft = max_pair.shaft_part
        ring = max_pair.ring_or_carrier_part

    return {
        "assembly_name": row.get("assembly_name"),
        "success": success,
        "completion": completion,
        "stop_reason": row.get("stop_reason"),
        "initial_num_parts": row.get("initial_num_parts"),
        "initial_num_connections": row.get("initial_num_connections"),
        "remaining_parts_count": len(remaining) if remaining is not None else None,
        "scanned_part_count": scanned_part_count,
        "final_failure_targets": ",".join(sorted(frontier_targets)),
        "max_thread_like_score": max_score,
        "max_confidence": max_conf,
        "failure_relevant_thread_like": failure_relevant,
        "max_relation": relation,
        "max_shaft_part": shaft,
        "max_ring_or_carrier_part": ring,
        "source_jsonl": row.get("_source_jsonl"),
        "xml_path": str(asm.xml_path),
        "num_detected_pairs": 0,
        "num_failure_relevant_pairs": 0,
        "num_high_confidence_pairs": 0,
        "num_likely_or_high_pairs": 0,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def confidence_rank(conf: str) -> int:
    return {"none": 0, "possible": 1, "likely": 2, "high": 3}.get(conf, 0)


def summarize_results(
    all_rows: list[dict], processed_summaries: list[dict], pair_rows: list[dict]
) -> dict:
    total_cases = len(all_rows)
    baseline_success = sum(1 for r in all_rows if parse_bool(r.get("success", False)))
    baseline_incomplete = total_cases - baseline_success

    processed = len(processed_summaries)

    def count_cases(predicate) -> int:
        names = {r["assembly_name"] for r in processed_summaries if predicate(r)}
        return len(names)

    possible_cases = count_cases(
        lambda r: confidence_rank(str(r.get("max_confidence"))) >= 1
    )
    likely_cases = count_cases(
        lambda r: confidence_rank(str(r.get("max_confidence"))) >= 2
    )
    high_cases = count_cases(
        lambda r: confidence_rank(str(r.get("max_confidence"))) >= 3
    )

    failure_relevant_possible = count_cases(
        lambda r: (
            bool(r.get("failure_relevant_thread_like"))
            and confidence_rank(str(r.get("max_confidence"))) >= 1
        )
    )
    failure_relevant_likely = count_cases(
        lambda r: (
            bool(r.get("failure_relevant_thread_like"))
            and confidence_rank(str(r.get("max_confidence"))) >= 2
        )
    )
    failure_relevant_high = count_cases(
        lambda r: (
            bool(r.get("failure_relevant_thread_like"))
            and confidence_rank(str(r.get("max_confidence"))) >= 3
        )
    )

    def rate_with(extra: int) -> float | None:
        if total_cases <= 0:
            return None
        return float((baseline_success + extra) / total_cases)

    return {
        "total_latest_multi_assembly_reports": total_cases,
        "baseline_successful_rollouts": baseline_success,
        "baseline_incomplete_rollouts": baseline_incomplete,
        "baseline_completion_rate": rate_with(0),
        "processed_cases": processed,
        "detected_thread_like_cases_possible_or_above": possible_cases,
        "detected_thread_like_cases_likely_or_above": likely_cases,
        "detected_thread_like_cases_high_confidence": high_cases,
        "failure_relevant_thread_like_cases_possible_or_above": failure_relevant_possible,
        "failure_relevant_thread_like_cases_likely_or_above": failure_relevant_likely,
        "failure_relevant_thread_like_cases_high_confidence": failure_relevant_high,
        "upper_completion_rate_if_possible_failure_relevant_cases_solved": rate_with(
            failure_relevant_possible
        ),
        "upper_completion_rate_if_likely_failure_relevant_cases_solved": rate_with(
            failure_relevant_likely
        ),
        "upper_completion_rate_if_high_confidence_failure_relevant_cases_solved": rate_with(
            failure_relevant_high
        ),
        "num_pair_candidates": len(pair_rows),
        "interpretation_note": (
            "These adjusted rates are upper estimates of affected cases, not predicted performance. "
            "The screening identifies thread-like or rotational-locking geometry that may require future screw-aware, "
            "rotational-release, or dependency-aware capabilities."
        ),
    }


def make_render_manifest(
    pair_candidates: list[PairCandidate], max_rows: int = 200
) -> list[dict]:
    ranked = sorted(
        [p for p in pair_candidates if p.failure_relevant],
        key=lambda p: (confidence_rank(p.confidence), p.score),
        reverse=True,
    )
    rows = []
    seen = set()
    for p in ranked:
        key = (p.assembly_name, p.shaft_part)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "assembly_name": p.assembly_name,
                "target_part": p.shaft_part,
                "confidence": p.confidence,
                "score": p.score,
                "relation": p.relation,
                "paired_part": p.ring_or_carrier_part,
                "xml_path": p.xml_path,
                "source_jsonl": p.source_jsonl,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen multi-assembly cases for thread-like / bolt-nut-like structures."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[3].joinpath("private/results"),
        help="Root folder containing JSONL result folders.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="multi_assembly",
        help="Result group to analyze.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("external/test_cases_external"),
        help="Root folder containing converted external test cases.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root added to sys.path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("private/results/thread_like_screening"),
        help="Output directory for CSV/JSON files.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Scan all latest reports. Default scans only incomplete / failed cases for completion-rate impact estimation.",
    )
    parser.add_argument(
        "--full-assembly",
        action="store_true",
        help="Scan the full assembly instead of the remaining residual parts when remaining_parts are available.",
    )
    parser.add_argument(
        "--xml-connected-only",
        action="store_true",
        help="Only scan XML-connected pairs. Default also scans shaft-like vs ring-like candidate pairs not connected in XML.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of reports to process for debugging.",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=8000,
        help="Maximum vertices sampled per part for descriptor computation.",
    )
    parser.add_argument(
        "--write-part-descriptors",
        action="store_true",
        help="Also write per-part descriptor CSV. This can be large.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_project_paths(args.project_root)

    results_root = args.results_root.expanduser().resolve()
    cases_root = args.cases_root.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = result_jsonl_files(results_root, args.group)
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found under {results_root / args.group}"
        )

    all_rows = load_latest_reports(jsonl_files)

    if args.all_cases:
        rows_to_process = list(all_rows)
    else:
        rows_to_process = [
            r for r in all_rows if not parse_bool(r.get("success", False))
        ]

    if args.max_cases is not None:
        rows_to_process = rows_to_process[: max(0, args.max_cases)]

    print(f"[INFO] Loaded latest reports: {len(all_rows)}")
    print(f"[INFO] Processing reports:    {len(rows_to_process)}")
    print(f"[INFO] Results root:          {results_root}")
    print(f"[INFO] Cases root:            {cases_root}")
    print(f"[INFO] Output:                {out_dir}")

    case_summaries: list[dict] = []
    pair_candidates: list[PairCandidate] = []
    pair_rows: list[dict] = []
    part_descriptor_rows: list[dict] = []
    errors: list[dict] = []

    for idx, row in enumerate(rows_to_process, start=1):
        name = str(row.get("assembly_name", f"case_{idx}"))
        print(f"[{idx:04d}/{len(rows_to_process):04d}] {name}")
        try:
            summary, pairs, desc_rows = analyze_case(
                row=row,
                cases_root=cases_root,
                residual_only=not args.full_assembly,
                include_all_candidate_pairs=not args.xml_connected_only,
                max_vertices=args.max_vertices,
            )
            case_summaries.append(summary)
            pair_candidates.extend(pairs)
            pair_rows.extend([asdict(p) for p in pairs])
            if args.write_part_descriptors:
                for d in desc_rows:
                    d["assembly_name"] = name
                    part_descriptor_rows.append(d)
        except Exception as exc:
            errors.append(
                {
                    "assembly_name": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "source_jsonl": row.get("_source_jsonl"),
                }
            )
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")

    summary = summarize_results(all_rows, case_summaries, pair_rows)

    write_csv(out_dir / "thread_like_cases.csv", case_summaries)
    write_csv(out_dir / "thread_like_pairs.csv", pair_rows)
    write_csv(
        out_dir / "thread_like_manifest_for_rendering.csv",
        make_render_manifest(pair_candidates),
    )

    if args.write_part_descriptors:
        write_csv(out_dir / "thread_like_part_descriptors.csv", part_descriptor_rows)

    (out_dir / "thread_like_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if errors:
        (out_dir / "thread_like_errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[SAVED]")
    print(out_dir / "thread_like_cases.csv")
    print(out_dir / "thread_like_pairs.csv")
    print(out_dir / "thread_like_manifest_for_rendering.csv")
    print(out_dir / "thread_like_summary.json")
    if errors:
        print(f"[WARN] Errors: {len(errors)} -> {out_dir / 'thread_like_errors.json'}")


if __name__ == "__main__":
    main()
