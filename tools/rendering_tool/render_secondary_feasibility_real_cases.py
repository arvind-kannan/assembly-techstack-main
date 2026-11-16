from __future__ import annotations

"""
Render real rotational-joint-style secondary feasibility check figures.

This script reads the external-case JSONL result log, finds gripping candidate
failures rejected by the projected-footprint ray-bundle feasibility check, loads
that case's XML assembly, and renders four thesis-style figures:

01_local_motion_vs_blocked_path
02_projection_plane_and_footprint
03_2d_footprint_sampling
04_ray_bundle_blocking_hits

It intentionally does NOT render the pipeline diagram.

Recommended local run:

python tools/rendering_tool/render_secondary_feasibility_real_cases.py \
  --zoom-target \
  --short-rays \
  --show-only-blocking-neighborhood

Optional specific case:

python tools/rendering_tool/render_secondary_feasibility_real_cases.py \
  --case joint_assembly_rotation_general_11356 \
  --candidate-index -1 \
  --zoom-target \
  --short-rays \
  --show-only-blocking-neighborhood
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Circle, Patch
from matplotlib.patches import Polygon as Polygon2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# -----------------------------------------------------------------------------
# Project import bootstrap
# -----------------------------------------------------------------------------


def add_project_paths(project_root: Path | None) -> None:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(project_root)

    # Common case when this file is stored under pomdp_disassembly_gym/tools.
    candidates.append(Path.cwd())
    candidates.append(Path.cwd().parent)

    for p in candidates:
        if p and p.exists():
            sp = str(p.resolve())
            if sp not in sys.path:
                sys.path.insert(0, sp)


# Imports are intentionally delayed until after add_project_paths() in main.


# -----------------------------------------------------------------------------
# Visual style: KIT / wbk
# -----------------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838f"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
KIT_GREY = "#D9DEE2"
DARK = "#222222"
MUTED = "#666666"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


@dataclass
class FailureCandidate:
    assembly_name: str
    xml_path: Path
    attempt_index: int
    target_part: str
    candidate_index: int
    direction: np.ndarray
    first_blocking_geometry: str | None
    json_summary: dict

    @property
    def json_first_blocking_t(self) -> float | None:
        value = self.json_summary.get("first_blocking_t")
        try:
            if value is None:
                return None
            value_f = float(value)
            if np.isfinite(value_f):
                return value_f
        except Exception:
            return None
        return None

    @property
    def json_num_blocking_hits(self) -> int | None:
        value = self.json_summary.get("num_blocking_hits")
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None


@dataclass
class RayBundleResult:
    center_world: np.ndarray
    direction: np.ndarray
    u: np.ndarray
    v: np.ndarray

    ray_length: float
    effective_ray_length: float
    origin_offset: float
    hit_margin: float

    pts2: np.ndarray
    hull2: np.ndarray
    origins2: np.ndarray
    origins_world: np.ndarray

    t_hit: np.ndarray
    geometry_ids: np.ndarray | None
    geometry_names: dict[int, str]
    hit_mask: np.ndarray
    blocking_mask: np.ndarray

    assembly_diag: float
    target_diag: float
    footprint_diag: float

    @property
    def first_blocking_t_total(self) -> float | None:
        if not np.any(self.blocking_mask):
            return None
        idx = int(np.flatnonzero(self.blocking_mask)[0])
        return float(self.t_hit[idx] + self.origin_offset)

    @property
    def blocking_geometry_names(self) -> set[str]:
        names: set[str] = set()
        if self.geometry_ids is None:
            return names
        for idx in np.flatnonzero(self.blocking_mask):
            gid = int(self.geometry_ids[int(idx)])
            if gid in self.geometry_names:
                names.add(self.geometry_names[gid])
        return names


@dataclass
class TeachingViewConfig:
    zoom_target: bool
    short_rays: bool
    show_only_blocking_neighborhood: bool
    focus_footprint: bool

    display_ray_length: float
    neighborhood_x_max: float
    marker_size_ray_origin_2d: float
    marker_size_center_2d: float
    marker_size_hit_3d: float
    title_fontsize: float
    legend_fontsize: float
    mesh_max_faces: int


# -----------------------------------------------------------------------------
# JSONL parsing and candidate selection
# -----------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_xml_path(raw_xml: str | None, assembly_name: str, cases_root: Path) -> Path:
    if raw_xml:
        p = Path(raw_xml)
        if p.exists():
            return p

        # Replace any old prefix with current cases_root by locating assembly name.
        parts = list(p.parts)
        if assembly_name in parts:
            idx = parts.index(assembly_name)
            tail = Path(*parts[idx:])
            candidate = cases_root / tail
            if candidate.exists():
                return candidate

    # Fallback expected layout.
    return cases_root / assembly_name / "xml" / f"{assembly_name}.xml"


def extract_failure_candidates(jsonl: Path, cases_root: Path) -> list[FailureCandidate]:
    out: list[FailureCandidate] = []

    for row in iter_jsonl(jsonl):
        assembly_name = str(row.get("assembly_name", ""))
        if not assembly_name:
            continue

        xml_path = resolve_xml_path(row.get("xml_path"), assembly_name, cases_root)

        for ai, attempt in enumerate(row.get("attempts", [])):
            debug = attempt.get("debug", {}) or {}
            target_part = str(debug.get("target_part") or attempt.get("target") or "")

            details = debug.get("exec_fail_details", {}) or {}
            failures = (
                details.get("candidate_failures")
                or debug.get("operation_data", {}).get("candidate_failures")
                or []
            )

            for ci, cf in enumerate(failures):
                if (
                    cf.get("feasibility_reason")
                    != "removal_path_projected_footprint_hit"
                ):
                    continue
                if cf.get("local_translation_feasible") is not True:
                    continue

                direction = np.asarray(cf.get("direction", []), dtype=float)
                if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                    continue

                n = np.linalg.norm(direction)
                if n < 1e-12:
                    continue

                out.append(
                    FailureCandidate(
                        assembly_name=assembly_name,
                        xml_path=xml_path,
                        attempt_index=ai,
                        target_part=target_part,
                        candidate_index=ci,
                        direction=direction / n,
                        first_blocking_geometry=cf.get("first_blocking_geometry"),
                        json_summary=cf,
                    )
                )

    return out


def candidate_score_for_teaching(
    c: FailureCandidate, min_first_blocking_t: float
) -> tuple[int, float, int]:
    """
    Higher score is better for teaching visualization.

    Preference:
    - First blocking hit farther than min_first_blocking_t.
    - Larger first_blocking_t, because very early hits are hard to see.
    - More blocking hits, because the ray-bundle effect is clearer.
    """
    t = c.json_first_blocking_t
    n_block = c.json_num_blocking_hits or 0

    if t is None:
        return (0, -1.0, n_block)

    passes_min = int(t >= min_first_blocking_t)
    return (passes_min, float(t), n_block)


def choose_candidates(
    all_candidates: list[FailureCandidate],
    case: str | None,
    target: str | None,
    candidate_index: int | None,
    max_cases: int,
    prefer_teaching_case: bool,
    min_first_blocking_t: float,
) -> list[FailureCandidate]:
    cands = all_candidates

    if case:
        cands = [c for c in cands if c.assembly_name == case]
    if target:
        cands = [c for c in cands if c.target_part == target]
    if candidate_index is not None:
        cands = [c for c in cands if c.candidate_index == candidate_index]

    if not cands:
        return []

    if prefer_teaching_case:
        cands = sorted(
            cands,
            key=lambda c: candidate_score_for_teaching(c, min_first_blocking_t),
            reverse=True,
        )

    return cands[:max_cases]


# -----------------------------------------------------------------------------
# Assembly/XML loading helpers
# -----------------------------------------------------------------------------


def resolve_existing_file(path_str: str, xml_path: Path) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)

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
            return str(c)

    # Return original; downstream error will show the path.
    return str(p)


def load_assembly_with_resolved_paths(xml_path: Path):
    """Load Assembly while resolving relative mesh/npy paths robustly."""
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
        if conn_type == "generic":
            conn = Connection.from_xml_element(elem)
        elif conn_type == "geometric":
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
# Geometry helpers
# -----------------------------------------------------------------------------


def orthonormal_basis_perpendicular_to(
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / (np.linalg.norm(d) + 1e-12)

    if abs(float(np.dot(d, [1.0, 0.0, 0.0]))) < 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)

    u = np.cross(d, helper)
    u = u / (np.linalg.norm(u) + 1e-12)

    v = np.cross(d, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    return u, v


def to_local(
    points: np.ndarray, center: np.ndarray, d: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    rel = np.asarray(points, dtype=float).reshape((-1, 3)) - center.reshape(1, 3)
    return np.column_stack([rel @ d, rel @ u, rel @ v])


def from_local(
    local: np.ndarray, center: np.ndarray, d: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    local = np.asarray(local, dtype=float).reshape((-1, 3))
    return (
        center.reshape(1, 3) + local[:, 0:1] * d + local[:, 1:2] * u + local[:, 2:3] * v
    )


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if pts.shape[0] <= 1:
        return pts.copy()

    pts = np.unique(np.round(pts, decimals=9), axis=0)
    if pts.shape[0] <= 2:
        return pts.copy()

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def point_in_convex_polygon_2d(
    point: np.ndarray, polygon: np.ndarray, eps: float = 1e-9
) -> bool:
    p = np.asarray(point, dtype=float).reshape(2)
    poly = np.asarray(polygon, dtype=float).reshape((-1, 2))

    if poly.shape[0] < 3:
        return False

    signs = []
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        edge = b - a
        rel = p - a
        cross = float(edge[0] * rel[1] - edge[1] * rel[0])
        if abs(cross) > eps:
            signs.append(np.sign(cross))

    if not signs:
        return True

    return all(s == signs[0] for s in signs)


def downsample_2d_points_with_center(
    pts2: np.ndarray,
    center2: np.ndarray,
    max_points: int,
    include_center: bool = True,
) -> np.ndarray:
    pts2 = np.asarray(pts2, dtype=float).reshape((-1, 2))
    center2 = np.asarray(center2, dtype=float).reshape(2)

    if max_points <= 0:
        return np.zeros((0, 2), dtype=float)

    if pts2.shape[0] == 0:
        return (
            center2.reshape(1, 2) if include_center else np.zeros((0, 2), dtype=float)
        )

    rel = pts2 - center2.reshape(1, 2)
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.linalg.norm(rel, axis=1)

    # Prefer outer points at similar angles.
    order = np.lexsort((-radii, angles))
    pts_sorted = pts2[order]

    rounded = np.round(pts_sorted, decimals=6)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    pts_sorted = pts_sorted[np.sort(idx)]

    budget = max_points - 1 if include_center else max_points
    budget = max(0, budget)

    if pts_sorted.shape[0] > budget and budget > 0:
        idx = np.linspace(0, pts_sorted.shape[0] - 1, budget).round().astype(int)
        pts_sorted = pts_sorted[idx]
    elif budget == 0:
        pts_sorted = np.zeros((0, 2), dtype=float)

    if include_center:
        pts_sorted = np.vstack([center2.reshape(1, 2), pts_sorted])

    return pts_sorted


def mesh_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    return np.asarray(mesh.vertices, dtype=float)


def mesh_triangles(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    return np.asarray(mesh.triangles, dtype=int)


def combined_vertices(meshes: list[o3d.geometry.TriangleMesh]) -> np.ndarray:
    arrays = [mesh_vertices(m) for m in meshes if m is not None and len(m.vertices) > 0]
    if not arrays:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(arrays)


def choose_center(target_mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    aabb = target_mesh.get_axis_aligned_bounding_box()
    return np.asarray(aabb.get_center(), dtype=float)


def compute_diag(points: np.ndarray, fallback: float = 1.0) -> float:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return fallback
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag) or diag <= 1e-12:
        return fallback
    return diag


def make_projected_footprint_ray_origins(
    vertices_world: np.ndarray,
    center_world: np.ndarray,
    direction: np.ndarray,
    ray_bundle_spacing: float | None = None,
    max_rays: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    d = direction / (np.linalg.norm(direction) + 1e-12)
    u, v = orthonormal_basis_perpendicular_to(d)

    local = to_local(vertices_world, center_world, d, u, v)
    pts2 = local[:, 1:3]

    if pts2.shape[0] == 0:
        return pts2, np.zeros((0, 2)), np.zeros((0, 2)), u, v, 0.0

    min_xy = np.min(pts2, axis=0)
    max_xy = np.max(pts2, axis=0)
    extent = max_xy - min_xy
    footprint_diag = float(np.linalg.norm(extent))

    if footprint_diag <= 1e-9 or not np.isfinite(footprint_diag):
        return (
            pts2,
            np.array([[0.0, 0.0]], dtype=float),
            np.array([[0.0, 0.0]], dtype=float),
            u,
            v,
            footprint_diag,
        )

    spacing = ray_bundle_spacing
    if spacing is None or spacing <= 1e-9 or not np.isfinite(spacing):
        spacing = max(0.02 * footprint_diag, min(8.0, max(0.02, 0.08 * footprint_diag)))

    hull = convex_hull_2d(pts2)

    if hull.shape[0] < 3:
        origins2 = downsample_2d_points_with_center(
            pts2, np.array([0.0, 0.0]), max_rays
        )
        return pts2, hull, origins2, u, v, footprint_diag

    xs = np.arange(min_xy[0], max_xy[0] + 0.5 * spacing, spacing, dtype=float)
    ys = np.arange(min_xy[1], max_xy[1] + 0.5 * spacing, spacing, dtype=float)

    grid: list[np.ndarray] = []
    for xx in xs:
        for yy in ys:
            p = np.array([xx, yy], dtype=float)
            if point_in_convex_polygon_2d(p, hull):
                grid.append(p)

    origins2 = np.asarray(grid, dtype=float) if grid else np.zeros((0, 2), dtype=float)

    # Always include centerline ray.
    origins2 = np.vstack([origins2, np.array([[0.0, 0.0]], dtype=float)])

    # Add a small subset of boundary/projected vertices.
    vertex_budget = min(32, max(8, max_rays // 5)) if max_rays > 0 else 32
    vertex_samples = downsample_2d_points_with_center(
        pts2,
        np.array([0.0, 0.0]),
        vertex_budget,
        include_center=False,
    )

    if vertex_samples.shape[0] > 0:
        origins2 = np.vstack([origins2, vertex_samples])

    if origins2.shape[0] > 1:
        rounded = np.round(origins2, decimals=6)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        origins2 = origins2[np.sort(idx)]

    if max_rays > 0 and origins2.shape[0] > max_rays:
        d_center = np.linalg.norm(origins2, axis=1)
        center_idx = int(np.argmin(d_center))
        other_idx = [i for i in range(origins2.shape[0]) if i != center_idx]
        keep_n_other = max(0, max_rays - 1)

        if keep_n_other > 0 and other_idx:
            sampled_pos = (
                np.linspace(0, len(other_idx) - 1, keep_n_other).round().astype(int)
            )
            sampled_other_idx = [other_idx[int(j)] for j in sampled_pos]
            keep_idx = [center_idx] + sampled_other_idx
        else:
            keep_idx = [center_idx]

        keep_idx = np.asarray(sorted(set(keep_idx)), dtype=int)
        origins2 = origins2[keep_idx]

    return pts2, hull, origins2, u, v, footprint_diag


def build_raycast_scene(assembly, moving_part_name: str):
    scene = o3d.t.geometry.RaycastingScene()
    geometry_names: dict[int, str] = {}

    for p in assembly.parts.values():
        name = getattr(p, "name", None)
        if name == moving_part_name:
            continue

        mesh = getattr(p, "mesh", None)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            continue

        try:
            gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            geometry_names[int(gid)] = str(name)
        except Exception as exc:
            print(f"[warn] Could not add obstacle mesh {name}: {exc}")

    return scene, geometry_names


def compute_ray_bundle(
    assembly,
    target_part_name: str,
    direction: np.ndarray,
    pullout_length: float | None,
    ray_bundle_spacing: float | None,
    max_rays: int,
    origin_offset: float,
    hit_margin: float,
) -> RayBundleResult:
    target = assembly.get_part_by_name(target_part_name)
    if target is None:
        raise ValueError(f"Target part not found: {target_part_name}")

    target_mesh = target.mesh
    verts = mesh_vertices(target_mesh)

    d = np.asarray(direction, dtype=float).reshape(3)
    d = d / (np.linalg.norm(d) + 1e-12)

    center = choose_center(target_mesh)

    pts2, hull2, origins2, u, v, footprint_diag = make_projected_footprint_ray_origins(
        verts,
        center,
        d,
        ray_bundle_spacing=ray_bundle_spacing,
        max_rays=max_rays,
    )

    all_verts = combined_vertices([p.mesh for p in assembly.parts.values()])
    assembly_diag = compute_diag(all_verts, fallback=100.0)
    target_diag = compute_diag(verts, fallback=max(assembly_diag * 0.05, 1.0))

    if pullout_length is None or pullout_length <= 0:
        # Keep this as the algorithmic/default evaluation length.
        pullout_length = max(40.0, min(200.0, 2.5 * assembly_diag))

    local3 = np.column_stack([np.zeros(origins2.shape[0]), origins2])
    origins_world = from_local(local3, center, d, u, v)
    origins_world = origins_world + d.reshape(1, 3) * float(origin_offset)

    effective_len = max(float(pullout_length) - float(origin_offset), 0.0)

    scene, geometry_names = build_raycast_scene(assembly, target_part_name)

    if origins_world.shape[0] == 0 or not geometry_names:
        t_hit = np.full((origins_world.shape[0],), np.inf, dtype=float)
        geometry_ids = None
    else:
        ray_dirs = np.tile(d.reshape(1, 3), (origins_world.shape[0], 1))
        rays = np.concatenate(
            [origins_world.astype(np.float32), ray_dirs.astype(np.float32)], axis=1
        )
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy().astype(float)
        geometry_ids = (
            ans["geometry_ids"].numpy().astype(int) if "geometry_ids" in ans else None
        )

    hit_mask = np.isfinite(t_hit)
    max_allowed_t = max(effective_len - float(hit_margin), 0.0)
    blocking_mask = hit_mask & (t_hit <= max_allowed_t)

    return RayBundleResult(
        center_world=center,
        direction=d,
        u=u,
        v=v,
        ray_length=float(pullout_length),
        effective_ray_length=float(effective_len),
        origin_offset=float(origin_offset),
        hit_margin=float(hit_margin),
        pts2=pts2,
        hull2=hull2,
        origins2=origins2,
        origins_world=origins_world,
        t_hit=t_hit,
        geometry_ids=geometry_ids,
        geometry_names=geometry_names,
        hit_mask=hit_mask,
        blocking_mask=blocking_mask,
        assembly_diag=float(assembly_diag),
        target_diag=float(target_diag),
        footprint_diag=float(footprint_diag),
    )


# -----------------------------------------------------------------------------
# Teaching view helpers
# -----------------------------------------------------------------------------


def make_teaching_view_config(
    rb: RayBundleResult,
    zoom_target: bool,
    short_rays: bool,
    show_only_blocking_neighborhood: bool,
    focus_footprint: bool,
    max_faces: int,
) -> TeachingViewConfig:
    first_t = rb.first_blocking_t_total

    if first_t is None:
        first_t = min(rb.ray_length, max(5.0, 0.25 * rb.assembly_diag))

    # Adaptive neighborhood size.
    # For tiny assemblies, absolute 5 mm may be too large; for large assemblies,
    # target-based scaling avoids excessive zoom-in.
    target_pad = max(0.8 * rb.target_diag, 0.15 * rb.assembly_diag, 1.0)
    hit_pad = max(5.0, 1.5 * rb.target_diag, 0.10 * rb.assembly_diag)

    neighborhood_x_max = max(first_t + hit_pad, rb.origin_offset + target_pad)

    if short_rays:
        display_ray_length = min(
            rb.ray_length,
            max(
                5.0,
                min(10.0, 0.40 * rb.assembly_diag),
                first_t + hit_pad,
            ),
        )
    else:
        display_ray_length = rb.ray_length

    if show_only_blocking_neighborhood:
        display_ray_length = min(display_ray_length, neighborhood_x_max)

    # Larger markers for small footprints.
    if rb.footprint_diag <= 1e-9:
        origin_marker = 40.0
    else:
        origin_marker = float(np.clip(260.0 / max(rb.origins2.shape[0], 1), 24.0, 70.0))

    center_marker = origin_marker * 2.2
    hit_marker = float(np.clip(origin_marker * 1.8, 36.0, 120.0))

    return TeachingViewConfig(
        zoom_target=zoom_target,
        short_rays=short_rays,
        show_only_blocking_neighborhood=show_only_blocking_neighborhood,
        focus_footprint=focus_footprint,
        display_ray_length=float(display_ray_length),
        neighborhood_x_max=float(neighborhood_x_max),
        marker_size_ray_origin_2d=float(origin_marker),
        marker_size_center_2d=float(center_marker),
        marker_size_hit_3d=float(hit_marker),
        title_fontsize=9.0,
        legend_fontsize=7.5,
        mesh_max_faces=max_faces,
    )


def should_draw_part(
    name: str, target_name: str, rb: RayBundleResult, cfg: TeachingViewConfig
) -> bool:
    if name == target_name:
        return True

    if not cfg.show_only_blocking_neighborhood:
        return True

    blocking_names = rb.blocking_geometry_names
    if not blocking_names:
        return True

    return name in blocking_names


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def setup_3d(
    ax, title: str | None = None, cfg: TeachingViewConfig | None = None
) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.6, 1.0, 0.8))
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()

    if title:
        fontsize = cfg.title_fontsize if cfg is not None else 9.0
        ax.set_title(title, pad=8, fontweight="bold", fontsize=fontsize)


def add_mesh_local(
    ax,
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    color: str,
    alpha: float,
    edgecolor: str | None = None,
    lw: float = 0.15,
    max_faces: int = 2500,
    translate_local: np.ndarray | None = None,
) -> np.ndarray:
    verts_world = mesh_vertices(mesh)
    tris = mesh_triangles(mesh)

    local = to_local(verts_world, center, d, u, v)
    if translate_local is not None:
        local = local + np.asarray(translate_local, dtype=float).reshape(1, 3)

    if tris.shape[0] > max_faces:
        idx = np.linspace(0, tris.shape[0] - 1, max_faces).round().astype(int)
        tris_draw = tris[idx]
    else:
        tris_draw = tris

    faces = [local[t] for t in tris_draw]

    pc = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=edgecolor or color,
        linewidths=lw,
        alpha=alpha,
    )
    ax.add_collection3d(pc)

    return local


def set_equal_bounds_local(
    ax,
    local_points: np.ndarray,
    extra: list[np.ndarray] | None = None,
    margin: float = 0.18,
    zoom_x_range: tuple[float, float] | None = None,
) -> None:
    pts = [np.asarray(local_points, dtype=float).reshape((-1, 3))]

    if extra:
        for e in extra:
            if e is None:
                continue
            arr = np.asarray(e, dtype=float)
            if arr.size:
                pts.append(arr.reshape((-1, 3)))

    allp = np.vstack([p for p in pts if p.size])

    if zoom_x_range is not None:
        x0, x1 = zoom_x_range
        mask = (allp[:, 0] >= x0) & (allp[:, 0] <= x1)
        if np.any(mask):
            allp = allp[mask]
        allp = np.vstack([allp, np.array([[x0, 0.0, 0.0], [x1, 0.0, 0.0]])])

    lo = allp.min(axis=0)
    hi = allp.max(axis=0)

    center = 0.5 * (lo + hi)
    span = hi - lo
    radius_yz = 0.5 * float(max(span[1], span[2]))
    radius_x = 0.5 * float(span[0])

    radius = max(radius_x / 1.6, radius_yz, 1.0) * (1.0 + margin)

    ax.set_xlim(center[0] - 1.6 * radius, center[0] + 1.6 * radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def set_footprint_focused_bounds(
    ax, rb: RayBundleResult, cfg: TeachingViewConfig, include_x: float | None = None
) -> None:
    """
    Focus the 3D camera limits around the projection plane, footprint, and
    short pull-out neighborhood instead of the whole assembly.
    """
    if rb.hull2.shape[0] >= 3:
        yz = rb.hull2
    elif rb.origins2.shape[0] > 0:
        yz = rb.origins2
    else:
        yz = np.zeros((1, 2), dtype=float)

    yz_lo = yz.min(axis=0)
    yz_hi = yz.max(axis=0)
    yz_center = 0.5 * (yz_lo + yz_hi)

    yz_radius = 0.5 * max(float(np.max(yz_hi - yz_lo)), 1e-6)
    yz_radius = max(yz_radius * 1.65, 0.45 * rb.target_diag, 0.5)

    x0 = -0.18 * rb.target_diag
    x1 = cfg.display_ray_length if include_x is None else include_x
    x1 = max(x1, cfg.neighborhood_x_max)

    ax.set_xlim(x0, x1)
    ax.set_ylim(yz_center[0] - yz_radius, yz_center[0] + yz_radius)
    ax.set_zlim(yz_center[1] - yz_radius, yz_center[1] + yz_radius)


def arrow3d_local(ax, start, end, color=KIT_RED, lw=2.0) -> None:
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)

    ax.plot(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, lw=lw
    )

    vec = end - start
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return

    d = vec / n
    head = min(0.12 * n, 8.0)
    p = end

    ax.plot(
        [p[0], p[0] - head * d[0]],
        [p[1], p[1] + 0.35 * head],
        [p[2], p[2]],
        color=color,
        lw=lw,
    )
    ax.plot(
        [p[0], p[0] - head * d[0]],
        [p[1], p[1] - 0.35 * head],
        [p[2], p[2]],
        color=color,
        lw=lw,
    )


def add_projection_plane(ax, rb: RayBundleResult, size_scale: float = 1.25) -> None:
    if rb.hull2.shape[0] >= 3:
        lo = rb.hull2.min(axis=0)
        hi = rb.hull2.max(axis=0)
    elif rb.pts2.shape[0] > 0:
        lo = rb.pts2.min(axis=0)
        hi = rb.pts2.max(axis=0)
    else:
        lo = np.array([-1.0, -1.0])
        hi = np.array([1.0, 1.0])

    c = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * size_scale
    half = np.maximum(half, max(rb.footprint_diag * 0.15, 0.5))

    y0, z0 = c - half
    y1, z1 = c + half

    plane = np.array(
        [
            [0, y0, z0],
            [0, y1, z0],
            [0, y1, z1],
            [0, y0, z1],
        ],
        dtype=float,
    )

    pc = Poly3DCollection(
        [plane],
        facecolors=KIT_BLUE_GREEN,
        edgecolors=KIT_BLUE_GREEN,
        linewidths=0.7,
        alpha=0.13,
    )
    ax.add_collection3d(pc)


def add_footprint_on_plane(
    ax, rb: RayBundleResult, x: float = 0.0, color: str = KIT_RED
) -> None:
    if rb.hull2.shape[0] < 2:
        return

    hull = np.vstack([rb.hull2, rb.hull2[0]])
    ax.plot([x] * hull.shape[0], hull[:, 0], hull[:, 1], color=color, lw=1.6)


def add_ray_origins_on_plane_3d(
    ax, rb: RayBundleResult, cfg: TeachingViewConfig, x: float = 0.0
) -> None:
    """Draw ray origins directly on the projection plane in local coordinates."""
    if rb.origins2.shape[0] == 0:
        return

    clear_mask = ~rb.blocking_mask
    block_mask = rb.blocking_mask

    if np.any(clear_mask):
        pts = rb.origins2[clear_mask]
        ax.scatter(
            np.full(pts.shape[0], x),
            pts[:, 0],
            pts[:, 1],
            s=max(10.0, cfg.marker_size_ray_origin_2d * 0.55),
            color=KIT_DEEP_BLUE,
            depthshade=False,
            alpha=0.95,
        )

    if np.any(block_mask):
        pts = rb.origins2[block_mask]
        ax.scatter(
            np.full(pts.shape[0], x),
            pts[:, 0],
            pts[:, 1],
            s=max(14.0, cfg.marker_size_ray_origin_2d * 0.70),
            color=KIT_RED,
            depthshade=False,
            alpha=0.95,
        )

    ax.scatter(
        [x],
        [0.0],
        [0.0],
        s=max(28.0, cfg.marker_size_center_2d * 0.45),
        color=KIT_BLUE_GREEN,
        marker="x",
        linewidths=1.8,
        depthshade=False,
    )


def save_formats(fig, out_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []

    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        out.append(path)

    plt.close(fig)
    return out


def collect_local_points_for_drawn_parts(
    assembly, target_name: str, rb: RayBundleResult, cfg: TeachingViewConfig
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for p in assembly.parts.values():
        if should_draw_part(p.name, target_name, rb, cfg):
            arrays.append(
                to_local(
                    mesh_vertices(p.mesh), rb.center_world, rb.direction, rb.u, rb.v
                )
            )
    if not arrays:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(arrays)


def figure_01(
    assembly,
    target_name: str,
    rb: RayBundleResult,
    cfg: TeachingViewConfig,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, "Local motion exists, full removal path is blocked", cfg)

    all_local = []

    for p in assembly.parts.values():
        if not should_draw_part(p.name, target_name, rb, cfg):
            continue

        if p.name == target_name:
            # Current target pose.
            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_YELLOW,
                0.42,
                edgecolor="#8A6D00",
                lw=0.14,
                max_faces=cfg.mesh_max_faces,
            )
        else:
            # Blocking / surrounding part.
            obstacle_alpha = 0.10 if cfg.focus_footprint else 0.18
            obstacle_lw = 0.045 if cfg.focus_footprint else 0.10

            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_DEEP_BLUE,
                obstacle_alpha,
                edgecolor=KIT_DEEP_BLUE,
                lw=obstacle_lw,
                max_faces=cfg.mesh_max_faces,
            )

        all_local.append(local)

    target = assembly.get_part_by_name(target_name)

    # Stage 1: target after short local translation.
    # This visualizes that the first-stage local motion is feasible.
    local_step = min(5.0, max(0.5, 0.25 * rb.target_diag))

    add_mesh_local(
        ax,
        target.mesh,
        rb.center_world,
        rb.direction,
        rb.u,
        rb.v,
        KIT_BLUE_GREEN,
        0.36,
        edgecolor=KIT_BLUE_GREEN,
        lw=0.12,
        max_faces=cfg.mesh_max_faces,
        translate_local=np.array([local_step, 0, 0]),
    )

    # Stage 2: target farther along the pull-out corridor.
    # This visualizes the candidate full removal path that is rejected.
    add_mesh_local(
        ax,
        target.mesh,
        rb.center_world,
        rb.direction,
        rb.u,
        rb.v,
        KIT_RED,
        0.16,
        edgecolor=KIT_RED,
        lw=0.10,
        max_faces=cfg.mesh_max_faces,
        translate_local=np.array([cfg.display_ray_length, 0, 0]),
    )

    # Local translation indicator.
    arrow3d_local(
        ax,
        (0, -0.35 * rb.target_diag, -0.10 * rb.target_diag),
        (local_step, -0.35 * rb.target_diag, -0.10 * rb.target_diag),
        KIT_BLUE_GREEN,
        2.0,
    )

    # Full pull-out path indicator.
    arrow3d_local(
        ax,
        (0, 0.20 * rb.target_diag, 0.10 * rb.target_diag),
        (cfg.display_ray_length, 0.20 * rb.target_diag, 0.10 * rb.target_diag),
        KIT_RED,
        2.2,
    )

    # Optional footprint-focused context.
    if cfg.focus_footprint:
        add_projection_plane(ax, rb, size_scale=1.15)
        add_footprint_on_plane(ax, rb, x=rb.origin_offset, color=KIT_RED)
        add_ray_origins_on_plane_3d(ax, rb, cfg, x=rb.origin_offset)

    points_for_bounds = (
        np.vstack(all_local)
        if all_local
        else collect_local_points_for_drawn_parts(assembly, target_name, rb, cfg)
    )

    # Extra points used only for robust camera bounds.
    extra = [
        np.array([[local_step, 0.0, 0.0]]),
        np.array([[cfg.display_ray_length, 0.0, 0.0]]),
        np.array(
            [
                [0.0, -0.35 * rb.target_diag, -0.10 * rb.target_diag],
                [local_step, -0.35 * rb.target_diag, -0.10 * rb.target_diag],
                [0.0, 0.20 * rb.target_diag, 0.10 * rb.target_diag],
                [cfg.display_ray_length, 0.20 * rb.target_diag, 0.10 * rb.target_diag],
            ]
        ),
    ]

    zoom_range = (0.0, cfg.neighborhood_x_max) if cfg.zoom_target else None

    if cfg.focus_footprint:
        set_footprint_focused_bounds(ax, rb, cfg, include_x=cfg.display_ray_length)
    else:
        set_equal_bounds_local(
            ax,
            points_for_bounds,
            extra=extra,
            zoom_x_range=zoom_range,
        )

    ax.legend(
        handles=[
            # Target pose semantics.
            plt.Line2D(
                [0],
                [0],
                color=KIT_YELLOW,
                lw=5,
                alpha=0.55,
                label="target at current state",
            ),
            plt.Line2D(
                [0],
                [0],
                color=KIT_DEEP_BLUE,
                lw=5,
                alpha=0.30,
                label="blocking part",
            ),
            # Motion indicators.
            plt.Line2D(
                [0],
                [0],
                color=KIT_BLUE_GREEN,
                lw=2.0,
                label="local check",
            ),
            plt.Line2D(
                [0],
                [0],
                color=KIT_BLUE_GREEN,
                lw=5,
                alpha=0.55,
                label="after local check: feasible",
            ),
            plt.Line2D(
                [0],
                [0],
                color=KIT_RED,
                lw=2.2,
                label="pull-out path",
            ),
            plt.Line2D(
                [0],
                [0],
                color=KIT_RED,
                lw=5,
                alpha=0.35,
                label="full pull-out pose: blocked",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=cfg.legend_fontsize,
    )

    return save_formats(fig, out_dir, "01_local_motion_vs_blocked_path", formats)


def figure_02(
    assembly,
    target_name: str,
    rb: RayBundleResult,
    cfg: TeachingViewConfig,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, "Projection plane and target footprint", cfg)

    all_local = []

    for p in assembly.parts.values():
        if not should_draw_part(p.name, target_name, rb, cfg):
            continue

        if p.name == target_name:
            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_YELLOW,
                0.60,
                edgecolor="#8A6D00",
                lw=0.16,
                max_faces=cfg.mesh_max_faces,
            )
        else:
            obstacle_alpha = 0.055 if cfg.focus_footprint else 0.12
            obstacle_lw = 0.05 if cfg.focus_footprint else 0.10

            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_DEEP_BLUE,
                obstacle_alpha,
                edgecolor=KIT_DEEP_BLUE,
                lw=obstacle_lw,
                max_faces=cfg.mesh_max_faces,
            )
        all_local.append(local)

    add_projection_plane(ax, rb)
    add_footprint_on_plane(ax, rb, x=0.0, color=KIT_RED)
    arrow3d_local(ax, (0, 0, 0), (cfg.display_ray_length, 0, 0), KIT_RED, 1.9)
    if cfg.focus_footprint:
        add_ray_origins_on_plane_3d(ax, rb, cfg, x=0.0)

    hull3 = (
        np.column_stack([np.zeros(rb.hull2.shape[0]), rb.hull2])
        if rb.hull2.size
        else None
    )
    points_for_bounds = (
        np.vstack(all_local)
        if all_local
        else collect_local_points_for_drawn_parts(assembly, target_name, rb, cfg)
    )
    zoom_range = (
        (-0.15 * rb.target_diag, cfg.neighborhood_x_max) if cfg.zoom_target else None
    )
    if cfg.focus_footprint:
        set_footprint_focused_bounds(ax, rb, cfg, include_x=cfg.display_ray_length)
    else:
        set_equal_bounds_local(
            ax,
            points_for_bounds,
            extra=[hull3, np.array([[cfg.display_ray_length, 0, 0]])],
            zoom_x_range=zoom_range,
        )

    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=KIT_BLUE_GREEN,
                lw=5,
                alpha=0.25,
                label="projection plane",
            ),
            plt.Line2D(
                [0], [0], color=KIT_RED, lw=1.8, label="footprint / convex hull"
            ),
            plt.Line2D([0], [0], color=KIT_RED, lw=1.8, label="pull-out direction"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=cfg.legend_fontsize,
    )

    return save_formats(fig, out_dir, "02_projection_plane_and_footprint", formats)


def figure_03(
    rb: RayBundleResult,
    cfg: TeachingViewConfig,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.set_title(
        "2D footprint sampling on the projection plane",
        fontweight="bold",
        pad=6,
        fontsize=cfg.title_fontsize,
    )
    ax.set_facecolor("white")

    if rb.hull2.shape[0] >= 3:
        hull_closed = np.vstack([rb.hull2, rb.hull2[0]])
        ax.add_patch(
            Polygon2D(
                rb.hull2,
                closed=True,
                facecolor=KIT_YELLOW,
                edgecolor=KIT_RED,
                linewidth=1.6,
                alpha=0.22,
                label="projected footprint",
            )
        )
        ax.plot(
            hull_closed[:, 0],
            hull_closed[:, 1],
            color=KIT_RED,
            lw=1.6,
            label="convex hull",
        )

    clear_mask = ~rb.blocking_mask
    block_mask = rb.blocking_mask

    if np.any(clear_mask):
        ax.scatter(
            rb.origins2[clear_mask, 0],
            rb.origins2[clear_mask, 1],
            s=cfg.marker_size_ray_origin_2d,
            color=KIT_DEEP_BLUE,
            label="clear ray origins",
            zorder=3,
        )

    if np.any(block_mask):
        ax.scatter(
            rb.origins2[block_mask, 0],
            rb.origins2[block_mask, 1],
            s=cfg.marker_size_ray_origin_2d * 1.25,
            color=KIT_RED,
            label="blocking ray origins",
            zorder=4,
        )

    ax.scatter(
        [0],
        [0],
        s=cfg.marker_size_center_2d,
        color=KIT_BLUE_GREEN,
        marker="x",
        linewidths=2.1,
        label="centerline ray",
        zorder=5,
    )

    if rb.hull2.shape[0] > 0:
        vertex_radius = max(0.006 * max(rb.footprint_diag, 1.0), 0.015)
        for p in rb.hull2:
            ax.add_patch(
                Circle((p[0], p[1]), vertex_radius, color=KIT_RED, alpha=0.45, zorder=2)
            )

    ax.set_xlabel("footprint axis u")
    ax.set_ylabel("footprint axis v")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E8EB", linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=7)

    # Adaptive limits with extra margin.
    pts = rb.hull2 if rb.hull2.shape[0] >= 3 else rb.origins2
    if pts.shape[0] > 0:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = 0.5 * (lo + hi)
        radius = 0.5 * max(float(np.max(hi - lo)), 1e-6)
        radius = max(radius * 1.22, 0.5 * max(rb.footprint_diag, 1.0) * 0.12)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=cfg.legend_fontsize,
    )

    return save_formats(fig, out_dir, "03_2d_footprint_sampling", formats)


def draw_figure_03_panel(
    ax,
    rb: RayBundleResult,
    cfg: TeachingViewConfig,
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Draw one 2D projected-footprint sampling panel without a local legend."""
    ax.set_title(title, fontweight="bold", pad=6, fontsize=cfg.title_fontsize)
    ax.set_facecolor("white")

    if rb.hull2.shape[0] >= 3:
        hull_closed = np.vstack([rb.hull2, rb.hull2[0]])
        ax.add_patch(
            Polygon2D(
                rb.hull2,
                closed=True,
                facecolor=KIT_YELLOW,
                edgecolor=KIT_RED,
                linewidth=1.6,
                alpha=0.22,
            )
        )
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], color=KIT_RED, lw=1.6)

    clear_mask = ~rb.blocking_mask
    block_mask = rb.blocking_mask

    if np.any(clear_mask):
        ax.scatter(
            rb.origins2[clear_mask, 0],
            rb.origins2[clear_mask, 1],
            s=cfg.marker_size_ray_origin_2d,
            color=KIT_DEEP_BLUE,
            zorder=3,
        )

    if np.any(block_mask):
        ax.scatter(
            rb.origins2[block_mask, 0],
            rb.origins2[block_mask, 1],
            s=cfg.marker_size_ray_origin_2d * 1.25,
            color=KIT_RED,
            zorder=4,
        )

    ax.scatter(
        [0],
        [0],
        s=cfg.marker_size_center_2d,
        color=KIT_BLUE_GREEN,
        marker="x",
        linewidths=2.1,
        zorder=5,
    )

    if rb.hull2.shape[0] > 0:
        vertex_radius = max(0.006 * max(rb.footprint_diag, 1.0), 0.015)
        for p in rb.hull2:
            ax.add_patch(
                Circle((p[0], p[1]), vertex_radius, color=KIT_RED, alpha=0.45, zorder=2)
            )

    ax.set_xlabel("footprint axis u")
    if show_ylabel:
        ax.set_ylabel("footprint axis v")
    else:
        ax.set_ylabel("")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E8EB", linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=7)

    pts = rb.hull2 if rb.hull2.shape[0] >= 3 else rb.origins2
    if pts.shape[0] > 0:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = 0.5 * (lo + hi)
        radius = 0.5 * max(float(np.max(hi - lo)), 1e-6)
        radius = max(radius * 1.22, 0.5 * max(rb.footprint_diag, 1.0) * 0.12)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)


def figure_03_two_parts(
    assembly,
    target_names: list[str],
    direction: np.ndarray,
    out_dir: Path,
    formats: tuple[str, ...],
    pullout_length: float | None,
    ray_bundle_spacing: float | None,
    max_rays: int,
    origin_offset: float,
    hit_margin: float,
    max_faces: int,
    zoom_target: bool,
    short_rays: bool,
    show_only_blocking_neighborhood: bool,
    focus_footprint: bool,
) -> list[Path]:
    """
    Render figure_03 for two target parts as one shared-legend multi-panel figure.

    This is only a visualization helper. It reuses the same candidate direction
    for both parts so that the two footprints are shown in a comparable projection frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rbs: list[RayBundleResult] = []
    cfgs: list[TeachingViewConfig] = []

    for target_name in target_names:
        rb = compute_ray_bundle(
            assembly=assembly,
            target_part_name=target_name,
            direction=direction,
            pullout_length=pullout_length,
            ray_bundle_spacing=ray_bundle_spacing,
            max_rays=max_rays,
            origin_offset=origin_offset,
            hit_margin=hit_margin,
        )

        cfg = make_teaching_view_config(
            rb=rb,
            zoom_target=zoom_target,
            short_rays=short_rays,
            show_only_blocking_neighborhood=show_only_blocking_neighborhood,
            focus_footprint=focus_footprint,
            max_faces=max_faces,
        )

        rbs.append(rb)
        cfgs.append(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))

    draw_figure_03_panel(
        axes[0],
        rbs[0],
        cfgs[0],
        title=f"{target_names[0]}",
        show_ylabel=True,
    )
    draw_figure_03_panel(
        axes[1],
        rbs[1],
        cfgs[1],
        title=f"{target_names[1]}",
        show_ylabel=False,
    )

    handles = [
        Patch(
            facecolor=KIT_YELLOW,
            edgecolor=KIT_RED,
            alpha=0.22,
            label="projected footprint",
        ),
        plt.Line2D(
            [0],
            [0],
            color=KIT_RED,
            lw=1.6,
            label="convex hull",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=KIT_DEEP_BLUE,
            markersize=6,
            label="clear ray origins",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=KIT_RED,
            markersize=6,
            label="blocking ray origins",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color=KIT_BLUE_GREEN,
            markersize=8,
            markeredgewidth=2.0,
            linestyle="None",
            label="centerline ray",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=cfgs[0].legend_fontsize,
    )

    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

    generated = save_formats(
        fig,
        out_dir,
        "03_2d_footprint_sampling_two_parts",
        formats,
    )

    summary = {
        "target_names": target_names,
        "direction": direction.tolist(),
        "rendered": [
            {
                "target_part": target_names[i],
                "rendered_num_rays": int(rbs[i].origins2.shape[0]),
                "rendered_num_hits": int(np.sum(rbs[i].hit_mask)),
                "rendered_num_blocking_hits": int(np.sum(rbs[i].blocking_mask)),
                "footprint_diag": rbs[i].footprint_diag,
                "blocking_geometry_names": sorted(rbs[i].blocking_geometry_names),
            }
            for i in range(len(target_names))
        ],
    }

    summary_path = out_dir / "03_2d_footprint_sampling_two_parts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated.append(summary_path)

    return generated


def figure_04(
    assembly,
    target_name: str,
    rb: RayBundleResult,
    cfg: TeachingViewConfig,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, "Ray bundle detects blocked pull-out corridor", cfg)

    all_local = []

    for p in assembly.parts.values():
        if not should_draw_part(p.name, target_name, rb, cfg):
            continue

        if p.name == target_name:
            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_YELLOW,
                0.32,
                edgecolor="#8A6D00",
                lw=0.10,
                max_faces=cfg.mesh_max_faces,
            )
        else:
            obstacle_alpha = 0.10 if cfg.focus_footprint else 0.18
            obstacle_lw = 0.045 if cfg.focus_footprint else 0.10

            local = add_mesh_local(
                ax,
                p.mesh,
                rb.center_world,
                rb.direction,
                rb.u,
                rb.v,
                KIT_DEEP_BLUE,
                obstacle_alpha,
                edgecolor=KIT_DEEP_BLUE,
                lw=obstacle_lw,
                max_faces=cfg.mesh_max_faces,
            )
        all_local.append(local)

    segments = []
    colors = []
    widths = []
    hit_points_local = []

    for i, yz in enumerate(rb.origins2):
        y, z = float(yz[0]), float(yz[1])

        if rb.blocking_mask[i]:
            t_total = float(rb.t_hit[i]) + rb.origin_offset
            t_draw = min(t_total, cfg.display_ray_length)
            segments.append([(rb.origin_offset, y, z), (t_draw, y, z)])
            colors.append(KIT_RED)
            widths.append(1.7)
            hit_points_local.append([t_draw, y, z])
        else:
            segments.append([(rb.origin_offset, y, z), (cfg.display_ray_length, y, z)])
            colors.append(KIT_BLUE_GREEN)
            widths.append(0.9)

    lc = Line3DCollection(segments, colors=colors, linewidths=widths, alpha=0.92)
    ax.add_collection3d(lc)
    if cfg.focus_footprint:
        add_projection_plane(ax, rb, size_scale=1.15)
        add_footprint_on_plane(ax, rb, x=rb.origin_offset, color=KIT_RED)
        add_ray_origins_on_plane_3d(ax, rb, cfg, x=rb.origin_offset)

    if hit_points_local:
        hp = np.asarray(hit_points_local)
        ax.scatter(
            hp[:, 0],
            hp[:, 1],
            hp[:, 2],
            s=cfg.marker_size_hit_3d,
            color=KIT_RED,
            depthshade=False,
        )

    points_for_bounds = (
        np.vstack(all_local)
        if all_local
        else collect_local_points_for_drawn_parts(assembly, target_name, rb, cfg)
    )

    extra = [
        np.array([[cfg.display_ray_length, 0, 0]]),
        np.asarray(hit_points_local) if hit_points_local else None,
        np.column_stack([np.full(rb.origins2.shape[0], rb.origin_offset), rb.origins2]),
    ]

    zoom_range = (0.0, cfg.neighborhood_x_max) if cfg.zoom_target else None

    if cfg.focus_footprint:
        set_footprint_focused_bounds(ax, rb, cfg, include_x=cfg.display_ray_length)
    else:
        set_equal_bounds_local(
            ax,
            points_for_bounds,
            extra=extra,
            zoom_x_range=zoom_range,
        )

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=KIT_BLUE_GREEN, lw=2.0, label="clear rays"),
            plt.Line2D([0], [0], color=KIT_RED, lw=2.0, label="blocking rays"),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=KIT_RED,
                markersize=5,
                label="blocking hit",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=cfg.legend_fontsize,
    )

    return save_formats(fig, out_dir, "04_ray_bundle_blocking_hits", formats)


def figure_00_assembly_overview(
    assembly,
    target_name: str,
    out_dir: Path,
    formats: tuple[str, ...],
    max_faces: int = 2500,
) -> list[Path]:
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, "Assembly overview", None)

    all_pts = []

    for p in assembly.parts.values():
        if p.name == target_name:
            color = KIT_YELLOW
            alpha = 0.72
            edge = "#8A6D00"
            lw = 0.12
        else:
            color = KIT_DEEP_BLUE
            alpha = 0.28
            edge = KIT_DEEP_BLUE
            lw = 0.08

        verts = mesh_vertices(p.mesh)
        tris = mesh_triangles(p.mesh)

        if tris.shape[0] > max_faces:
            idx = np.linspace(0, tris.shape[0] - 1, max_faces).round().astype(int)
            tris_draw = tris[idx]
        else:
            tris_draw = tris

        faces = [verts[t] for t in tris_draw]

        pc = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors=edge,
            linewidths=lw,
            alpha=alpha,
        )
        ax.add_collection3d(pc)
        all_pts.append(verts)

    if all_pts:
        pts = np.vstack(all_pts)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = 0.5 * (lo + hi)
        span = hi - lo
        radius = 0.5 * max(float(np.max(span)), 1.0) * 1.18

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

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
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )

    return save_formats(fig, out_dir, "00_assembly_overview", formats)


def write_summary(
    out_dir: Path, cand: FailureCandidate, rb: RayBundleResult, cfg: TeachingViewConfig
) -> Path:
    path = out_dir / "summary.json"

    first_idx = (
        int(np.flatnonzero(rb.blocking_mask)[0]) if np.any(rb.blocking_mask) else None
    )
    first_gid = None
    first_geom = None

    if first_idx is not None and rb.geometry_ids is not None:
        try:
            first_gid = int(rb.geometry_ids[first_idx])
            first_geom = rb.geometry_names.get(first_gid)
        except Exception:
            pass

    data = {
        "assembly_name": cand.assembly_name,
        "xml_path": str(cand.xml_path),
        "attempt_index": cand.attempt_index,
        "target_part": cand.target_part,
        "candidate_index": cand.candidate_index,
        "direction": cand.direction.tolist(),
        "json_first_blocking_geometry": cand.first_blocking_geometry,
        "json_first_blocking_t": cand.json_first_blocking_t,
        "json_num_blocking_hits": cand.json_num_blocking_hits,
        "json_summary": cand.json_summary,
        "rendered_num_rays": int(rb.origins2.shape[0]),
        "rendered_num_hits": int(np.sum(rb.hit_mask)),
        "rendered_num_blocking_hits": int(np.sum(rb.blocking_mask)),
        "rendered_first_blocking_ray_index": first_idx,
        "rendered_first_blocking_geometry_id": first_gid,
        "rendered_first_blocking_geometry": first_geom,
        "rendered_first_blocking_t_total": rb.first_blocking_t_total,
        "rendered_blocking_geometry_names": sorted(rb.blocking_geometry_names),
        "ray_length_algorithmic": rb.ray_length,
        "ray_length_display": cfg.display_ray_length,
        "effective_ray_length": rb.effective_ray_length,
        "origin_offset": rb.origin_offset,
        "hit_margin": rb.hit_margin,
        "assembly_diag": rb.assembly_diag,
        "target_diag": rb.target_diag,
        "footprint_diag": rb.footprint_diag,
        "teaching_mode": {
            "zoom_target": cfg.zoom_target,
            "short_rays": cfg.short_rays,
            "show_only_blocking_neighborhood": cfg.show_only_blocking_neighborhood,
            "neighborhood_x_max": cfg.neighborhood_x_max,
            "marker_size_ray_origin_2d": cfg.marker_size_ray_origin_2d,
            "marker_size_hit_3d": cfg.marker_size_hit_3d,
            "focus_footprint": cfg.focus_footprint,
        },
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def render_candidate(
    cand: FailureCandidate,
    out_root: Path,
    formats: tuple[str, ...],
    pullout_length: float | None,
    ray_bundle_spacing: float | None,
    max_rays: int,
    origin_offset: float,
    hit_margin: float,
    max_faces: int,
    zoom_target: bool,
    short_rays: bool,
    show_only_blocking_neighborhood: bool,
    focus_footprint: bool,
) -> list[Path]:
    assembly = load_assembly_with_resolved_paths(cand.xml_path)

    if not cand.target_part or assembly.get_part_by_name(cand.target_part) is None:
        # Fallback: prefer part_0, otherwise first part.
        if assembly.get_part_by_name("part_0") is not None:
            cand.target_part = "part_0"
        else:
            cand.target_part = next(iter(assembly.parts.keys()))

    rb = compute_ray_bundle(
        assembly=assembly,
        target_part_name=cand.target_part,
        direction=cand.direction,
        pullout_length=pullout_length,
        ray_bundle_spacing=ray_bundle_spacing,
        max_rays=max_rays,
        origin_offset=origin_offset,
        hit_margin=hit_margin,
    )

    cfg = make_teaching_view_config(
        rb=rb,
        zoom_target=zoom_target,
        short_rays=short_rays,
        show_only_blocking_neighborhood=show_only_blocking_neighborhood,
        max_faces=max_faces,
        focus_footprint=focus_footprint,
    )

    out_dir = (
        out_root
        / cand.assembly_name
        / f"target_{cand.target_part}_candidate_{cand.candidate_index:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.extend(
        figure_00_assembly_overview(
            assembly, cand.target_part, out_dir, formats, max_faces=max_faces
        )
    )
    generated.extend(figure_01(assembly, cand.target_part, rb, cfg, out_dir, formats))
    generated.extend(figure_02(assembly, cand.target_part, rb, cfg, out_dir, formats))
    generated.extend(figure_03(rb, cfg, out_dir, formats))
    generated.extend(figure_04(assembly, cand.target_part, rb, cfg, out_dir, formats))
    generated.append(write_summary(out_dir, cand, rb, cfg))

    return generated


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render real secondary feasibility check figures for rotational joint cases."
    )

    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[2]
        .joinpath(
            "tests/external_cases/results/joint_assembly_rotation/test_joint_assembly_rotation_1-24_20260512_083214.jsonl"
        ),
        help="Path to the rotational joint JSONL result log.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("external/test_cases_external"),
        help="Root folder containing external test case folders.",
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
        .parents[2]
        .joinpath("figures/secondary_feasibility_real"),
        help="Output root directory.",
    )

    parser.add_argument(
        "--case",
        nargs="+",
        default=["auto"],
        help="Assembly name(s) to render. Use 'auto' to automatically choose a teaching-friendly case.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target part name. Default: target from JSONL attempt.",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help="Candidate failure index to render. Use -1 to accept the best matching candidate after filters.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=1,
        help="Maximum number of candidates/cases to render.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )

    parser.add_argument(
        "--pullout-length",
        type=float,
        default=None,
        help="Override algorithmic pull-out ray length in model units. Default: adaptive from assembly size.",
    )
    parser.add_argument(
        "--ray-bundle-spacing",
        type=float,
        default=None,
        help="Override projected-footprint grid spacing. Default: adaptive.",
    )
    parser.add_argument(
        "--max-rays",
        type=int,
        default=200,
        help="Maximum ray count before deterministic downsampling.",
    )
    parser.add_argument(
        "--origin-offset",
        type=float,
        default=1.0,
        help="Small offset along pull-out direction before ray casting.",
    )
    parser.add_argument(
        "--hit-margin",
        type=float,
        default=1.0,
        help="End-of-segment margin ignored for blocking-hit decisions.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=2500,
        help="Maximum mesh faces rendered per part for speed/readability.",
    )

    parser.add_argument(
        "--zoom-target",
        action="store_true",
        help="Teaching view: zoom into the target and first blocking neighborhood.",
    )
    parser.add_argument(
        "--short-rays",
        action="store_true",
        help="Teaching view: draw clear rays only over a short, adaptive display length.",
    )
    parser.add_argument(
        "--show-only-blocking-neighborhood",
        action="store_true",
        help="Teaching view: draw only the target and blocking/surrounding parts involved in blocking hits.",
    )
    parser.add_argument(
        "--min-first-blocking-t",
        type=float,
        default=2.0,
        help="When automatically selecting a case, prefer candidates whose first blocking hit is at least this far from the ray origin.",
    )
    parser.add_argument(
        "--no-prefer-teaching-case",
        action="store_true",
        help="Disable automatic preference for teaching-friendly cases.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching projected-footprint failures; do not render.",
    )
    parser.add_argument(
        "--focus-footprint",
        action="store_true",
        help="Teaching view: emphasize projection plane, projected footprint, ray origins, and hits in 3D figures.",
    )
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Render only assembly overview for explicitly requested case(s), without requiring projected-footprint failure candidates.",
    )
    parser.add_argument(
        "--figure03-two-parts",
        action="store_true",
        help="Render only figure_03 for two parts of one selected case as a shared-legend two-panel figure.",
    )
    parser.add_argument(
        "--two-part-targets",
        nargs="+",
        default=["part_0", "part_1"],
        help="Two target part names for --figure03-two-parts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_project_paths(args.project_root)

    jsonl = args.jsonl.expanduser()
    cases_root = args.cases_root.expanduser()
    out_root = args.out.expanduser()
    formats = tuple(args.formats)

    if not cases_root.exists():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")

    if args.overview_only:
        case_arg = args.case
        if isinstance(case_arg, str):
            case_names = [case_arg]
        else:
            case_names = list(case_arg)

        case_names = [c for c in case_names if c not in (None, "", "auto")]

        if not case_names:
            raise ValueError(
                "--overview-only requires explicit --case case_name [case_name ...]"
            )

        out_root.mkdir(parents=True, exist_ok=True)

        generated_all: list[Path] = []

        for case_name in case_names:
            xml_path = resolve_xml_path(None, case_name, cases_root)
            assembly = load_assembly_with_resolved_paths(xml_path)

            target_name = args.target
            if target_name is None or assembly.get_part_by_name(target_name) is None:
                target_name = (
                    "part_0"
                    if assembly.get_part_by_name("part_0") is not None
                    else next(iter(assembly.parts.keys()))
                )

            out_dir = out_root / case_name / f"target_{target_name}_overview"
            out_dir.mkdir(parents=True, exist_ok=True)

            generated_all.extend(
                figure_00_assembly_overview(
                    assembly=assembly,
                    target_name=target_name,
                    out_dir=out_dir,
                    formats=formats,
                    max_faces=args.max_faces,
                )
            )

        print("\nGenerated overview files:")
        for p in generated_all:
            print(p)

        return

    all_candidates = extract_failure_candidates(jsonl, cases_root)

    if not all_candidates:
        print("No projected-footprint ray-bundle failures found.")
        return

    # In teaching mode, automatically prefer cases that are easier to see.
    teaching_flags_on = bool(
        args.zoom_target
        or args.short_rays
        or args.show_only_blocking_neighborhood
        or args.focus_footprint
    )
    prefer_teaching_case = (not args.no_prefer_teaching_case) and (
        teaching_flags_on or args.case in (None, "", "auto")
    )

    case_list = args.case
    if case_list is None:
        case_list = ["auto"]

    case_list = [c for c in case_list if c not in (None, "", "auto")]
    case = None if not case_list else case_list[0]
    candidate_index = (
        None
        if args.candidate_index is None or args.candidate_index < 0
        else args.candidate_index
    )

    selected = choose_candidates(
        all_candidates=all_candidates,
        case=case,
        target=args.target,
        candidate_index=candidate_index,
        max_cases=args.max_cases,
        prefer_teaching_case=prefer_teaching_case,
        min_first_blocking_t=args.min_first_blocking_t,
    )

    if args.list_only or not selected:
        print(f"Found {len(all_candidates)} projected-footprint failures in total.")
        print()

        listed = sorted(
            all_candidates,
            key=lambda c: candidate_score_for_teaching(c, args.min_first_blocking_t),
            reverse=True,
        )

        for c in listed[:150]:
            print(
                f"case={c.assembly_name} "
                f"target={c.target_part} "
                f"attempt={c.attempt_index} "
                f"candidate={c.candidate_index} "
                f"first_t={c.json_first_blocking_t} "
                f"num_blocking={c.json_num_blocking_hits} "
                f"blocking={c.first_blocking_geometry} "
                f"xml={c.xml_path}"
            )

        if not selected and not args.list_only:
            print("\nNo candidate matched the requested filters. Try:")
            print(
                "  --case auto --candidate-index -1 --zoom-target --short-rays --show-only-blocking-neighborhood"
            )
        return

    out_root.mkdir(parents=True, exist_ok=True)

    generated_all: list[Path] = []

    if args.figure03_two_parts:
        cand = selected[0]

        print(
            f"Rendering shared-legend figure_03 for two parts: "
            f"case={cand.assembly_name}, "
            f"targets={args.two_part_targets}, "
            f"candidate_direction={cand.direction.tolist()}"
        )

        assembly = load_assembly_with_resolved_paths(cand.xml_path)

        missing_targets = [
            t for t in args.two_part_targets if assembly.get_part_by_name(t) is None
        ]
        if missing_targets:
            raise ValueError(
                f"Target part(s) not found in assembly {cand.assembly_name}: {missing_targets}. "
                f"Available parts: {list(assembly.parts.keys())}"
            )

        out_dir = out_root / cand.assembly_name / "figure03_two_parts"

        generated_all.extend(
            figure_03_two_parts(
                assembly=assembly,
                target_names=list(args.two_part_targets),
                direction=cand.direction,
                out_dir=out_dir,
                formats=formats,
                pullout_length=args.pullout_length,
                ray_bundle_spacing=args.ray_bundle_spacing,
                max_rays=args.max_rays,
                origin_offset=args.origin_offset,
                hit_margin=args.hit_margin,
                max_faces=args.max_faces,
                zoom_target=args.zoom_target,
                short_rays=args.short_rays,
                show_only_blocking_neighborhood=args.show_only_blocking_neighborhood,
                focus_footprint=args.focus_footprint,
            )
        )

        print("\nGenerated files:")
        for p in generated_all:
            print(p)

        return

    for cand in selected:
        print(
            f"Rendering {cand.assembly_name}, "
            f"target={cand.target_part}, "
            f"candidate={cand.candidate_index}, "
            f"json_first_t={cand.json_first_blocking_t}, "
            f"json_num_blocking={cand.json_num_blocking_hits}, "
            f"blocking={cand.first_blocking_geometry}"
        )

        generated = render_candidate(
            cand=cand,
            out_root=out_root,
            formats=formats,
            pullout_length=args.pullout_length,
            ray_bundle_spacing=args.ray_bundle_spacing,
            max_rays=args.max_rays,
            origin_offset=args.origin_offset,
            hit_margin=args.hit_margin,
            max_faces=args.max_faces,
            zoom_target=args.zoom_target,
            short_rays=args.short_rays,
            show_only_blocking_neighborhood=args.show_only_blocking_neighborhood,
            focus_footprint=args.focus_footprint,
        )
        generated_all.extend(generated)

    print("\nGenerated files:")
    for p in generated_all:
        print(p)


if __name__ == "__main__":
    main()

"""
python tools/rendering_tool/render_secondary_feasibility_real_cases.py \
  --zoom-target \
  --short-rays \
  --show-only-blocking-neighborhood \
  --focus-footprint

cd <repo>

python tools/rendering_tool/render_secondary_feasibility_real_cases.py \
  --jsonl tests/external_cases/results/joint_assembly/test_joint_assembly_1-100_20260508_224553.jsonl \
  --cases-root ../external/test_cases_external \
  --case joint_assembly_00614 \
  --candidate-index -1 \
  --max-cases 7 \
  --formats png \
  --zoom-target \
  --short-rays \
  --show-only-blocking-neighborhood \
  --focus-footprint
"""
