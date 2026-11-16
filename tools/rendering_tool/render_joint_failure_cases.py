from __future__ import annotations

import html
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly

"""../private/figures/failure_case_gallery_2part/index.html"""
# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = ROOT.parent / "private"

RESULT_ROOT = PRIVATE_ROOT / "results"

RESULT_DIRS = {
    "joint_assembly": RESULT_ROOT / "joint_assembly",
    "joint_assembly_rotation": RESULT_ROOT / "joint_assembly_rotation",
}

# Keep this if converted XMLs are still outside the repo here.
CONVERTED_ROOT = ROOT.parent / "external" / "test_cases_external"

# Save generated paper figures / HTML gallery under private/figures
# so they are also kept outside the main repo.
OUT_ROOT = PRIVATE_ROOT / "figures" / "failure_cases_joint_assembly_cases"
IMG_ROOT = OUT_ROOT / "images"
HTML_PATH = OUT_ROOT / "index.html"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
IMG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

RENDER_WIDTH = 1200
RENDER_HEIGHT = 900

# Failed rollouts are always included.
INCLUDE_FAILED_ROLLOUTS = True

# Also include successful rollouts that contain informative failed attempts,
# such as sequence-dependent cases where the first part was blocked before
# the other part was removed.
INCLUDE_INTERMEDIATE_FAILURES = True

# Limit can be useful for debugging.
MAX_CASES_PER_GROUP: int | None = None

# Render only selected assemblies. Use None to render all selected cases.
ONLY_ASSEMBLY_NAMES: set[str] | None = None
ONLY_GROUPS: set[str] | None = None

# For candidate direction views, avoid clutter.
MAX_CANDIDATE_DIRECTIONS = 12


# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

KIT_DEEP_BLUE = [20 / 255, 68 / 255, 102 / 255, 1.0]
KIT_BLUE_GREEN = [0 / 255, 131 / 255, 143 / 255, 1.0]
KIT_YELLOW = [238 / 255, 183 / 255, 13 / 255, 1.0]
KIT_RED = [178 / 255, 55 / 255, 44 / 255, 1.0]
WBK_GREY = [59 / 255, 79 / 255, 88 / 255, 1.0]
LIGHT_GREY = [196 / 255, 208 / 255, 207 / 255, 1.0]
VERY_LIGHT_GREY = [0.82, 0.86, 0.86, 1.0]
BLACK = [18 / 255, 22 / 255, 28 / 255, 1.0]
WHITE = [1.0, 1.0, 1.0, 1.0]


DEFAULT_PART_COLORS = [
    KIT_DEEP_BLUE,
    KIT_BLUE_GREEN,
    WBK_GREY,
    KIT_YELLOW,
]


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------


def to_jsonable(x: Any) -> Any:
    if x is None:
        return None

    if isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    return str(x)


def short_json(obj: Any, max_chars: int = 1200) -> str:
    text = json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False)
    if len(text) > max_chars:
        return text[:max_chars] + "\n..."
    return text


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


# ---------------------------------------------------------------------
# Loading reports
# ---------------------------------------------------------------------


def load_jsonl_reports(result_dir: Path, group: str) -> list[dict]:
    """
    Read all JSONL reports in one result folder.

    If an assembly occurs multiple times, keep the newest entry based on
    JSONL file modification time.
    """
    result_dir = Path(result_dir).resolve()
    jsonl_files = sorted(result_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {result_dir}")

    records: list[dict] = []

    for path in jsonl_files:
        mtime = path.stat().st_mtime

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    r = json.loads(line)
                except Exception as exc:
                    print(
                        f"[WARN] Failed to parse JSONL line {line_idx} in {path}: {exc}"
                    )
                    continue

                r["_source_jsonl"] = str(path)
                r["_source_mtime"] = mtime
                r["group"] = r.get("group", group)
                records.append(r)

    latest: dict[str, dict] = {}

    for r in records:
        name = str(r.get("assembly_name", ""))
        if not name:
            continue

        prev = latest.get(name)
        if prev is None or float(r["_source_mtime"]) >= float(prev["_source_mtime"]):
            latest[name] = r

    out = list(latest.values())
    out.sort(key=lambda x: str(x.get("assembly_name", "")))
    return out


def load_all_reports() -> dict[str, list[dict]]:
    return {
        group: load_jsonl_reports(folder, group)
        for group, folder in RESULT_DIRS.items()
    }


# ---------------------------------------------------------------------
# Diagnostics extraction
# ---------------------------------------------------------------------


def get_attempt_debug(attempt: dict) -> dict:
    debug = attempt.get("debug", {})
    return debug if isinstance(debug, dict) else {}


def get_operation_data_from_attempt(attempt: dict) -> dict:
    debug = get_attempt_debug(attempt)
    data = debug.get("operation_data", {})
    return data if isinstance(data, dict) else {}


def attempt_has_candidate_failures(attempt: dict) -> bool:
    data = get_operation_data_from_attempt(attempt)
    cands = data.get("candidate_failures", None)
    return isinstance(cands, list) and len(cands) > 0


def attempt_is_failure(attempt: dict) -> bool:
    return not parse_bool(attempt.get("success", False))


def attempt_is_informative_failure(attempt: dict) -> bool:
    """
    Select failures worth visualizing even inside successful rollouts.
    """
    if not attempt_is_failure(attempt):
        return False

    debug = get_attempt_debug(attempt)

    if attempt_has_candidate_failures(attempt):
        return True

    mask_reason = debug.get("mask_reason")
    why = debug.get("why")
    exec_reason = debug.get("exec_fail_reason")

    if mask_reason in {"geometrically_infeasible", "masked_out"}:
        return True

    if why in {"empty_dof"}:
        return True

    if exec_reason in {"no_feasible_dof_candidate", "check_feasability_failed"}:
        return True

    return False


def categorize_attempt(attempt: dict) -> str:
    debug = get_attempt_debug(attempt)

    if not attempt_is_failure(attempt):
        return "success"

    if debug.get("stage") == "action_construction":
        return "action_construction_failed"

    illegal = parse_bool(debug.get("illegal_action", False))
    mask_reason = debug.get("mask_reason")
    why = debug.get("why")
    illegal_reason = debug.get("illegal_reason")

    if illegal:
        if mask_reason == "geometrically_infeasible" and why == "empty_dof":
            return "empty_dof_masked_out"
        if why == "empty_dof":
            return "empty_dof_masked_out"
        if mask_reason:
            return f"masked_out:{mask_reason}"
        if illegal_reason:
            return f"illegal:{illegal_reason}"
        return "illegal_or_masked_out"

    exec_reason = debug.get("exec_fail_reason")
    failure_stage = debug.get("failure_stage")

    if exec_reason == "no_feasible_dof_candidate":
        if attempt_has_path_blocking(attempt):
            return "path_blocked_after_candidate_dof"
        return "no_feasible_dof_candidate"

    if failure_stage:
        return str(failure_stage)

    return str(debug.get("failure_label", "unknown_failure"))


def attempt_has_path_blocking(attempt: dict) -> bool:
    data = get_operation_data_from_attempt(attempt)
    cands = data.get("candidate_failures", [])

    if not isinstance(cands, list):
        return False

    for c in cands:
        if not isinstance(c, dict):
            continue

        if c.get("path_ray_bundle_reason") == "blocking_hit":
            return True
        if c.get("feasibility_reason") == "removal_path_projected_footprint_hit":
            return True
        if c.get("first_blocking_geometry") is not None:
            return True

    return False


def summarize_report_categories(report: dict) -> list[str]:
    attempts = report.get("attempts", [])
    categories = []

    for a in attempts:
        if attempt_is_failure(a):
            categories.append(categorize_attempt(a))

    if not categories:
        return ["no_failed_attempts"]

    # Preserve order but remove duplicates.
    out = []
    seen = set()
    for c in categories:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def report_should_be_included(report: dict) -> bool:
    success = parse_bool(report.get("success", False))

    if INCLUDE_FAILED_ROLLOUTS and not success:
        return True

    if INCLUDE_INTERMEDIATE_FAILURES and success:
        attempts = report.get("attempts", [])
        return any(attempt_is_informative_failure(a) for a in attempts)

    return False


# ---------------------------------------------------------------------
# Assembly loading
# ---------------------------------------------------------------------


def assembly_xml_path(assembly_name: str) -> Path:
    return CONVERTED_ROOT / assembly_name / "xml" / f"{assembly_name}.xml"


def load_assembly_for_report(report: dict) -> Assembly:
    name = str(report["assembly_name"])

    # Prefer the xml_path recorded in the report if available.
    xml_raw = report.get("xml_path", None)
    if xml_raw:
        xml_path = Path(xml_raw)
    else:
        xml_path = assembly_xml_path(name)

    if not xml_path.exists():
        # Fall back to converted-root convention.
        fallback = assembly_xml_path(name)
        if fallback.exists():
            xml_path = fallback
        else:
            raise FileNotFoundError(
                f"Could not find XML for assembly {name}.\n"
                f"report xml_path={xml_raw}\n"
                f"fallback={fallback}"
            )

    return Assembly.load_from_xml(str(xml_path))


def get_part_mesh(part) -> o3d.geometry.TriangleMesh | None:
    mesh = getattr(part, "mesh", None)

    if isinstance(mesh, o3d.geometry.TriangleMesh):
        m = o3d.geometry.TriangleMesh(mesh)
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
        return m

    mesh_path = getattr(part, "mesh_path", None)

    if mesh_path:
        p = Path(mesh_path)
        if p.exists():
            m = o3d.io.read_triangle_mesh(str(p))
            if not m.is_empty():
                if not m.has_vertex_normals():
                    m.compute_vertex_normals()
                return m

    return None


def assembly_meshes(asm: Assembly) -> dict[str, o3d.geometry.TriangleMesh]:
    meshes: dict[str, o3d.geometry.TriangleMesh] = {}

    for part in asm.parts.values():
        name = str(getattr(part, "name", ""))
        if not name:
            continue

        m = get_part_mesh(part)
        if m is not None:
            meshes[name] = m

    return meshes


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def compute_bounds(
    meshes: dict[str, o3d.geometry.TriangleMesh],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mins = []
    maxs = []

    for m in meshes.values():
        aabb = m.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.min_bound, dtype=float))
        maxs.append(np.asarray(aabb.max_bound, dtype=float))

    if not mins:
        raise ValueError("No mesh bounds available")

    min_bound = np.min(np.vstack(mins), axis=0)
    max_bound = np.max(np.vstack(maxs), axis=0)

    center = 0.5 * (min_bound + max_bound)
    diag = float(np.linalg.norm(max_bound - min_bound))
    diag = max(diag, 1e-6)

    return min_bound, max_bound, center, diag


def mesh_center(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    aabb = mesh.get_axis_aligned_bounding_box()
    return np.asarray(aabb.get_center(), dtype=float)


def normalize(v: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(arr))
        if not np.isfinite(n) or n <= 1e-9:
            return None
        return arr / n
    except Exception:
        return None


def camera_up_for_view(view_dir: np.ndarray) -> np.ndarray:
    """
    Choose a stable camera up vector for axis and oblique views.
    """
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    y = np.array([0.0, 1.0, 0.0], dtype=float)

    view_dir = normalize(view_dir)
    if view_dir is None:
        return z

    if abs(float(np.dot(view_dir, z))) > 0.92:
        return y

    return z


def safe_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def make_material(
    color_rgba: list[float],
    *,
    transparent: bool = False,
) -> o3d.visualization.rendering.MaterialRecord:
    mat = o3d.visualization.rendering.MaterialRecord()

    if transparent or color_rgba[3] < 0.999:
        mat.shader = "defaultLitTransparency"
    else:
        mat.shader = "defaultLit"

    mat.base_color = color_rgba
    return mat


def make_line_material(
    color_rgba: list[float],
    *,
    width: float = 5.0,
) -> o3d.visualization.rendering.MaterialRecord:
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = color_rgba
    mat.line_width = width
    return mat


def render_mesh_scene(
    *,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    out_path: Path,
    title: str | None = None,
    view_dir: tuple[float, float, float] | np.ndarray = (1.4, -1.4, 0.9),
    color_by_part: dict[str, list[float]] | None = None,
    alpha_by_part: dict[str, float] | None = None,
    extra_geometries: list[
        tuple[str, o3d.geometry.Geometry, o3d.visualization.rendering.MaterialRecord]
    ]
    | None = None,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
    fov_deg: float = 42.0,
) -> Path:
    """
    Render one assembly view using Open3D OffscreenRenderer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not meshes:
        raise ValueError("No meshes to render")

    _, _, center, diag = compute_bounds(meshes)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    scene.set_background(WHITE)

    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=[-0.5, -0.5, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=65000,
    )
    scene.scene.enable_indirect_light(False)

    color_by_part = color_by_part or {}
    alpha_by_part = alpha_by_part or {}

    for idx, (name, mesh) in enumerate(meshes.items()):
        rgba = list(
            color_by_part.get(name, DEFAULT_PART_COLORS[idx % len(DEFAULT_PART_COLORS)])
        )
        if len(rgba) == 3:
            rgba = [rgba[0], rgba[1], rgba[2], 1.0]

        if name in alpha_by_part:
            rgba[3] = float(alpha_by_part[name])

        mat = make_material(rgba, transparent=rgba[3] < 0.999)
        scene.add_geometry(f"part_{idx}_{name}", mesh, mat)

    if extra_geometries:
        for idx, (name, geom, mat) in enumerate(extra_geometries):
            scene.add_geometry(f"extra_{idx}_{name}", geom, mat)

    view = normalize(view_dir)
    if view is None:
        view = normalize((1.4, -1.4, 0.9))

    # Eye lies in view_dir from the object and looks toward center.
    eye = center + view * diag * 2.35
    up = camera_up_for_view(view)

    renderer.setup_camera(fov_deg, center, eye, up)

    # Warm-up render helps avoid occasional blank first frame.
    _ = renderer.render_to_image()
    img = renderer.render_to_image()

    o3d.io.write_image(str(out_path), img)

    scene.clear_geometry()
    del renderer

    if title is not None:
        add_title_to_image(out_path, title)

    return out_path


def add_title_to_image(path: Path, title: str) -> None:
    """
    Add a compact title strip using matplotlib.

    This avoids relying on Open3D text rendering.
    """
    img = imageio.imread(path)

    fig, ax = plt.subplots(figsize=(img.shape[1] / 180, img.shape[0] / 180), dpi=180)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=3)

    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def make_six_axis_panel(
    *,
    image_paths: list[tuple[str, Path]],
    out_path: Path,
    title: str,
) -> Path:
    """
    Combine six axis-view images into one diagnostic panel.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(image_paths)
    cols = 3
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 5.8), dpi=180)
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, (label, path) in zip(axes_arr, image_paths):
        img = imageio.imread(path)
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    for ax in axes_arr[len(image_paths) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------
# Candidate direction visualization
# ---------------------------------------------------------------------


def extract_candidate_failures(report: dict) -> list[dict]:
    out = []

    for attempt in report.get("attempts", []):
        data = get_operation_data_from_attempt(attempt)
        cands = data.get("candidate_failures", None)

        if isinstance(cands, list) and cands:
            out.append(
                {
                    "attempt": attempt,
                    "target": attempt.get("target"),
                    "candidate_failures": cands,
                    "operation_data": data,
                }
            )

    return out


def extract_candidate_directions(candidate_failures: list[dict]) -> list[np.ndarray]:
    dirs = []

    for cand in candidate_failures:
        if not isinstance(cand, dict):
            continue

        for key in [
            "direction",
            "render_direction",
            "candidate_direction",
            "dof_direction",
        ]:
            if key in cand:
                v = normalize(cand.get(key))
                if v is not None:
                    dirs.append(v)
                    break

    # Remove near-duplicates.
    unique = []
    for v in dirs:
        if not any(abs(float(np.dot(v, u))) > 0.995 for u in unique):
            unique.append(v)

    return unique[:MAX_CANDIDATE_DIRECTIONS]


def extract_blocking_geometries(candidate_failures: list[dict]) -> list[str]:
    blockers = []

    for cand in candidate_failures:
        if not isinstance(cand, dict):
            continue

        for key in [
            "first_blocking_geometry",
            "blocking_geometry",
            "blocker",
            "blocking_part",
            "blocking_part_name",
        ]:
            value = cand.get(key, None)
            if isinstance(value, str) and value:
                blockers.append(value)

    out = []
    seen = set()
    for b in blockers:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def make_direction_lines(
    *,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    target_part: str | None,
    directions: list[np.ndarray],
) -> o3d.geometry.LineSet | None:
    if not directions:
        return None

    _, _, assembly_center, diag = compute_bounds(meshes)

    if target_part is not None and target_part in meshes:
        origin = mesh_center(meshes[target_part])
    else:
        origin = assembly_center

    length = diag * 0.55

    points = []
    lines = []

    for i, d in enumerate(directions):
        start = origin - d * length * 0.12
        end = origin + d * length

        points.append(start)
        points.append(end)
        lines.append([2 * i, 2 * i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.paint_uniform_color(KIT_YELLOW[:3])

    return line_set


def render_candidate_diagnostic(
    *,
    meshes: dict[str, o3d.geometry.TriangleMesh],
    report: dict,
    case_img_dir: Path,
) -> list[Path]:
    """
    Render candidate direction views for attempts that contain candidate_failures.
    """
    outputs = []
    cand_attempts = extract_candidate_failures(report)

    for idx, item in enumerate(cand_attempts):
        target = item.get("target")
        candidate_failures = item["candidate_failures"]

        directions = extract_candidate_directions(candidate_failures)
        blockers = extract_blocking_geometries(candidate_failures)

        if not directions:
            continue

        color_by_part = {}
        alpha_by_part = {}

        for j, name in enumerate(meshes.keys()):
            color_by_part[name] = DEFAULT_PART_COLORS[j % len(DEFAULT_PART_COLORS)]
            alpha_by_part[name] = 0.35

        if isinstance(target, str) and target in meshes:
            color_by_part[target] = KIT_RED
            alpha_by_part[target] = 1.0

        for b in blockers:
            if b in meshes:
                color_by_part[b] = KIT_DEEP_BLUE
                alpha_by_part[b] = 0.85

        line_set = make_direction_lines(
            meshes=meshes,
            target_part=target if isinstance(target, str) else None,
            directions=directions,
        )

        extra = []
        if line_set is not None:
            extra.append(
                (
                    "candidate_directions",
                    line_set,
                    make_line_material(KIT_YELLOW, width=7.0),
                )
            )

        title = f"Candidate directions for {target} | blockers: {', '.join(blockers) if blockers else 'n/a'}"

        out_path = case_img_dir / f"candidate_directions_{idx:02d}.png"

        render_mesh_scene(
            meshes=meshes,
            out_path=out_path,
            title=title,
            view_dir=(1.4, -1.4, 0.9),
            color_by_part=color_by_part,
            alpha_by_part=alpha_by_part,
            extra_geometries=extra,
            fov_deg=38.0,
        )

        outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------
# Case rendering
# ---------------------------------------------------------------------


def overview_colors(
    meshes: dict[str, o3d.geometry.TriangleMesh],
) -> dict[str, list[float]]:
    colors = {}
    for idx, name in enumerate(meshes.keys()):
        colors[name] = DEFAULT_PART_COLORS[idx % len(DEFAULT_PART_COLORS)]
    return colors


def target_highlight_colors(
    meshes: dict[str, o3d.geometry.TriangleMesh],
    target: str,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    color_by_part = {}
    alpha_by_part = {}

    for name in meshes.keys():
        if name == target:
            color_by_part[name] = KIT_RED
            alpha_by_part[name] = 1.0
        else:
            color_by_part[name] = WBK_GREY
            alpha_by_part[name] = 0.38

    return color_by_part, alpha_by_part


def render_case(report: dict) -> dict:
    """
    Render all diagnostic images for one case and return an HTML-friendly record.
    """
    assembly_name = str(report["assembly_name"])
    group = str(report.get("group", "unknown"))
    success = parse_bool(report.get("success", False))

    case_dir = IMG_ROOT / group / safe_filename(assembly_name)
    case_dir.mkdir(parents=True, exist_ok=True)

    asm = load_assembly_for_report(report)
    meshes = assembly_meshes(asm)

    if not meshes:
        raise ValueError(f"No meshes found for assembly {assembly_name}")

    image_paths: dict[str, Path | list[Path]] = {}

    # 1) Isometric overview.
    iso_path = case_dir / "overview_iso.png"
    render_mesh_scene(
        meshes=meshes,
        out_path=iso_path,
        title=f"{assembly_name} | overview",
        view_dir=(1.4, -1.4, 0.9),
        color_by_part=overview_colors(meshes),
        fov_deg=40.0,
    )
    image_paths["overview_iso"] = iso_path

    # 2) Six-axis panel.
    axis_views = [
        ("+X", (1, 0, 0)),
        ("-X", (-1, 0, 0)),
        ("+Y", (0, 1, 0)),
        ("-Y", (0, -1, 0)),
        ("+Z", (0, 0, 1)),
        ("-Z", (0, 0, -1)),
    ]

    axis_image_paths = []
    for label, view in axis_views:
        p = case_dir / f"axis_{label.replace('+', 'pos').replace('-', 'neg')}.png"
        render_mesh_scene(
            meshes=meshes,
            out_path=p,
            title=None,
            view_dir=view,
            color_by_part=overview_colors(meshes),
            fov_deg=35.0,
        )
        axis_image_paths.append((label, p))

    panel_path = case_dir / "six_axis_panel.png"
    make_six_axis_panel(
        image_paths=axis_image_paths,
        out_path=panel_path,
        title=f"{assembly_name} | six-axis views",
    )
    image_paths["six_axis_panel"] = panel_path

    # 3) Target-highlight views.
    target_views = []
    target_names = sorted(meshes.keys())

    for target in target_names:
        color_by_part, alpha_by_part = target_highlight_colors(meshes, target)
        p = case_dir / f"target_{safe_filename(target)}.png"

        render_mesh_scene(
            meshes=meshes,
            out_path=p,
            title=f"Target highlight: {target}",
            view_dir=(1.4, -1.4, 0.9),
            color_by_part=color_by_part,
            alpha_by_part=alpha_by_part,
            fov_deg=38.0,
        )

        target_views.append(p)

    image_paths["target_views"] = target_views

    # 4) Candidate direction views, only if candidate_failures exist.
    candidate_paths = render_candidate_diagnostic(
        meshes=meshes,
        report=report,
        case_img_dir=case_dir,
    )
    image_paths["candidate_views"] = candidate_paths

    categories = summarize_report_categories(report)

    return {
        "assembly_name": assembly_name,
        "group": group,
        "success": success,
        "stop_reason": report.get("stop_reason"),
        "completion": report.get("completion"),
        "initial_num_parts": report.get("initial_num_parts"),
        "initial_num_connections": report.get("initial_num_connections"),
        "num_attempts": report.get("num_attempts"),
        "num_env_steps": report.get("num_env_steps"),
        "categories": categories,
        "report": report,
        "images": image_paths,
    }


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(OUT_ROOT.resolve()))


def html_escape(x: Any) -> str:
    return html.escape(str(x))


def attempt_summary_rows(report: dict) -> str:
    rows = []

    for a in report.get("attempts", []):
        debug = get_attempt_debug(a)
        data = get_operation_data_from_attempt(a)

        target = a.get("target")
        success = parse_bool(a.get("success", False))
        category = categorize_attempt(a)
        failure_label = debug.get("failure_label")
        mask_reason = debug.get("mask_reason")
        why = debug.get("why")
        exec_reason = debug.get("exec_fail_reason")

        cand_count = None
        cands = data.get("candidate_failures", None)
        if isinstance(cands, list):
            cand_count = len(cands)

        rows.append(f"""
        <tr>
          <td>{html_escape(a.get("step"))}</td>
          <td>{html_escape(target)}</td>
          <td>{html_escape(success)}</td>
          <td>{html_escape(category)}</td>
          <td>{html_escape(failure_label)}</td>
          <td>{html_escape(mask_reason)}</td>
          <td>{html_escape(why)}</td>
          <td>{html_escape(exec_reason)}</td>
          <td>{html_escape(cand_count if cand_count is not None else "")}</td>
        </tr>
        """)

    return "\n".join(rows)


def compact_debug_block(report: dict) -> str:
    compact = {
        "assembly_name": report.get("assembly_name"),
        "success": report.get("success"),
        "stop_reason": report.get("stop_reason"),
        "completion": report.get("completion"),
        "removed_parts": report.get("removed_parts"),
        "remaining_parts": report.get("remaining_parts"),
        "initial_num_parts": report.get("initial_num_parts"),
        "initial_num_connections": report.get("initial_num_connections"),
        "num_attempts": report.get("num_attempts"),
        "num_env_steps": report.get("num_env_steps"),
    }
    return short_json(compact, max_chars=1600)


def image_tag(path: Path, caption: str, *, wide: bool = False) -> str:
    cls = "imgbox wide" if wide else "imgbox"
    return f"""
    <figure class="{cls}">
      <img src="{html_escape(rel(path))}" loading="lazy">
      <figcaption>{html_escape(caption)}</figcaption>
    </figure>
    """


def build_html(records: list[dict]) -> None:
    category_counter = Counter()
    group_counter = Counter()

    for r in records:
        group_counter[r["group"]] += 1
        for c in r["categories"]:
            category_counter[c] += 1

    summary_rows = "\n".join(
        f"<tr><td>{html_escape(k)}</td><td>{v}</td></tr>"
        for k, v in category_counter.most_common()
    )

    group_rows = "\n".join(
        f"<tr><td>{html_escape(k)}</td><td>{v}</td></tr>"
        for k, v in group_counter.most_common()
    )

    cards = []

    for rec in records:
        report = rec["report"]
        images = rec["images"]

        image_html = []
        image_html.append(image_tag(images["overview_iso"], "Isometric overview"))
        image_html.append(
            image_tag(images["six_axis_panel"], "Six-axis diagnostic panel", wide=True)
        )

        for p in images.get("target_views", []):
            image_html.append(image_tag(p, p.stem.replace("target_", "Target: ")))

        for p in images.get("candidate_views", []):
            image_html.append(
                image_tag(p, "Candidate directions / blockers", wide=True)
            )

        categories = ", ".join(rec["categories"])

        cards.append(f"""
        <section class="card">
          <h2>{html_escape(rec["assembly_name"])}</h2>
          <div class="meta">
            <span><b>Group:</b> {html_escape(rec["group"])}</span>
            <span><b>Success:</b> {html_escape(rec["success"])}</span>
            <span><b>Stop:</b> {html_escape(rec["stop_reason"])}</span>
            <span><b>Completion:</b> {html_escape(rec["completion"])}</span>
            <span><b>Parts:</b> {html_escape(rec["initial_num_parts"])}</span>
            <span><b>Connections:</b> {html_escape(rec["initial_num_connections"])}</span>
            <span><b>Categories:</b> {html_escape(categories)}</span>
          </div>

          <div class="gallery">
            {"".join(image_html)}
          </div>

          <details open>
            <summary>Attempt summary</summary>
            <table>
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Target</th>
                  <th>Success</th>
                  <th>Category</th>
                  <th>Failure label</th>
                  <th>Mask reason</th>
                  <th>Why</th>
                  <th>Exec reason</th>
                  <th>Candidate failures</th>
                </tr>
              </thead>
              <tbody>
                {attempt_summary_rows(report)}
              </tbody>
            </table>
          </details>

          <details>
            <summary>Compact report</summary>
            <pre>{html_escape(compact_debug_block(report))}</pre>
          </details>
        </section>
        """)

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>2-Part Failure Diagnostic Gallery</title>
<style>
  body {{
    margin: 24px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #222;
    background: #f7f8f8;
  }}
  h1 {{
    margin-bottom: 0.2rem;
  }}
  .subtitle {{
    color: #555;
    margin-bottom: 1.5rem;
  }}
  .summary {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-bottom: 24px;
  }}
  .summary-box {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 0.9rem;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 5px 7px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background: #eef1f2;
  }}
  .card {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 14px;
    padding: 18px;
    margin: 0 0 28px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    margin: 10px 0 16px;
    color: #333;
  }}
  .gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin: 12px 0 16px;
  }}
  .imgbox {{
    margin: 0;
    border: 1px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
  }}
  .imgbox.wide {{
    grid-column: span 2;
  }}
  .imgbox img {{
    width: 100%;
    display: block;
  }}
  figcaption {{
    padding: 6px 10px;
    font-size: 0.9rem;
    color: #444;
    background: #f2f4f4;
  }}
  details {{
    margin-top: 12px;
  }}
  pre {{
    background: #f3f5f5;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
  }}
</style>
</head>
<body>
  <h1>2-Part Failure Diagnostic Gallery</h1>
  <div class="subtitle">
    Failed rollouts and selected informative intermediate failures from joint and rotational-joint external cases.
  </div>

  <div class="summary">
    <div class="summary-box">
      <h3>Groups</h3>
      <table>
        <thead><tr><th>Group</th><th>Cases</th></tr></thead>
        <tbody>{group_rows}</tbody>
      </table>
    </div>

    <div class="summary-box">
      <h3>Failure categories</h3>
      <table>
        <thead><tr><th>Category</th><th>Count</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
  </div>

  {"".join(cards)}
</body>
</html>
"""

    HTML_PATH.write_text(html_text, encoding="utf-8")
    print(f"[SAVED HTML] {HTML_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    all_reports_by_group = load_all_reports()

    selected_reports = []

    # for group, reports in all_reports_by_group.items():
    #    selected = [r for r in reports if report_should_be_included(r)]
    #
    #    if MAX_CASES_PER_GROUP is not None:
    #        selected = selected[:MAX_CASES_PER_GROUP]
    #
    #    print("=" * 80)
    #    print(f"[GROUP] {group}")
    #    print(f"[INFO] total reports:    {len(reports)}")
    #    print(f"[INFO] selected reports: {len(selected)}")
    #
    #    selected_reports.extend(selected)
    for group, reports in all_reports_by_group.items():
        if ONLY_GROUPS is not None and group not in ONLY_GROUPS:
            continue

        selected = [r for r in reports if report_should_be_included(r)]

        if ONLY_ASSEMBLY_NAMES is not None:
            selected = [
                r
                for r in selected
                if str(r.get("assembly_name", "")) in ONLY_ASSEMBLY_NAMES
            ]

        if MAX_CASES_PER_GROUP is not None:
            selected = selected[:MAX_CASES_PER_GROUP]

        print("=" * 80)
        print(f"[GROUP] {group}")
        print(f"[INFO] total reports:    {len(reports)}")
        print(f"[INFO] selected reports: {len(selected)}")

        selected_reports.extend(selected)

    records = []
    errors = []

    for idx, report in enumerate(selected_reports, start=1):
        name = str(report.get("assembly_name", f"case_{idx}"))
        group = str(report.get("group", "unknown"))

        print("=" * 80)
        print(f"[RENDER] [{idx}/{len(selected_reports)}] {group} / {name}")

        try:
            rec = render_case(report)
            records.append(rec)
            print(f"[OK] {name}")
        except Exception as exc:
            err = {
                "assembly_name": name,
                "group": group,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"[ERROR] {name}: {err['error']}")

    build_html(records)

    if errors:
        error_path = OUT_ROOT / "render_errors.json"
        error_path.write_text(
            json.dumps(to_jsonable(errors), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[WARN] {len(errors)} cases failed during rendering.")
        print(f"[SAVED ERRORS] {error_path}")

    print("=" * 80)
    print(f"[DONE] rendered records: {len(records)}")
    print(f"[DONE] output folder: {OUT_ROOT}")
    print(f"[DONE] html: {HTML_PATH}")


if __name__ == "__main__":
    main()
