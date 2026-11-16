from __future__ import annotations

import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import disassembly_pomdp  # noqa: F401  (puts the subtree folders on sys.path)

# isort: split

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly

from crc_disassembly_sequence_generator.operations import (
    GrippingAndRemovingOperation,
    ScrewingOperation,
)
from crc_disassembly_sequence_generator.planning import (
    HierarchicalDisassemblySequencePlanner,
)
from disassembly_pomdp.config import CellSettings
from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)

# Import your working POMDP observer implementation
from disassembly_pomdp.pomdp_observer import ObserverConfig, SimVisibilityObserver
from disassembly_pomdp.screwing_capability import ScrewingCapability

CASE_CONFIG = {
    "joint_assembly_00009": {
        "base_parts": ("part_0",),
        "target_part_names": ["part_1"],
        "enable_screwing": False,
    },
    "joint_assembly_00016": {
        "base_parts": ("part_0",),
        "target_part_names": ["part_1"],
        "enable_screwing": False,
    },
    "joint_assembly_00030": {
        "base_parts": ("part_0",),
        "target_part_names": ["part_1"],
        "enable_screwing": False,
    },
    "joint_assembly_00125": {
        "base_parts": ("part_0",),
        "target_part_names": ["part_1"],
        "enable_screwing": False,
    },
    "joint_assembly_00201": {
        "base_parts": ("part_0",),
        "target_part_names": ["part_1"],
        "enable_screwing": False,
    },
}

"""
CASE_CONFIG = {
    "multi_assembly_00306": {
        "base_parts": ("part_0",),
        "enable_screwing": False,
    },
}
"""


def rgb255_to_rgb01(rgb):
    arr = np.asarray(rgb, dtype=np.float64)
    return (arr / 255.0).tolist()


KIT_DEEP_BLUE = rgb255_to_rgb01([20, 68, 102])  # #144466
KIT_BLUE_GREEN = rgb255_to_rgb01([0, 131, 143])  # #00838f
KIT_YELLOW = rgb255_to_rgb01([238, 183, 13])  # #EEB70D
KIT_RED = rgb255_to_rgb01([178, 55, 44])  # #B2372C

WBK_GREEN = rgb255_to_rgb01([0, 150, 130])
WBK_GREY = rgb255_to_rgb01([59, 79, 88])
WBK_LIGHT_GREY = rgb255_to_rgb01([196, 208, 207])
WBK_LIGHT_BLUE = rgb255_to_rgb01([60, 152, 217])
WBK_LIGHT_GREEN = rgb255_to_rgb01([39, 255, 226])

BLACK = rgb255_to_rgb01([18, 22, 28])

ASSEMBLY_PALETTE = [
    KIT_DEEP_BLUE,
    KIT_RED,
    KIT_BLUE_GREEN,
    BLACK,
    WBK_GREEN,
    WBK_LIGHT_BLUE,
    WBK_LIGHT_GREEN,
]


def to_jsonable(x: Any):
    """Best-effort conversion of debug objects into JSON-serializable values."""
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

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            # Avoid dumping full assemblies or large geometry objects
            if k in {"result_assembly", "assembly", "mesh", "meshes"}:
                out[k] = f"<omitted:{type(v).__name__}>"
            else:
                out[str(k)] = to_jsonable(v)
        return out

    # Avoid huge custom object serialization
    return str(x)


def summarize_operation_data(data: dict) -> dict:
    """
    Compact operation.data for large-scale debug logs.

    Keeps diagnostic fields, but avoids dumping huge arrays, paths,
    meshes, result assemblies, or full custom objects.
    """
    if not isinstance(data, dict):
        return {}

    keep_keys = [
        "type",
        "part_name",
        "screw_part_name",
        "base_part_name",
        "cost",
        "planned",
        "direction",
        "render_direction",
        "path_length",
        "path_duration",
        "screw_search_duration",
        "motion_program",
        "grip_width",
        "max_grip_width",
        "tool_name",
        # large-scale feasibility/debug
        "last_feasibility_debug",
        "failure_stage",
        "failure_reason",
        "dof_candidate_count",
        "raw_dof_count",
        "invalid_dof_count",
        "candidate_failures",
        "candidate_failures_before_selected",
        "secondary_path_lock_enabled",
        "secondary_path_lock_should_apply",
        "secondary_path_lock_applied",
        "secondary_path_lock_current_parts",
        "secondary_path_lock_current_geometric_neighbors",
        "secondary_path_lock_added_fixed_connections",
        "secondary_path_lock_added_count",
        "reverse_secondary_path_lock_enabled",
        "reverse_secondary_path_lock_applied",
        "reverse_secondary_path_lock_rejected_part",
        "reverse_secondary_path_lock_checked_neighbors",
        "reverse_secondary_path_lock_blocking_neighbors",
        "reverse_secondary_path_lock_added_fixed_connections",
        "reverse_secondary_path_lock_added_count",
        "reverse_secondary_path_lock_neighbor_debug",
        "would_have_selected_direction",
        "would_have_selected_path_length",
        "would_have_selected_path_duration",
    ]

    out = {}

    for k in keep_keys:
        if k in data:
            out[k] = to_jsonable(data[k])

    # Compact path summary instead of full path array
    path = data.get("path", None)
    if path is not None:
        try:
            arr = np.asarray(path, dtype=float)
            out["path_shape"] = list(arr.shape)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3:
                out["path_start"] = arr[0].tolist()
                out["path_end"] = arr[-1].tolist()
        except Exception as exc:
            out["path_summary_error"] = f"{type(exc).__name__}: {exc}"

    # Compact connection
    conn = data.get("connection", None)
    if conn is not None:
        out["connection"] = serialize_connection(conn)

    # Keep potentially useful planner / feasibility flags if present
    for k, v in data.items():
        lk = str(k).lower()
        if any(
            token in lk
            for token in [
                "fail",
                "reason",
                "feasible",
                "feasibility",
                "collision",
                "dof",
                "direction",
                "warning",
                "error",
                "ray",
                "hit",
                "blocking",
                "lock",
            ]
        ):
            if k not in out and k not in {
                "result_assembly",
                "assembly",
                "mesh",
                "meshes",
            }:
                out[str(k)] = to_jsonable(v)

    return out


def serialize_connection(conn) -> dict:
    """Serialize a connection into a compact dictionary."""
    return {
        "type": str(getattr(conn, "connection_type", type(conn).__name__)),
        "part_name1": getattr(conn, "part_name1", None),
        "part_name2": getattr(conn, "part_name2", None),
        "name": getattr(conn, "name", None),
        "repr": str(conn),
    }


def serialize_connections(asm: Assembly) -> list[dict]:
    return [serialize_connection(c) for c in asm.connections]


def get_screw_part_names(asm: Assembly) -> set[str]:
    screw_names = set()
    for c in asm.connections:
        if getattr(c, "connection_type", None) == "screw":
            if getattr(c, "part_name1", None) is not None:
                screw_names.add(c.part_name1)
    return screw_names


def load_imported_assembly(xml_path: str | Path) -> Assembly:
    xml_path = Path(xml_path).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    return Assembly.load_from_xml(str(xml_path))


def build_part_color_map(asm: Assembly) -> dict[str, list[float]]:
    screw_names = get_screw_part_names(asm)
    color_map: dict[str, list[float]] = {}

    palette_idx = 0
    for part in asm.parts.values():
        if part.name in screw_names:
            color_map[part.name] = WBK_GREY
        else:
            color_map[part.name] = ASSEMBLY_PALETTE[palette_idx % len(ASSEMBLY_PALETTE)]
            palette_idx += 1

    return color_map


def paint_mesh_copy(
    mesh: o3d.geometry.TriangleMesh, color
) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m


def make_cell_settings() -> CellSettings:
    return CellSettings(
        feed_rate_milling=1000.0,
        feed_rate_gripping_and_removing_approaching=2000.0,
        feed_rate_gripping_and_removing_leaving=1000.0,
        feed_rate_screwing_approaching=2000.0,
        feed_rate_screwing_leaving=1000.0,
        toolchange_time_dropoff_to_storage=10.0,
        toolchange_time_assembly_to_storage=10.0,
        toolchange_time_at_toolstorage=60.0,
        toolchange_time_storage_to_assembly=10.0,
        toolchange_time_start_to_toolstorage=10.0,
        traveltime_assembly_to_dropoff=15.0,
        traveltime_dropoff_to_assembly=15.0,
        screwing_start_offset=100.0,
        screwing_pullout_factor=1.1,
        screwing_minimal_pullout_value=5.0,
        gripping_start_offset=100.0,
        gripping_pullout_factor=2.1,
        gripping_minimal_pullout_value=5.0,
        milling_penalty=100.0,
    )


def make_planner(asm: Assembly) -> HierarchicalDisassemblySequencePlanner:
    return HierarchicalDisassemblySequencePlanner(
        assembly=asm,
        cost_milling=10.0,
        cost_screwing=5.0,
        cost_gripping_and_removing=0.0,
        cost_safety_ring=5.0,
        mill_diameter=6.0,
        path_to_working_dir=None,
        available_screwdrivers=["H5", "M4"],
    )


def make_plan_screwing_adapter(planner: HierarchicalDisassemblySequencePlanner):
    def _merge_without_result_assembly(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if k == "result_assembly":
                continue
            dst[k] = v

    def plan_screwing_adapter(assembly: Assembly, op: ScrewingOperation, rng=None):
        data = getattr(op, "data", {})
        if not isinstance(data, dict):
            return op, None

        connection = data.get("connection")
        if connection is None:
            return op, None

        cost_screwing = float(data.get("cost", 5.0))
        planned_op, candidate_asm = planner.plan_screwing(
            assembly=assembly,
            connection=connection,
            cost_screwing=cost_screwing,
        )

        out_op = planned_op if planned_op is not None else op
        out_data: dict = {}
        _merge_without_result_assembly(out_data, data)

        if isinstance(getattr(out_op, "data", None), dict):
            _merge_without_result_assembly(out_data, out_op.data)

        out_data["planned"] = True
        screw_name = out_data.get("screw_part_name") or getattr(
            connection, "part_name1", None
        )

        valid_new_asm = (
            isinstance(candidate_asm, Assembly)
            and screw_name is not None
            and candidate_asm.get_part_by_name(screw_name) is None
        )

        if valid_new_asm:
            out_data["result_assembly"] = candidate_asm
            out_op.data = out_data
            return out_op, candidate_asm

        out_op.data = out_data
        return out_op, None

    return plan_screwing_adapter


def make_plan_gripping_and_removing_adapter(
    planner: HierarchicalDisassemblySequencePlanner,
):
    def _merge_without_result_assembly(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if k == "result_assembly":
                continue
            dst[k] = v

    def plan_gripping_and_removing_adapter(
        assembly: Assembly, op: GrippingAndRemovingOperation, rng=None
    ):
        data = getattr(op, "data", {})
        if not isinstance(data, dict):
            return op, None

        part_name = data.get("part_name")
        if part_name is None:
            return op, None

        part = assembly.get_part_by_name(part_name)
        if part is None:
            return op, None

        cost_gripping_and_removing = float(data.get("cost", 5.0))
        max_grip_width = 1e6  # universal gripper assumption

        planned_op, candidate_asm = planner.plan_gripping_and_removing(
            assembly=assembly,
            part=part,
            cost_gripping_and_removing=cost_gripping_and_removing,
            max_grip_width=max_grip_width,
        )

        out_op = planned_op if planned_op is not None else op
        out_data: dict = {}
        _merge_without_result_assembly(out_data, data)

        if isinstance(getattr(out_op, "data", None), dict):
            _merge_without_result_assembly(out_data, out_op.data)

        out_data["planned"] = True

        valid_new_asm = (
            isinstance(candidate_asm, Assembly)
            and part_name is not None
            and candidate_asm.get_part_by_name(part_name) is None
        )

        if valid_new_asm:
            out_data["result_assembly"] = candidate_asm
            out_op.data = out_data
            return out_op, candidate_asm

        out_op.data = out_data
        return out_op, None

    return plan_gripping_and_removing_adapter


def has_screw_connections(asm: Assembly) -> bool:
    for c in asm.connections:
        if getattr(c, "connection_type", None) == "screw":
            return True
    return False


def make_external_case_observer(debug: bool = True) -> SimVisibilityObserver:
    return SimVisibilityObserver(
        cfg=ObserverConfig(
            viewing_directions=(
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, -1),
            ),
            debug=debug,
        )
    )


def make_env_from_imported_xml(xml_path: str | Path, observer) -> DisassemblyPOMDPEnv:
    if observer is None:
        raise ValueError("POMDP external-case env requires a real observer.")

    base_asm = load_imported_assembly(xml_path)
    settings = make_cell_settings()
    case_name = base_asm.name

    cfg = CASE_CONFIG.get(case_name, None)
    if cfg is None:
        raise KeyError(
            f"No CASE_CONFIG found for assembly '{case_name}'. "
            "Please add base_parts and target_part_names manually for this case."
        )

    all_part_names = sorted([p.name for p in base_asm.parts.values()])
    base_parts = tuple(cfg["base_parts"])
    cfg_target = cfg.get("target_part_names", None)

    if cfg_target is None:
        target_part_names = [n for n in all_part_names if n not in set(base_parts)]
    else:
        target_part_names = list(cfg_target)

    planner = make_planner(copy.deepcopy(base_asm))
    plan_screwing_adapter = make_plan_screwing_adapter(planner)
    plan_gripping_and_removing_adapter = make_plan_gripping_and_removing_adapter(
        planner
    )

    capabilities = []

    enable_screwing = bool(cfg.get("enable_screwing", False)) and has_screw_connections(
        base_asm
    )
    if enable_screwing:
        screwing = ScrewingCapability(
            available_screwdrivers=["H5", "M4"],
            cell_settings=settings,
            operation_cls=ScrewingOperation,
            plan_func=plan_screwing_adapter,
        )
        capabilities.append(screwing)

    gripping_and_removing = GrippingAndRemovingCapability.from_assembly(
        base_asm,
        available_grippers=["H5"],
        cell_settings=settings,
        operation_cls=GrippingAndRemovingOperation,
        plan_func=plan_gripping_and_removing_adapter,
        base_parts=tuple(cfg["base_parts"]),
    )
    capabilities.append(gripping_and_removing)

    env = DisassemblyPOMDPEnv(
        assembly=base_asm,
        capabilities=capabilities,
        cell_settings=settings,
        target_part_names=target_part_names,
        max_steps=80,
        truncation_penalty=200.0,
        rejected_penalty=200.0,
        noop_penalty=100.0,
        belief_mode="pomdp",
        inspection_prior_sampler=None,
        world_sampler=None,
        eol_op=None,
        expose_eol_belief=False,
        observer=observer,
        initial_mounted_tool="H5",
    )
    return env


def make_env_from_imported_xml_auto(
    xml_path: str | Path,
    observer,
    *,
    base_parts: tuple[str, ...] = (),
    target_parts: tuple[str, ...] | None = None,
    max_steps: int = 200,
    enable_screwing_if_available: bool = True,
) -> DisassemblyPOMDPEnv:
    """
    Create an external-case POMDP env without a manually specified CASE_CONFIG.

    Intended for large-scale stress testing:
    - no manually fixed base part by default
    - the case's target parts; without them, every part that is not a base part
    - screwing is enabled only if screw connections exist
    """
    if observer is None:
        raise ValueError("POMDP external-case env requires a real observer.")

    base_asm = load_imported_assembly(xml_path)
    settings = make_cell_settings()

    all_part_names = sorted([p.name for p in base_asm.parts.values()])
    base_part_set = set(base_parts)
    target_part_names = (
        [n for n in all_part_names if n not in base_part_set]
        if target_parts is None
        else list(target_parts)
    )

    planner = make_planner(copy.deepcopy(base_asm))
    plan_screwing_adapter = make_plan_screwing_adapter(planner)
    plan_gripping_and_removing_adapter = make_plan_gripping_and_removing_adapter(
        planner
    )

    capabilities = []

    enable_screwing = enable_screwing_if_available and has_screw_connections(base_asm)
    if enable_screwing:
        screwing = ScrewingCapability(
            available_screwdrivers=["H5", "M4"],
            cell_settings=settings,
            operation_cls=ScrewingOperation,
            plan_func=plan_screwing_adapter,
        )
        capabilities.append(screwing)

    gripping_and_removing = GrippingAndRemovingCapability.from_assembly(
        base_asm,
        available_grippers=["H5"],
        cell_settings=settings,
        operation_cls=GrippingAndRemovingOperation,
        plan_func=plan_gripping_and_removing_adapter,
        base_parts=base_parts,
    )
    capabilities.append(gripping_and_removing)

    env = DisassemblyPOMDPEnv(
        assembly=base_asm,
        capabilities=capabilities,
        cell_settings=settings,
        target_part_names=target_part_names,
        max_steps=max_steps,
        truncation_penalty=200.0,
        rejected_penalty=200.0,
        noop_penalty=100.0,
        belief_mode="pomdp",
        inspection_prior_sampler=None,
        world_sampler=None,
        eol_op=None,
        expose_eol_belief=False,
        observer=observer,
        initial_mounted_tool="H5",
    )
    return env


def extract_step_debug(info: dict) -> dict:
    """
    Extract compact debug information from env.step info.

    This version is aligned with DisassemblyPOMDPEnv.step().
    It keeps:
    - predecode / mask diagnostics
    - execution-gate diagnostics
    - stochastic transition diagnostics
    - observer / visibility diagnostics
    - compact operation data
    """
    debug_keys = [
        # General outcome
        "success",
        "noop",
        "illegal_action",
        "illegal_reason",
        "mask_reason",
        "masked",
        "why",
        "n0",
        "n1",
        "required",
        "mounted",
        "requested",
        # Rejection / validation
        "failure_stage",
        "noop_reason",
        "attempted",
        "target_part",
        "truth_present_pre",
        "target_ref",
        # Exec gate / feasibility
        "exec_gate_ok",
        "exec_fail_reason",
        "exec_fail_details",
        "exec_stage",
        "exec_gate_warning",
        # Stochastic / EOL transition
        "sampled_success",
        "p_exec",
        "p_succ",
        "eol_event",
        "eol_event_error",
        # Timing / reward decomposition
        "time_cost",
        "penalty",
        "truncation_penalty",
        "prefix_time",
        "toolchange_time",
        "start_to_action_start_time",
        "base_op_time",
        # Tool / robot state
        "tool_before",
        "tool_after",
        "tool_required",
        "tool_mounted_at_mask",
        "predecode_toolchange_hint",
        "mask_was_toolchange_override",
        "robot_at_dropoff_after",
        # Observer / measurement
        "detected_count_pre",
        "detected_count_post",
        "measurement_debug_pre",
        "measurement_debug_post",
        # Task
        "task_success",
        # Raw action info
        "capability_index",
        "cap_type",
        "raw_action",
    ]

    debug = {}

    for k in debug_keys:
        if k in info:
            debug[k] = to_jsonable(info.get(k))

    semantic_action = info.get("semantic_action", None)

    if semantic_action is not None:
        debug["semantic_action_type"] = type(semantic_action).__name__

        # Common semantic action attributes
        for attr in [
            "part_name",
            "screw_part_name",
            "base_part_name",
            "tool_name",
        ]:
            if hasattr(semantic_action, attr):
                debug[f"semantic_{attr}"] = to_jsonable(getattr(semantic_action, attr))

        op = getattr(semantic_action, "operation", None)

        if op is not None:
            debug["operation_type"] = type(op).__name__
            data = getattr(op, "data", None)
            if isinstance(data, dict):
                debug["operation_data"] = summarize_operation_data(data)

    # Derived compact failure label for easier later statistics
    if bool(info.get("illegal_action", False)):
        debug["failure_label"] = (
            f"illegal:{info.get('illegal_reason') or info.get('mask_reason') or 'unknown'}"
        )
    elif info.get("failure_stage") == "exec_gate":
        debug["failure_label"] = (
            f"exec_gate:{info.get('exec_fail_reason') or info.get('noop_reason') or 'unknown'}"
        )
    elif info.get("failure_stage") == "stochastic":
        debug["failure_label"] = "stochastic:sample_miss"
    elif info.get("failure_stage") == "semantic":
        debug["failure_label"] = "semantic:sampled_but_no_effect"
    elif info.get("success") is True:
        debug["failure_label"] = "success"
    else:
        debug["failure_label"] = str(
            info.get("noop_reason") or info.get("failure_stage") or "unknown"
        )

    # Keep all keys for post-hoc inspection
    debug["info_keys"] = sorted([str(k) for k in info.keys()])

    return debug


def get_detected_parts_snapshot(env) -> set[str]:
    """
    Safely read the current observer-detected part set from the environment.
    """
    detected = getattr(env, "_detected_parts", None)
    if detected is None:
        detected = getattr(env, "_last_detected_parts", None)

    if detected is None:
        return set()

    try:
        return set(str(x) for x in detected)
    except Exception:
        return set()


def compute_visibility_delta(
    *,
    detected_before: set[str],
    detected_after: set[str],
    removed_by_step: list[str],
) -> dict:
    """
    Detect occlusion-induced visibility changes.

    A partial observability event is recorded when removing one or more
    cover parts makes previously hidden parts observable.
    """
    before = set(detected_before)
    after = set(detected_after)
    removed = set(removed_by_step)

    newly_visible = sorted(after - before)

    # Parts removed in this step should not count as newly visible.
    newly_visible = [p for p in newly_visible if p not in removed]

    event = {
        "partial_observability_event": bool(removed and newly_visible),
        "cover_parts": sorted(removed),
        "newly_visible_parts": newly_visible,
        "detected_before": sorted(before),
        "detected_after": sorted(after),
        "num_detected_before": len(before),
        "num_detected_after": len(after),
    }

    return event


def render_assembly(asm, title="assembly"):
    geometries = []
    screw_names = get_screw_part_names(asm)

    palette_idx = 0
    for part in asm.parts.values():
        mesh = getattr(part, "mesh", None)
        if mesh is None:
            continue

        m = o3d.geometry.TriangleMesh(mesh)
        m.compute_vertex_normals()

        if part.name in screw_names:
            m.paint_uniform_color(WBK_GREY)
        else:
            m.paint_uniform_color(ASSEMBLY_PALETTE[palette_idx % len(ASSEMBLY_PALETTE)])
            palette_idx += 1

        geometries.append(m)

    if not geometries:
        print(f"[{title}] No geometries to render")
        return

    print(f"[{title}] Rendering {len(geometries)} parts...")
    o3d.visualization.draw_geometries(geometries, window_name=title)


def compute_scene_bounds_from_geoms(geoms, *, include_lines: bool = False):
    mins = []
    maxs = []

    for g in geoms:
        if isinstance(g, o3d.geometry.TriangleMesh):
            aabb = g.get_axis_aligned_bounding_box()
            mins.append(np.asarray(aabb.min_bound, dtype=float))
            maxs.append(np.asarray(aabb.max_bound, dtype=float))
        elif include_lines and isinstance(g, o3d.geometry.LineSet):
            aabb = g.get_axis_aligned_bounding_box()
            mins.append(np.asarray(aabb.min_bound, dtype=float))
            maxs.append(np.asarray(aabb.max_bound, dtype=float))

    if not mins:
        raise ValueError("No geometry bounds available")

    min_bound = np.min(np.vstack(mins), axis=0)
    max_bound = np.max(np.vstack(maxs), axis=0)
    center = 0.5 * (min_bound + max_bound)
    diag = float(np.linalg.norm(max_bound - min_bound))

    # Do not force tiny objects to behave like size=1.0
    diag = max(diag, 1e-3)
    return min_bound, max_bound, center, diag


def export_o3d_animation_gif(
    frames: list[list[o3d.geometry.Geometry]],
    out_gif: str | Path,
    width: int = 640,
    height: int = 480,
    fps: int = 10,
    repeat_last: int = 8,
) -> Path:
    out_gif = Path(out_gif).resolve()
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to export")

    _, _, center, diag = compute_scene_bounds_from_geoms(frames[0], include_lines=False)
    print(f"[DEBUG] center={center}, diag={diag}")
    print(f"[DEBUG] input frame count={len(frames)}")

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light(
        direction=[-0.5, -0.5, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=50000,
    )
    scene.scene.enable_indirect_light(False)

    mesh_mat = o3d.visualization.rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 6.0

    def add_frame_geometries(frame_idx: int, geoms):
        scene.clear_geometry()
        for j, g in enumerate(geoms):
            name = f"frame{frame_idx}_geom{j}"
            if isinstance(g, o3d.geometry.TriangleMesh):
                scene.add_geometry(name, g, mesh_mat)
            elif isinstance(g, o3d.geometry.LineSet):
                scene.add_geometry(name, g, line_mat)

    add_frame_geometries(0, frames[0])

    eye = center + diag * np.array([1.2, 1.2, 0.8])
    up = np.array([0.0, 0.0, 1.0])
    renderer.setup_camera(55.0, center, eye, up)

    _ = renderer.render_to_image()

    images = []
    for frame_idx, geoms in enumerate(frames):
        add_frame_geometries(frame_idx, geoms)
        _ = renderer.render_to_image()
        img = renderer.render_to_image()
        img_np = np.asarray(img)

        # force RGB uint8
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        images.append(img_np)

    for _ in range(repeat_last):
        images.append(images[-1].copy())

    print(f"[DEBUG] final image count written={len(images)}")

    # IMPORTANT: explicit GIF writer
    duration_ms = int(round(1000 / fps))
    with imageio.get_writer(
        out_gif, mode="I", format="GIF", duration=duration_ms / 1000.0, loop=0
    ) as writer:
        for im in images:
            writer.append_data(im)

    scene.clear_geometry()
    del renderer

    return out_gif


def assembly_to_o3d_geoms(asm, part_color_map: dict[str, list[float]] | None = None):
    geometries = []
    screw_names = get_screw_part_names(asm)

    palette_idx = 0
    for part in asm.parts.values():
        mesh = getattr(part, "mesh", None)
        if mesh is None:
            continue

        m = o3d.geometry.TriangleMesh(mesh)
        m.compute_vertex_normals()

        if part_color_map is not None and part.name in part_color_map:
            color = part_color_map[part.name]
        else:
            if part.name in screw_names:
                color = WBK_GREY
            else:
                color = ASSEMBLY_PALETTE[palette_idx % len(ASSEMBLY_PALETTE)]
                palette_idx += 1

        m.paint_uniform_color(color)
        geometries.append(m)

    return geometries


def collect_multi_rollout_frames(
    env,
    removal_order: list[str],
    seed: int = 0,
    num_frames_per_step: int = 20,
    hold_last_frame: int = 5,
    verbose: bool = True,
):
    obs, info = env.reset(seed=seed)
    all_frames: list[list[o3d.geometry.Geometry]] = []

    if verbose:
        print("\n=== COLLECT MULTI ROLLOUT FRAMES ===")

    for t, target in enumerate(removal_order):
        if env.assembly.get_part_by_name(target) is None:
            if verbose:
                print(f"[{t}] skip missing {target}")
            continue

        assembly_before = copy.deepcopy(env.assembly)
        a = find_grip_action_for_target(env, target)
        obs, r, term, trunc, inf = env.step(a)

        if verbose:
            print(f"[{t}] target={target} success={inf.get('success')}")

        assert inf.get("success", False), f"Removal failed for {target}"

        semantic_action = inf.get("semantic_action", None)
        op = getattr(semantic_action, "operation", None)
        if op is None or not hasattr(op, "animation_frames_o3d"):
            continue

        step_frames = op.animation_frames_o3d(
            assembly_before,
            num_frames=num_frames_per_step,
        )
        if not step_frames:
            continue

        all_frames.extend(step_frames)

        post_step_geoms = assembly_to_o3d_geoms(copy.deepcopy(env.assembly))
        for _ in range(hold_last_frame):
            all_frames.append(post_step_geoms)

    return all_frames


def sanity_external_pomdp_env(
    env, seed: int = 0, n_steps: int = 10, verbose: bool = True
) -> bool:
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "reset obs not in observation_space"

    if verbose:
        print("\n=== SANITY external POMDP env ===")
        print("reset obs keys:", obs.keys())
        print("detected after reset:", getattr(env, "_detected_parts", None))
        print("measurement_debug_reset:", info.get("measurement_debug_reset"))

    rng = np.random.default_rng(seed)

    for t in range(n_steps):
        cap_idx = int(rng.integers(0, int(env.action_space.nvec[0])))
        sp = env._cap_action_spaces[cap_idx]
        sub0 = int(rng.integers(0, int(sp.nvec[0])))
        sub1 = int(rng.integers(0, int(sp.nvec[1])))
        a = np.array([cap_idx, sub0, sub1], dtype=np.int64)

        obs2, r, term, trunc, inf = env.step(a)
        assert env.observation_space.contains(obs2), (
            f"step obs not in observation_space at t={t}"
        )

        if verbose:
            print(
                f"[{t}] action={a.tolist()} "
                f"reward={r:.3f} illegal={inf.get('illegal_action')} "
                f"success={inf.get('success')} detected={getattr(env, '_detected_parts', None)}"
            )

        if term or trunc:
            break

    print("✅ PASS: external POMDP env basic sanity")
    return True


def find_screw_action_for_target(env, target_name: str) -> np.ndarray:
    cap_idx = None
    cap = None

    for i, c in enumerate(env.capabilities):
        if isinstance(c, ScrewingCapability):
            cap_idx = i
            cap = c
            break

    if cap is None:
        raise RuntimeError("ScrewingCapability not found")

    if not hasattr(cap, "_global_screw_conn_keys") or not cap._global_screw_conn_keys:
        raise RuntimeError("ScrewingCapability has no built screw connection index")

    tool_idx = 0  # H5 in your current setup

    for conn_idx, key in enumerate(cap._global_screw_conn_keys):
        screw_name, base_name, head = key
        if screw_name == target_name:
            return np.array([cap_idx, tool_idx, int(conn_idx)], dtype=np.int64)

    raise RuntimeError(f"Target {target_name} not found in screwing capability")


def find_grip_action_for_target(env, target_name: str) -> np.ndarray:
    cap_idx = None
    cap = None

    for i, c in enumerate(env.capabilities):
        if isinstance(c, GrippingAndRemovingCapability):
            cap_idx = i
            cap = c
            break

    if cap is None:
        raise RuntimeError("GrippingAndRemovingCapability not found")

    tool_idx = 0  # universal_gripper only

    try:
        part_idx = cap.name_to_part_index[target_name]
    except Exception as e:
        raise RuntimeError(
            f"Target {target_name} not found in gripping_and_removing capability"
        ) from e

    return np.array([cap_idx, tool_idx, int(part_idx)], dtype=np.int64)


def find_action_for_target(env, target_name: str) -> np.ndarray:
    # 1) try screwing first
    for c in env.capabilities:
        if isinstance(c, ScrewingCapability):
            try:
                return find_screw_action_for_target(env, target_name)
            except RuntimeError:
                pass

    # 2) then try gripping
    for c in env.capabilities:
        if isinstance(c, GrippingAndRemovingCapability):
            try:
                return find_grip_action_for_target(env, target_name)
            except RuntimeError:
                pass

    raise RuntimeError(f"No executable action found for target {target_name}")


def split_gallery_overflow(items, primary_capacity: int):
    primary = items[:primary_capacity]
    overflow = items[primary_capacity:]
    return primary, overflow


def split_parts_screw_vs_normal(asm: Assembly):
    screw_names = get_screw_part_names(asm)

    screws = []
    normals = []

    for p in asm.parts.values():
        if p.name in screw_names:
            screws.append(p.name)
        else:
            normals.append(p.name)

    return sorted(screws), sorted(normals)


def compute_fixed_camera_basis(center: np.ndarray, diag: float):
    eye = center + diag * np.array([1.2, 1.2, 0.8], dtype=np.float64)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    view_dir = center - eye
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)

    cam_right = np.cross(view_dir, up_world)
    cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-12)

    cam_up = np.cross(cam_right, view_dir)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    return eye, view_dir, cam_right, cam_up


def make_billboard_gallery_geoms(
    removed_parts: list[tuple[str, o3d.geometry.TriangleMesh]],
    *,
    anchor_center: np.ndarray,
    scene_diag: float,
    side: str,  # "left" | "right" | "top" | "bottom"
    max_rows: int = 6,
    row_spacing_scale: float = 0.24,
    col_spacing_scale: float = 0.26,
    side_offset_scale: float = 1.40,
    plane_forward_scale: float = 0.28,
    target_max_extent_scale: float = 0.18,
):
    geoms: list[o3d.geometry.Geometry] = []
    if not removed_parts:
        return geoms

    anchor_center = np.asarray(anchor_center, dtype=np.float64)
    _, view_dir, cam_right, cam_up = compute_fixed_camera_basis(
        anchor_center, scene_diag
    )

    # Put gallery on a plane parallel to the image plane, slightly toward camera
    plane_center = anchor_center + view_dir * (plane_forward_scale * scene_diag)

    row_spacing = max(scene_diag * row_spacing_scale, 10.0)
    col_spacing = max(scene_diag * col_spacing_scale, 10.0)
    target_max_extent = max(scene_diag * target_max_extent_scale, 8.0)

    n = len(removed_parts)
    n_rows = min(max_rows, max(1, n))

    for idx, (_, mesh) in enumerate(removed_parts):
        row = idx % n_rows
        col = idx // n_rows

        # symmetric layout around center line
        row_center = 0.5 * (n_rows - 1)
        row_offset = (row_center - row) * row_spacing

        m = o3d.geometry.TriangleMesh(mesh)
        m.compute_vertex_normals()

        aabb = m.get_axis_aligned_bounding_box()
        part_center = np.asarray(aabb.get_center(), dtype=np.float64)
        extent = np.asarray(aabb.get_extent(), dtype=np.float64)
        max_extent = float(np.max(extent))

        # shrink only if necessary
        if max_extent > target_max_extent and max_extent > 1e-9:
            scale = target_max_extent / max_extent
            m.scale(scale, center=part_center)
            aabb = m.get_axis_aligned_bounding_box()
            part_center = np.asarray(aabb.get_center(), dtype=np.float64)

        if side == "left":
            target_center = (
                plane_center
                - cam_right * (side_offset_scale * scene_diag + col * col_spacing)
                + cam_up * row_offset
            )
        elif side == "right":
            target_center = (
                plane_center
                + cam_right * (side_offset_scale * scene_diag + col * col_spacing)
                + cam_up * row_offset
            )
        elif side == "top":
            target_center = (
                plane_center
                + cam_up * (side_offset_scale * scene_diag + col * col_spacing)
                + cam_right * row_offset
            )
        elif side == "bottom":
            target_center = (
                plane_center
                - cam_up * (side_offset_scale * scene_diag + col * col_spacing)
                + cam_right * row_offset
            )
        else:
            raise ValueError(f"Unknown gallery side: {side}")

        m.translate(target_center - part_center)
        geoms.append(m)

    return geoms


def build_partitioned_gallery_geoms_for_static_frame(
    *,
    removed_screws,
    removed_parts,
    reference_geoms: list[o3d.geometry.Geometry],
    max_primary_side_items: int = 16,
):
    if not reference_geoms:
        return []

    _, _, center, diag = compute_scene_bounds_from_geoms(
        reference_geoms, include_lines=False
    )

    # overflow split
    screws_left, screws_top = split_gallery_overflow(
        removed_screws, max_primary_side_items
    )
    parts_right, parts_bottom = split_gallery_overflow(
        removed_parts, max_primary_side_items
    )

    geoms: list[o3d.geometry.Geometry] = []

    # screws: LEFT first
    geoms.extend(
        make_billboard_gallery_geoms(
            screws_left,
            anchor_center=center,
            scene_diag=diag,
            side="left",
            max_rows=8,
            row_spacing_scale=0.16,
            col_spacing_scale=0.16,
            side_offset_scale=1.10,
            plane_forward_scale=0.28,
            target_max_extent_scale=0.07,  # screws can stay small
        )
    )

    # screw overflow: TOP
    geoms.extend(
        make_billboard_gallery_geoms(
            screws_top,
            anchor_center=center,
            scene_diag=diag,
            side="top",
            max_rows=8,
            row_spacing_scale=0.14,
            col_spacing_scale=0.16,
            side_offset_scale=1.10,
            plane_forward_scale=0.28,
            target_max_extent_scale=0.07,
        )
    )

    # normal parts: RIGHT first
    geoms.extend(
        make_billboard_gallery_geoms(
            parts_right,
            anchor_center=center,
            scene_diag=diag,
            side="right",
            max_rows=5,
            row_spacing_scale=0.28,
            col_spacing_scale=0.34,
            side_offset_scale=1.10,
            plane_forward_scale=0.28,
            target_max_extent_scale=0.26,  # larger budget for normal parts
        )
    )

    # normal overflow: BOTTOM
    geoms.extend(
        make_billboard_gallery_geoms(
            parts_bottom,
            anchor_center=center,
            scene_diag=diag,
            side="bottom",
            max_rows=5,
            row_spacing_scale=0.24,
            col_spacing_scale=0.30,
            side_offset_scale=1.10,
            plane_forward_scale=0.28,
            target_max_extent_scale=0.22,
        )
    )

    return geoms


def append_partitioned_gallery_to_frames(
    frames: list[list[o3d.geometry.Geometry]],
    *,
    removed_screws,
    removed_parts,
    reference_geoms: list[o3d.geometry.Geometry],
):
    if not frames:
        return frames

    gallery_geoms = build_partitioned_gallery_geoms_for_static_frame(
        removed_screws=removed_screws,
        removed_parts=removed_parts,
        reference_geoms=reference_geoms,
    )

    out = []
    for frame in frames:
        merged = [copy.deepcopy(g) for g in frame]
        merged.extend(copy.deepcopy(g) for g in gallery_geoms)
        out.append(merged)

    return out


def append_final_hold_frames(
    frames: list[list[o3d.geometry.Geometry]],
    final_hold_frames: int,
) -> list[list[o3d.geometry.Geometry]]:
    if not frames or final_hold_frames <= 0:
        return frames

    final_frame = [copy.deepcopy(g) for g in frames[-1]]
    for _ in range(final_hold_frames):
        frames.append([copy.deepcopy(g) for g in final_frame])

    return frames


def brutal_force_rollout_with_gif(
    env,
    seed: int = 0,
    max_steps: int | None = None,
    num_frames_per_step: int = 25,
    hold_last_frame: int = 8,
    final_hold_frames: int = 60,
    verbose: bool = True,
):
    """
    Run a simple target-selecting brutal-force agent through the full POMDP pipeline.
    Collect animation frames only for successful removals.

    Extra visualization:
    - removed screw parts are parked on the LEFT side
    - removed non-screw parts are parked on the RIGHT side
    - current removed part is added to the gallery ONLY AFTER the action finishes

    Returns (success, frames).
    """
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    if max_steps is None:
        max_steps = env.max_steps

    all_frames: list[list[o3d.geometry.Geometry]] = []

    # fixed whole-rollout color map
    base_color_map = build_part_color_map(copy.deepcopy(env.base_assembly))

    # partitioned removed-part galleries
    removed_screws: list[tuple[str, o3d.geometry.TriangleMesh]] = []
    removed_parts: list[tuple[str, o3d.geometry.TriangleMesh]] = []

    # initial static frame
    initial_geoms = assembly_to_o3d_geoms(
        copy.deepcopy(env.assembly),
        part_color_map=base_color_map,
    )
    initial_gallery = build_partitioned_gallery_geoms_for_static_frame(
        removed_screws=removed_screws,
        removed_parts=removed_parts,
        reference_geoms=initial_geoms,
    )
    initial_combined = [copy.deepcopy(g) for g in initial_geoms]
    initial_combined.extend(copy.deepcopy(g) for g in initial_gallery)

    for _ in range(hold_last_frame):
        all_frames.append([copy.deepcopy(g) for g in initial_combined])

    if verbose:
        print("\n=== BRUTAL FORCE ROLLOUT WITH GIF ===")
        print("targets:", env.target_part_names)
        print("initial parts:", sorted([p.name for p in env.assembly.parts.values()]))

    for t in range(max_steps):
        remaining_targets = [
            name
            for name in env.target_part_names
            if env.assembly.get_part_by_name(name) is not None
        ]

        if not remaining_targets:
            if verbose:
                print("✅ PASS: all targets removed")

            all_frames = append_final_hold_frames(all_frames, final_hold_frames)
            return True, all_frames

        target = rng.choice(remaining_targets)
        assembly_before = copy.deepcopy(env.assembly)

        a = find_action_for_target(env, target)
        obs, r, term, trunc, inf = env.step(a)

        if verbose:
            print(
                f"[{t}] target={target} action={a.tolist()} "
                f"reward={r:.3f} success={inf.get('success')} "
                f"illegal={inf.get('illegal_action')} term={term} trunc={trunc}"
            )

        if inf.get("success", False):
            semantic_action = inf.get("semantic_action", None)
            op = getattr(semantic_action, "operation", None)

            # ---------------------------------------
            # 1) first render THIS step with OLD gallery only
            # ---------------------------------------
            if op is not None and hasattr(op, "animation_frames_o3d"):
                step_frames = op.animation_frames_o3d(
                    assembly_before,
                    num_frames=num_frames_per_step,
                )
                if step_frames:
                    step_frames = append_partitioned_gallery_to_frames(
                        step_frames,
                        removed_screws=removed_screws,
                        removed_parts=removed_parts,
                        reference_geoms=assembly_to_o3d_geoms(
                            copy.deepcopy(assembly_before),
                            part_color_map=base_color_map,
                        ),
                    )
                    all_frames.extend(step_frames)

            # ---------------------------------------
            # 2) determine which part was actually removed
            # ---------------------------------------
            removed_part_name = None

            if op is not None and isinstance(getattr(op, "data", None), dict):
                removed_part_name = op.data.get("part_name", None)

                if removed_part_name is None:
                    conn = op.data.get("connection", None)
                    if conn is not None:
                        removed_part_name = getattr(conn, "part_name1", None)

            # ---------------------------------------
            # 3) AFTER action ends, add removed part to gallery
            #    keep original rollout color
            # ---------------------------------------
            if isinstance(removed_part_name, str):
                removed_part_obj = assembly_before.get_part_by_name(removed_part_name)
                if (
                    removed_part_obj is not None
                    and getattr(removed_part_obj, "mesh", None) is not None
                ):
                    color = base_color_map.get(removed_part_name, WBK_GREY)
                    mesh_copy = paint_mesh_copy(removed_part_obj.mesh, color)

                    is_screw = False
                    lname = removed_part_name.lower()
                    if "screw" in lname or "schraube" in lname:
                        is_screw = True
                    else:
                        for c in assembly_before.connections:
                            if getattr(c, "connection_type", None) == "screw":
                                if getattr(c, "part_name1", None) == removed_part_name:
                                    is_screw = True
                                    break

                    if is_screw:
                        removed_screws.append((removed_part_name, mesh_copy))
                    else:
                        removed_parts.append((removed_part_name, mesh_copy))

            # ---------------------------------------
            # 4) hold real post-step static assembly + UPDATED gallery
            # ---------------------------------------
            post_step_geoms = assembly_to_o3d_geoms(
                copy.deepcopy(env.assembly),
                part_color_map=base_color_map,
            )
            post_gallery = build_partitioned_gallery_geoms_for_static_frame(
                removed_screws=removed_screws,
                removed_parts=removed_parts,
                reference_geoms=post_step_geoms,
            )

            combined_post = [copy.deepcopy(g) for g in post_step_geoms]
            combined_post.extend(copy.deepcopy(g) for g in post_gallery)

            for _ in range(hold_last_frame):
                all_frames.append([copy.deepcopy(g) for g in combined_post])

        if term:
            if verbose:
                print("✅ PASS: terminated successfully")

            all_frames = append_final_hold_frames(all_frames, final_hold_frames)
            return True, all_frames

        if trunc:
            if verbose:
                remaining_targets = [
                    name
                    for name in env.target_part_names
                    if env.assembly.get_part_by_name(name) is not None
                ]
                remaining_parts = sorted([p.name for p in env.assembly.parts.values()])
                screw_parts, normal_parts = split_parts_screw_vs_normal(env.assembly)

                print("⚠️ truncated")
                print("[DEBUG] rollout success: False")
                print("[DEBUG] reason: truncation")
                print("[DEBUG] remaining target parts:")
                print("   ", remaining_targets)
                print("[DEBUG] remaining assembly parts:")
                print("   ", remaining_parts)
                print("[DEBUG] remaining screws:")
                print("   ", screw_parts)
                print("[DEBUG] remaining non-screw parts:")
                print("   ", normal_parts)
                print("[DEBUG] detected parts:")
                print("   ", getattr(env, "_detected_parts", None))
                print("[DEBUG] current mounted tool:")
                print("   ", getattr(env, "current_mounted_tool", None))

            all_frames = append_final_hold_frames(all_frames, final_hold_frames)
            return False, all_frames

    if verbose:
        remaining_targets = [
            name
            for name in env.target_part_names
            if env.assembly.get_part_by_name(name) is not None
        ]
        remaining_parts = sorted([p.name for p in env.assembly.parts.values()])
        screw_parts, normal_parts = split_parts_screw_vs_normal(env.assembly)

        print("⚠️ max_steps reached without termination")
        print("[DEBUG] rollout success: False")
        print("[DEBUG] reason: max_steps_reached")
        print("[DEBUG] remaining target parts:")
        print("   ", remaining_targets)
        print("[DEBUG] remaining assembly parts:")
        print("   ", remaining_parts)
        print("[DEBUG] remaining screws:")
        print("   ", screw_parts)
        print("[DEBUG] remaining non-screw parts:")
        print("   ", normal_parts)
        print("[DEBUG] detected parts:")
        print("   ", getattr(env, "_detected_parts", None))
        print("[DEBUG] current mounted tool:")
        print("   ", getattr(env, "current_mounted_tool", None))

    all_frames = append_final_hold_frames(all_frames, final_hold_frames)
    return False, all_frames


def brutal_force_rollout_headless(
    env: DisassemblyPOMDPEnv,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Headless brutal-force rollout for large-scale evaluation.

    The policy repeatedly tries to remove remaining target parts.
    In each pass, all remaining targets are shuffled and attempted once.
    If a full pass removes no part, the rollout stops as no_progress.
    """
    obs, reset_info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    initial_detected_parts = sorted(get_detected_parts_snapshot(env))

    if max_steps is None:
        max_steps = env.max_steps

    initial_parts = sorted([p.name for p in env.assembly.parts.values()])
    initial_connections = serialize_connections(env.assembly)

    initial_hidden_parts = sorted(set(initial_parts) - set(initial_detected_parts))
    partial_observability_events: list[dict] = []

    initial_num_parts = len(initial_parts)
    initial_num_connections = len(initial_connections)

    attempts: list[dict] = []
    removed_parts: list[str] = []

    step_idx = 0
    stop_reason = None
    terminated = False
    truncated = False

    if verbose:
        print("\n=== HEADLESS BRUTAL FORCE ROLLOUT ===")
        print("assembly:", env.base_assembly.name)
        print("initial parts:", initial_num_parts)
        print("initial connections:", initial_num_connections)
        print("targets:", env.target_part_names)

    while step_idx < max_steps:
        remaining_targets = [
            name
            for name in env.target_part_names
            if env.assembly.get_part_by_name(name) is not None
        ]

        if not remaining_targets:
            stop_reason = "all_targets_removed"
            terminated = True
            break

        rng.shuffle(remaining_targets)
        removed_this_pass = 0
        attempted_this_pass = 0

        for target in remaining_targets:
            if step_idx >= max_steps:
                stop_reason = "max_steps_reached"
                break

            if env.assembly.get_part_by_name(target) is None:
                continue

            before_parts = set(p.name for p in env.assembly.parts.values())
            before_connections = serialize_connections(env.assembly)
            detected_before = get_detected_parts_snapshot(env)

            attempt_record = {
                "step": step_idx,
                "target": target,
                "action": None,
                "step_executed": False,
                "success": False,
                "reward": None,
                "terminated": False,
                "truncated": False,
                "removed_by_step": [],
                "before_num_parts": len(before_parts),
                "before_num_connections": len(before_connections),
                "after_num_parts": None,
                "after_num_connections": None,
                "debug": {},
            }

            try:
                action = find_action_for_target(env, target)
                attempt_record["action"] = to_jsonable(action)
            except Exception as exc:
                attempt_record["debug"] = {
                    "stage": "action_construction",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                attempts.append(attempt_record)

                if verbose:
                    print(
                        f"[{step_idx}] target={target} action construction failed: {exc}"
                    )

                # This is not an env.step, so do not increment step_idx
                continue

            try:
                current_step = step_idx

                obs, reward, term, trunc, info = env.step(action)
                attempted_this_pass += 1
                step_idx += 1

                after_parts = set(p.name for p in env.assembly.parts.values())
                after_connections = serialize_connections(env.assembly)
                detected_after = get_detected_parts_snapshot(env)

                removed_now = sorted(before_parts - after_parts)

                visibility_delta = compute_visibility_delta(
                    detected_before=detected_before,
                    detected_after=detected_after,
                    removed_by_step=removed_now,
                )

                if visibility_delta["partial_observability_event"]:
                    po_event = {
                        "step": current_step,
                        "target": target,
                        "removed_by_step": removed_now,
                        "cover_parts": visibility_delta["cover_parts"],
                        "occluded_parts_revealed": visibility_delta[
                            "newly_visible_parts"
                        ],
                        "detected_before": visibility_delta["detected_before"],
                        "detected_after": visibility_delta["detected_after"],
                    }
                    partial_observability_events.append(po_event)

                if removed_now:
                    removed_this_pass += len(removed_now)
                    for p in removed_now:
                        if p not in removed_parts:
                            removed_parts.append(p)

                attempt_record.update(
                    {
                        "step": current_step,
                        "step_executed": True,
                        "success": bool(info.get("success", False)),
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "removed_by_step": removed_now,
                        "before_num_parts": len(before_parts),
                        "before_num_connections": len(before_connections),
                        "after_num_parts": len(after_parts),
                        "after_num_connections": len(after_connections),
                        # Visibility / partial observability trace
                        "detected_before": sorted(detected_before),
                        "detected_after": sorted(detected_after),
                        "newly_visible_parts": visibility_delta["newly_visible_parts"],
                        "partial_observability_event": visibility_delta[
                            "partial_observability_event"
                        ],
                        "debug": extract_step_debug(info),
                    }
                )

                attempts.append(attempt_record)

                if verbose:
                    print(
                        f"[{current_step}] target={target} "
                        f"success={attempt_record['success']} "
                        f"removed={removed_now} "
                        f"remaining_parts={len(after_parts)}"
                    )

                if term:
                    stop_reason = "env_terminated"
                    terminated = True
                    break

                if trunc:
                    stop_reason = "env_truncated"
                    truncated = True
                    break

            except Exception as exc:
                step_idx += 1

                attempt_record.update(
                    {
                        "step_executed": True,
                        "success": False,
                        "debug": {
                            "stage": "env_step_exception",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    }
                )
                attempts.append(attempt_record)

                if verbose:
                    print(f"[{step_idx - 1}] target={target} env.step exception: {exc}")

        if stop_reason is not None:
            break

        if attempted_this_pass == 0:
            stop_reason = "no_executable_actions"
            break

        if removed_this_pass == 0:
            stop_reason = "no_progress"
            break

    if stop_reason is None:
        stop_reason = "max_steps_reached"

    final_parts = sorted([p.name for p in env.assembly.parts.values()])
    final_connections = serialize_connections(env.assembly)

    removed_parts = sorted(set(initial_parts) - set(final_parts))
    removed_num_parts = len(removed_parts)

    completion = removed_num_parts / initial_num_parts if initial_num_parts > 0 else 1.0

    success = len(final_parts) == 0 or all(
        env.assembly.get_part_by_name(name) is None for name in env.target_part_names
    )

    report = {
        "assembly_name": env.base_assembly.name,
        "success": bool(success),
        "stop_reason": stop_reason,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_num_parts": initial_num_parts,
        "initial_num_connections": initial_num_connections,
        "initial_parts": initial_parts,
        "initial_connections": initial_connections,
        "initial_detected_parts": initial_detected_parts,
        "initial_hidden_parts": initial_hidden_parts,
        "initial_num_detected_parts": len(initial_detected_parts),
        "initial_num_hidden_parts": len(initial_hidden_parts),
        "partial_observability_observed": bool(len(partial_observability_events) > 0),
        "num_partial_observability_events": len(partial_observability_events),
        "partial_observability_events": partial_observability_events,
        "removed_num_parts": removed_num_parts,
        "removed_parts": removed_parts,
        "completion": float(completion),
        "remaining_num_parts": len(final_parts),
        "remaining_num_connections": len(final_connections),
        "remaining_parts": final_parts,
        "remaining_connections": final_connections,
        "num_attempts": len(attempts),
        "num_env_steps": step_idx,
        "attempts": attempts,
        "reset_info": to_jsonable(reset_info),
    }

    return report


def infer_external_group(case_name: str) -> str:
    """Infer dataset group from converted case folder / assembly name."""
    if case_name.startswith("joint_assembly_rotation_"):
        return "joint_assembly_rotation"
    if case_name.startswith("joint_assembly_"):
        return "joint_assembly"
    if case_name.startswith("multi_assembly_"):
        return "multi_assembly"
    return "unknown"


def discover_external_case_xmls(
    converted_root: Path,
    *,
    group: str | None = None,
) -> list[Path]:
    """
    Discover converted external case XML files.

    converted_root example:
        <repo>/../external/test_cases_external
    """
    converted_root = Path(converted_root).resolve()

    xml_paths = []
    for case_dir in sorted(converted_root.iterdir()):
        if not case_dir.is_dir():
            continue

        inferred = infer_external_group(case_dir.name)
        if group is not None and inferred != group:
            continue

        candidate = case_dir / "xml" / f"{case_dir.name}.xml"
        if candidate.exists():
            xml_paths.append(candidate)

    return sorted(xml_paths)


def write_batch_results(
    reports: list[dict],
    *,
    results_dir: Path,
    group: str,
    case_range_label: str | None = None,
    output_group: str | None = None,
) -> tuple[Path, Path]:
    results_dir = Path(results_dir).resolve()
    folder_name = output_group if output_group is not None else group
    group_dir = results_dir / folder_name
    group_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_cases = len(reports)
    label = case_range_label if case_range_label is not None else str(num_cases)

    prefix = f"test_{group}_{label}_{timestamp}"

    csv_path = group_dir / f"{prefix}.csv"
    jsonl_path = group_dir / f"{prefix}.jsonl"

    fieldnames = [
        "assembly_name",
        "status",
        "success",
        "stop_reason",
        "initial_num_parts",
        "initial_num_connections",
        "initial_num_detected_parts",
        "initial_num_hidden_parts",
        "partial_observability_observed",
        "num_partial_observability_events",
        "removed_num_parts",
        "completion",
        "remaining_num_parts",
        "remaining_num_connections",
        "num_attempts",
        "num_env_steps",
        "xml_path",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in reports:
            writer.writerow(
                {
                    "assembly_name": r.get("assembly_name"),
                    "status": r.get("status", "ok"),
                    "success": r.get("success"),
                    "stop_reason": r.get("stop_reason"),
                    "initial_num_parts": r.get("initial_num_parts"),
                    "initial_num_connections": r.get("initial_num_connections"),
                    "initial_num_detected_parts": r.get("initial_num_detected_parts"),
                    "initial_num_hidden_parts": r.get("initial_num_hidden_parts"),
                    "partial_observability_observed": r.get(
                        "partial_observability_observed"
                    ),
                    "num_partial_observability_events": r.get(
                        "num_partial_observability_events"
                    ),
                    "removed_num_parts": r.get("removed_num_parts"),
                    "completion": r.get("completion"),
                    "remaining_num_parts": r.get("remaining_num_parts"),
                    "remaining_num_connections": r.get("remaining_num_connections"),
                    "num_attempts": r.get("num_attempts"),
                    "num_env_steps": r.get("num_env_steps"),
                    "xml_path": r.get("xml_path"),
                    "error": r.get("error", ""),
                }
            )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def run_external_batch_brutal_force(
    *,
    converted_root: Path,
    results_root: Path,
    group: str,
    seed: int = 0,
    max_steps: int = 200,
    observer_debug: bool = False,
    limit: int | None = None,
    batch_start: int = 1,
    batch_size: int | None = None,
    verbose: bool = True,
    output_group: str | None = None,
) -> list[dict]:
    xml_paths_all = discover_external_case_xmls(converted_root, group=group)

    if batch_start < 1:
        raise ValueError("batch_start must be >= 1")

    start_idx = batch_start - 1

    if batch_size is not None:
        end_idx = start_idx + batch_size
        xml_paths = xml_paths_all[start_idx:end_idx]
        case_range_label = f"{batch_start}-{batch_start + len(xml_paths) - 1}"
    else:
        xml_paths = xml_paths_all[start_idx:]
        if limit is not None:
            xml_paths = xml_paths[:limit]
        case_range_label = f"{batch_start}-{batch_start + len(xml_paths) - 1}"

    if not xml_paths:
        raise ValueError(
            f"No XML cases selected for group={group}, batch_start={batch_start}, batch_size={batch_size}. "
            f"Available cases: {len(xml_paths_all)}"
        )

    reports: list[dict] = []

    if verbose:
        print("=" * 80)
        print(f"[BATCH] group={group}")
        print(f"[BATCH] available cases={len(xml_paths_all)}")
        print(f"[BATCH] selected cases={len(xml_paths)}")
        print(f"[BATCH] selected range={case_range_label}")
        print(f"[BATCH] converted_root={converted_root}")
        print(f"[BATCH] results_root={results_root}")

    for idx, xml_path in enumerate(xml_paths, start=1):
        assembly_name = xml_path.parent.parent.name

        if verbose:
            print("=" * 80)
            print(f"[BATCH] [{idx}/{len(xml_paths)}] {assembly_name}")

        try:
            observer = make_external_case_observer(debug=observer_debug)
            env = make_env_from_imported_xml_auto(
                xml_path,
                observer=observer,
                base_parts=(),
                max_steps=max_steps,
                enable_screwing_if_available=True,
            )

            report = brutal_force_rollout_headless(
                env,
                seed=seed,
                max_steps=max_steps,
                verbose=False,
            )
            report["status"] = "ok"
            report["xml_path"] = str(xml_path)
            report["group"] = group

            if verbose:
                print(
                    f"[RESULT] success={report['success']} "
                    f"completion={report['completion']:.3f} "
                    f"removed={report['removed_num_parts']}/{report['initial_num_parts']} "
                    f"stop={report['stop_reason']}"
                )

        except Exception as exc:
            report = {
                "assembly_name": assembly_name,
                "group": group,
                "xml_path": str(xml_path),
                "status": "failed_setup_or_runtime",
                "success": False,
                "stop_reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }

            if verbose:
                print(f"[ERROR] {assembly_name}: {report['error']}")

        reports.append(report)

    csv_path, jsonl_path = write_batch_results(
        reports,
        results_dir=results_root,
        group=group,
        case_range_label=case_range_label,
        output_group=output_group,
    )

    if verbose:
        print("=" * 80)
        print(f"[BATCH] selected range={case_range_label}")
        print(f"[BATCH] available cases={len(xml_paths_all)}")
        print(f"[BATCH] selected cases={len(xml_paths)}")
        print(f"[SAVED CSV]   {csv_path}")
        print(f"[SAVED JSONL] {jsonl_path}")

    return reports


def run_external_batch_brutal_force_in_chunks(
    *,
    converted_root: Path,
    results_root: Path,
    group: str,
    seed: int = 0,
    max_steps: int = 200,
    observer_debug: bool = False,
    chunk_size: int = 100,
    start: int = 1,
    stop: int | None = None,
    output_group: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Run external cases in chunks and save one CSV/JSONL per chunk.

    Example:
        chunk_size=100
        start=1
        stop=1000

    Produces:
        1-100
        101-200
        ...
        901-1000
    """
    xml_paths_all = discover_external_case_xmls(converted_root, group=group)
    total_available = len(xml_paths_all)

    if total_available == 0:
        raise ValueError(f"No XML cases found for group={group} under {converted_root}")

    if start < 1:
        raise ValueError("start must be >= 1")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if stop is None:
        stop = total_available

    stop = min(stop, total_available)

    if start > stop:
        raise ValueError(
            f"Invalid range: start={start}, stop={stop}, available={total_available}"
        )

    if verbose:
        print("=" * 80)
        print(f"[CHUNKED BATCH] group={group}")
        print(f"[CHUNKED BATCH] available cases={total_available}")
        print(f"[CHUNKED BATCH] selected global range={start}-{stop}")
        print(f"[CHUNKED BATCH] chunk_size={chunk_size}")
        print("=" * 80)

    batch_start = start

    while batch_start <= stop:
        remaining = stop - batch_start + 1
        current_batch_size = min(chunk_size, remaining)

        if verbose:
            print("\n" + "#" * 80)
            print(
                f"[CHUNKED BATCH] Running cases "
                f"{batch_start}-{batch_start + current_batch_size - 1}"
            )
            print("#" * 80)

        run_external_batch_brutal_force(
            converted_root=converted_root,
            results_root=results_root,
            group=group,
            seed=seed,
            max_steps=max_steps,
            observer_debug=observer_debug,
            batch_start=batch_start,
            batch_size=current_batch_size,
            output_group=output_group,
            verbose=verbose,
        )

        batch_start += current_batch_size

    if verbose:
        print("=" * 80)
        print("[CHUNKED BATCH] Done.")
        print("=" * 80)


# if __name__ == "__main__":
#    """
#    xml_path = ROOT / "tests" / "external_cases" / "joint_assembly_00201" / "xml" / "joint_assembly_00201.xml"
#    observer = make_external_case_observer(debug=True)
#    env = make_env_from_imported_xml(xml_path, observer=observer)
#
#    sanity_external_pomdp_env(env, seed=0, n_steps=10, verbose=True)
#
#    sanity_target_removal(
#        env,
#        seed=0,
#        verbose=True,
#        render_before=False,
#        render_after=False,
#        animate=False,
#        save_gif=True,
#        gif_dir=ROOT / "tests" / "external_case_gifs",
#    )
#
#    print("current mounted tool:", env.current_mounted_tool)
#    print("tool names:", getattr(env.encoder._vocab, "tool_names", None))
#    """
#
#    """
#    xml_path = (
#        ROOT
#        / "tests"
#        / "external_cases"
#        / "multi_assembly_00643"
#        / "xml"
#        / "multi_assembly_00643.xml"
#    )
#
#    CASE_CONFIG = {
#        "multi_assembly_00643": {
#            "base_parts": ("part_0",),
#            "enable_screwing": False,
#        },
#    }
#    """
#
#    """
#    xml_path = (
#        ROOT
#        / "tests"
#        / "internal_cases"
#        / "download"
#        / "BG2"
#        / "BG2_new.xml"
#    )
#
#    CASE_CONFIG = {
#        "BG2": {
#            "base_parts": ("00_st_7",),
#            "enable_screwing": True,
#        },
#    }
#    """
#
#    xml_path = (
#        ROOT
#        / "tests"
#        / "internal_cases"
#        / "Modell 2-Plannungssoftware"
#        / "Stl"
#        / "Assembly 1"
#        / "Assembly_1.xml"
#    )
#
#    CASE_CONFIG = {
#        "Assembly_1": {
#            "base_parts": ("BG_Gehauese_p",),
#            "enable_screwing": True,
#        },
#    }
#
#    observer = make_external_case_observer(debug=True)
#    env = make_env_from_imported_xml(xml_path, observer=observer)
#
#    print("assembly name:", env.base_assembly.name)
#    print("target parts:", env.target_part_names)
#    print("action space:", env.action_space)
#
#    # keep only basic env sanity
#    sanity_external_pomdp_env(env, seed=0, n_steps=10, verbose=True)
#
#    # run full pipeline with brutal-force agent
#    success, frames = brutal_force_rollout_with_gif(
#        env,
#        seed=0,
#        max_steps=env.max_steps,
#        num_frames_per_step=25,
#        hold_last_frame=8,
#        verbose=True,
#    )
#
#    print("current mounted tool:", env.current_mounted_tool)
#    print("tool names:", getattr(env.encoder._vocab, "tool_names", None))
#    print("[DEBUG] total collected frames:", len(frames))
#    print("[DEBUG] rollout success:", success)
#
#    if frames:
#        gif_path = ROOT / "tests" / "external_case_gifs" / f"{env.base_assembly.name}_brutal_force.gif"
#        out = export_o3d_animation_gif(
#            frames,
#            gif_path,
#            width=640,
#            height=480,
#            fps=10,
#        )
#        print(f"[SAVED FULL GIF] {out}")
#    else:
#        print("[WARN] no successful-removal frames collected")


# if __name__ == "__main__":
#    converted_root = ROOT.parent / "external" / "test_cases_external"
#
#    results_root = ROOT / "tests" / "external_cases" / "results"
#
#    # Run 1000 converted multi_assembly cases in 10 chunks.
#    # Output files:
#    #   test_multi_assembly_1-100_*.csv/jsonl
#    #   test_multi_assembly_101-200_*.csv/jsonl
#    #   ...
#    #   test_multi_assembly_901-1000_*.csv/jsonl
#
#    run_external_batch_brutal_force_in_chunks(
#        converted_root=converted_root,
#        results_root=results_root,
#        group="multi_assembly",
#        seed=0,
#        max_steps=200,
#        observer_debug=False,
#        chunk_size=100,
#        start=201,
#        stop=1000,
#        verbose=True,
#    )
def run_single_case_repeated_rollouts(
    *,
    case_root: Path,
    results_dir: Path,
    num_rollouts: int = 100,
    seed_start: int = 0,
    max_steps: int = 200,
    observer_debug: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the same benchmark case multiple times with different seeds.

    Saves:
    - one compact CSV summary
    - one full JSONL debug log
    """
    case_root = Path(case_root).resolve()
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    xml_dir = case_root / "xml"
    expected_xml = xml_dir / f"{case_root.name}.xml"

    if expected_xml.exists():
        xml_path = expected_xml
    else:
        xml_candidates = sorted(xml_dir.glob("*.xml"))
        if not xml_candidates:
            raise FileNotFoundError(f"No XML found under: {xml_dir}")
        xml_path = xml_candidates[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"test_{case_root.name}_{num_rollouts}_rollouts_{timestamp}"

    csv_path = results_dir / f"{prefix}.csv"
    jsonl_path = results_dir / f"{prefix}.jsonl"

    reports: list[dict] = []

    if verbose:
        print("=" * 80)
        print("[SINGLE CASE REPEATED ROLLOUTS]")
        print(f"case_root:   {case_root}")
        print(f"xml_path:    {xml_path}")
        print(f"results_dir: {results_dir}")
        print(f"num_rollouts: {num_rollouts}")
        print(f"seed range:  {seed_start}-{seed_start + num_rollouts - 1}")
        print("=" * 80)

    for i in range(num_rollouts):
        seed = seed_start + i

        if verbose:
            print("-" * 80)
            print(f"[ROLLOUT] {i + 1}/{num_rollouts} seed={seed}")

        try:
            observer = make_external_case_observer(debug=observer_debug)
            env = make_env_from_imported_xml_auto(
                xml_path,
                observer=observer,
                base_parts=(),
                max_steps=max_steps,
                enable_screwing_if_available=True,
            )

            report = brutal_force_rollout_headless(
                env,
                seed=seed,
                max_steps=max_steps,
                verbose=False,
            )

            report["status"] = "ok"
            report["seed"] = seed
            report["rollout_index"] = i
            report["xml_path"] = str(xml_path)
            report["case_root"] = str(case_root)
            report["group"] = "bolt_and_nut_pipe_benchmark"

            if verbose:
                print(
                    f"[RESULT] seed={seed} "
                    f"success={report['success']} "
                    f"completion={report['completion']:.3f} "
                    f"removed={report['removed_num_parts']}/{report['initial_num_parts']} "
                    f"stop={report['stop_reason']}"
                )

        except Exception as exc:
            report = {
                "assembly_name": case_root.name,
                "group": "bolt_and_nut_pipe_benchmark",
                "case_root": str(case_root),
                "xml_path": str(xml_path),
                "rollout_index": i,
                "seed": seed,
                "status": "failed_setup_or_runtime",
                "success": False,
                "stop_reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }

            if verbose:
                print(f"[ERROR] seed={seed}: {report['error']}")

        reports.append(report)

    fieldnames = [
        "rollout_index",
        "seed",
        "assembly_name",
        "status",
        "success",
        "stop_reason",
        "initial_num_parts",
        "initial_num_connections",
        "initial_num_detected_parts",
        "initial_num_hidden_parts",
        "partial_observability_observed",
        "num_partial_observability_events",
        "removed_num_parts",
        "completion",
        "remaining_num_parts",
        "remaining_num_connections",
        "num_attempts",
        "num_env_steps",
        "xml_path",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in reports:
            writer.writerow(
                {
                    "rollout_index": r.get("rollout_index"),
                    "seed": r.get("seed"),
                    "assembly_name": r.get("assembly_name"),
                    "status": r.get("status", "ok"),
                    "success": r.get("success"),
                    "stop_reason": r.get("stop_reason"),
                    "initial_num_parts": r.get("initial_num_parts"),
                    "initial_num_connections": r.get("initial_num_connections"),
                    "initial_num_detected_parts": r.get("initial_num_detected_parts"),
                    "initial_num_hidden_parts": r.get("initial_num_hidden_parts"),
                    "partial_observability_observed": r.get(
                        "partial_observability_observed"
                    ),
                    "num_partial_observability_events": r.get(
                        "num_partial_observability_events"
                    ),
                    "removed_num_parts": r.get("removed_num_parts"),
                    "completion": r.get("completion"),
                    "remaining_num_parts": r.get("remaining_num_parts"),
                    "remaining_num_connections": r.get("remaining_num_connections"),
                    "num_attempts": r.get("num_attempts"),
                    "num_env_steps": r.get("num_env_steps"),
                    "xml_path": r.get("xml_path"),
                    "error": r.get("error", ""),
                }
            )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")

    if verbose:
        num_ok = sum(1 for r in reports if r.get("status", "ok") == "ok")
        num_success = sum(1 for r in reports if r.get("success") is True)
        print("=" * 80)
        print("[DONE]")
        print(f"ok rollouts: {num_ok}/{len(reports)}")
        print(f"successes:   {num_success}/{len(reports)}")
        print(f"[SAVED CSV]   {csv_path}")
        print(f"[SAVED JSONL] {jsonl_path}")
        print("=" * 80)

    return reports


"""
if __name__ == "__main__":
    converted_root = ROOT.parent / "external" / "test_cases_external"

    results_root = ROOT / "tests" / "external_cases" / "results"

    # ------------------------------------------------------------------
    # Run 1000 converted joint_assembly cases in 10 chunks.
    #
    # Input group:
    #   joint_assembly
    #
    # Output folder:
    #   tests/external_cases/results/joint_assembly_new
    #
    # Output files:
    #   test_joint_assembly_1-100_*.csv/jsonl
    #   test_joint_assembly_101-200_*.csv/jsonl
    #   ...
    #   test_joint_assembly_901-1000_*.csv/jsonl
    # ------------------------------------------------------------------
    run_external_batch_brutal_force_in_chunks(
        converted_root=converted_root,
        results_root=results_root,
        group="joint_assembly",
        output_group="joint_assembly_new",
        seed=0,
        max_steps=200,
        observer_debug=False,
        chunk_size=1000,
        start=1,
        stop=1000,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Run 24 converted rotational joint assembly cases in 1 chunk.
    #
    # Input group:
    #   joint_assembly_rotation
    #
    # Output folder:
    #   tests/external_cases/results/joint_assembly_rotation_new
    #
    # Output files:
    #   test_joint_assembly_rotation_1-24_*.csv/jsonl
    # ------------------------------------------------------------------
    run_external_batch_brutal_force_in_chunks(
        converted_root=converted_root,
        results_root=results_root,
        group="joint_assembly_rotation",
        output_group="joint_assembly_rotation_new",
        seed=0,
        max_steps=200,
        observer_debug=False,
        chunk_size=24,
        start=1,
        stop=24,
        verbose=True,
    )
"""
if __name__ == "__main__":
    case_root = ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case"

    results_dir = (
        ROOT
        / "tests"
        / "internal_cases"
        / "bolt_and_nut_pipe_benchmark"
        / "translation_feasiblity_check"
    )

    run_single_case_repeated_rollouts(
        case_root=case_root,
        results_dir=results_dir,
        num_rollouts=100,
        seed_start=0,
        max_steps=200,
        observer_debug=False,
        verbose=True,
    )
