#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

DEFAULT_SCREWDRIVERS = ["H5", "T6", "H6", "Tx45", "M4"]

# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------
# This script is intended to live under:
#   /home/hans/Uni/Bachelorarbeit/pomdp_disassembly_gym/tools/
# Therefore parents[1] is the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
from disassembly_pomdp.eol_operator import EOLUnifiedSampler
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.pomdp_observer import ObserverConfig, SimVisibilityObserver
from disassembly_pomdp.screwing_capability import ScrewingCapability


# -----------------------------------------------------------------------------
# Colors: KIT / wbk-inspired palette for presentation rendering
# -----------------------------------------------------------------------------
def rgb255_to_rgb01(rgb: list[int] | tuple[int, int, int]) -> list[float]:
    return (np.asarray(rgb, dtype=np.float64) / 255.0).tolist()


KIT_DEEP_BLUE = rgb255_to_rgb01([20, 68, 102])
KIT_BLUE_GREEN = rgb255_to_rgb01([0, 131, 143])
KIT_YELLOW = rgb255_to_rgb01([238, 183, 13])
KIT_RED = rgb255_to_rgb01([178, 55, 44])
WBK_GREEN = rgb255_to_rgb01([0, 150, 130])
WBK_GREY = rgb255_to_rgb01([59, 79, 88])
WBK_LIGHT_BLUE = rgb255_to_rgb01([60, 152, 217])
WBK_LIGHT_GREEN = rgb255_to_rgb01([39, 255, 226])
BLACK = rgb255_to_rgb01([18, 22, 28])

ASSEMBLY_PALETTE = [
    KIT_DEEP_BLUE,
    KIT_RED,
    KIT_BLUE_GREEN,
    KIT_YELLOW,
    WBK_GREEN,
    WBK_LIGHT_BLUE,
    WBK_LIGHT_GREEN,
    BLACK,
]


# -----------------------------------------------------------------------------
# Environment construction, adapted from the existing external-case build script
# -----------------------------------------------------------------------------
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


def load_imported_assembly(xml_path: str | Path) -> Assembly:
    xml_path = Path(xml_path).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    old_cwd = Path.cwd()
    try:
        # Some converted/internal XML files store relative paths such as
        # EMO_SR_V2_data/*.npy. Assembly.load_from_xml resolves these
        # relative to the current working directory, so temporarily switch
        # to the XML folder.
        os.chdir(xml_path.parent)
        return Assembly.load_from_xml(str(xml_path))
    finally:
        os.chdir(old_cwd)


def make_planner(asm: Assembly) -> HierarchicalDisassemblySequencePlanner:
    return HierarchicalDisassemblySequencePlanner(
        assembly=asm,
        cost_milling=10.0,
        cost_screwing=5.0,
        cost_gripping_and_removing=0.0,
        cost_safety_ring=5.0,
        mill_diameter=6.0,
        path_to_working_dir=None,
        available_screwdrivers=DEFAULT_SCREWDRIVERS,
    )


def has_screw_connections(asm: Assembly) -> bool:
    return any(getattr(c, "connection_type", None) == "screw" for c in asm.connections)


def make_plan_screwing_adapter(planner: HierarchicalDisassemblySequencePlanner):
    def _merge_without_result_assembly(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if k != "result_assembly":
                dst[k] = v

    def plan_screwing_adapter(assembly: Assembly, op: ScrewingOperation, rng=None):
        data = getattr(op, "data", {})
        if not isinstance(data, dict):
            return op, None

        connection = data.get("connection")
        if connection is None:
            return op, None

        planned_op, candidate_asm = planner.plan_screwing(
            assembly=assembly,
            connection=connection,
            cost_screwing=float(data.get("cost", 5.0)),
        )

        out_op = planned_op if planned_op is not None else op
        out_data: dict[str, Any] = {}
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
            if k != "result_assembly":
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

        planned_op, candidate_asm = planner.plan_gripping_and_removing(
            assembly=assembly,
            part=part,
            cost_gripping_and_removing=float(data.get("cost", 0.0)),
            max_grip_width=1e6,
        )

        out_op = planned_op if planned_op is not None else op
        out_data: dict[str, Any] = {}
        _merge_without_result_assembly(out_data, data)
        if isinstance(getattr(out_op, "data", None), dict):
            _merge_without_result_assembly(out_data, out_op.data)

        out_data["planned"] = True
        valid_new_asm = (
            isinstance(candidate_asm, Assembly)
            and candidate_asm.get_part_by_name(part_name) is None
        )

        if valid_new_asm:
            out_data["result_assembly"] = candidate_asm
            out_op.data = out_data
            return out_op, candidate_asm

        out_op.data = out_data
        return out_op, None

    return plan_gripping_and_removing_adapter


def make_external_case_observer(debug: bool = False) -> SimVisibilityObserver:
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


def make_env_from_imported_xml_auto(
    xml_path: str | Path,
    observer,
    *,
    base_parts: tuple[str, ...] = (),
    target_part_names: list[str] | None = None,
    max_steps: int = 200,
    enable_screwing_if_available: bool = True,
) -> DisassemblyPOMDPEnv:
    if observer is None:
        raise ValueError("POMDP external-case env requires a real observer.")

    base_asm = load_imported_assembly(xml_path)
    settings = make_cell_settings()

    all_part_names = sorted([p.name for p in base_asm.parts.values()])
    base_part_set = set(base_parts)
    if target_part_names is None:
        target_part_names = [n for n in all_part_names if n not in base_part_set]
    else:
        target_part_names = [
            n
            for n in target_part_names
            if n in set(all_part_names) and n not in base_part_set
        ]

    planner = make_planner(copy.deepcopy(base_asm))
    capabilities = []

    if enable_screwing_if_available and has_screw_connections(base_asm):
        capabilities.append(
            ScrewingCapability(
                available_screwdrivers=DEFAULT_SCREWDRIVERS,
                cell_settings=settings,
                operation_cls=ScrewingOperation,
                plan_func=make_plan_screwing_adapter(planner),
            )
        )

    capabilities.append(
        GrippingAndRemovingCapability.from_assembly(
            base_asm,
            available_grippers=["H5"],
            cell_settings=settings,
            operation_cls=GrippingAndRemovingOperation,
            plan_func=make_plan_gripping_and_removing_adapter(planner),
            base_parts=base_parts,
        )
    )

    return DisassemblyPOMDPEnv(
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
        world_sampler=EOLUnifiedSampler(
            p_stuck_screw=0.0,
            p_stuck_upper=0.0,
            p_missing_screw=0.0,
        ),
        eol_op=None,
        expose_eol_belief=False,
        observer=observer,
    )


# -----------------------------------------------------------------------------
# Action selection
# -----------------------------------------------------------------------------
def _get_screwdriver_order(cap) -> list[str]:
    """Best-effort recovery of the screwdriver order used by ScrewingCapability."""
    for attr in (
        "available_screwdrivers",
        "_available_screwdrivers",
        "screwdrivers",
        "_screwdrivers",
        "tool_names",
        "_tool_names",
    ):
        value = getattr(cap, attr, None)
        if value:
            try:
                return [str(x) for x in value]
            except TypeError:
                pass

    # Fallback: this is the order passed into ScrewingCapability above.
    try:
        n_tools = int(cap.action_space.nvec[0])
        return DEFAULT_SCREWDRIVERS[:n_tools]
    except Exception:
        return list(DEFAULT_SCREWDRIVERS)


def find_screw_action_for_target(env, target_name: str) -> np.ndarray:
    for cap_idx, cap in enumerate(env.capabilities):
        if isinstance(cap, ScrewingCapability):
            if (
                not hasattr(cap, "_global_screw_conn_keys")
                or not cap._global_screw_conn_keys
            ):
                continue

            screwdriver_order = _get_screwdriver_order(cap)

            for conn_idx, key in enumerate(cap._global_screw_conn_keys):
                screw_name, _, screw_head = key
                if screw_name != target_name:
                    continue

                screw_head = str(screw_head)
                if screw_head in screwdriver_order:
                    tool_idx = screwdriver_order.index(screw_head)
                else:
                    print(
                        f"[WARN] screwdriver head {screw_head!r} for {target_name} "
                        f"not in capability tools {screwdriver_order}; falling back to tool index 0"
                    )
                    tool_idx = 0

                return np.array([cap_idx, int(tool_idx), int(conn_idx)], dtype=np.int64)

    raise RuntimeError(f"Target {target_name} not found in screwing capability")


def prepare_toolchange_for_legacy_env(env, action: np.ndarray) -> None:
    """
    Compatibility workaround for the current DisassemblyPOMDPEnv tool-change gate.

    Some env versions only allow a toolchange override when the previous attempted
    capability differs from the current one. For consecutive screw actions this can
    reject a valid H5->H6 / H6->H5 screwdriver change before the actual action is
    decoded. For presentation playback we explicitly clear the previous-capability
    guard right before a valid screwing action that requests a different tool.

    This does not force success and does not bypass head compatibility: the action
    still has to request the screwdriver encoded by its tool_idx, and the env hard
    gate still checks the screw head against that requested tool.
    """
    try:
        cap_idx, sub0, sub1 = [int(x) for x in np.asarray(action, dtype=int).reshape(3)]
        cap = env.capabilities[cap_idx]
    except Exception:
        return

    if not isinstance(cap, ScrewingCapability):
        return

    try:
        sem = cap.decode_action(env.assembly, np.array([sub0, sub1], dtype=np.int64))
        requested_tool = getattr(sem, "tool_name", None)
    except Exception:
        return

    mounted_tool = getattr(env, "current_mounted_tool", None)
    if (
        isinstance(requested_tool, str)
        and requested_tool
        and mounted_tool != requested_tool
    ):
        # Clear the same-capability guard so env._validate_or_reject_predecode()
        # can use its existing toolchange override path for this valid request.
        if hasattr(env, "_last_attempted_cap_idx_prev"):
            env._last_attempted_cap_idx_prev = None
        if hasattr(env, "_last_attempted_cap_idx"):
            env._last_attempted_cap_idx = None


def find_grip_action_for_target(env, target_name: str) -> np.ndarray:
    for cap_idx, cap in enumerate(env.capabilities):
        if isinstance(cap, GrippingAndRemovingCapability):
            try:
                part_idx = cap.name_to_part_index[target_name]
                return np.array([cap_idx, 0, int(part_idx)], dtype=np.int64)
            except Exception as exc:
                raise RuntimeError(
                    f"Target {target_name} not found in gripping_and_removing capability"
                ) from exc
    raise RuntimeError("GrippingAndRemovingCapability not found")


def find_action_for_target(env, target_name: str) -> np.ndarray:
    try:
        return find_screw_action_for_target(env, target_name)
    except RuntimeError:
        return find_grip_action_for_target(env, target_name)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def get_screw_part_names(asm: Assembly) -> set[str]:
    screw_names = set()
    for c in asm.connections:
        if getattr(c, "connection_type", None) == "screw":
            part_name = getattr(c, "part_name1", None)
            if part_name is not None:
                screw_names.add(part_name)
    return screw_names


def paint_mesh_copy(
    mesh: o3d.geometry.TriangleMesh, color: list[float]
) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m


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


def assembly_to_o3d_geoms(
    asm: Assembly,
    part_color_map: dict[str, list[float]] | None = None,
    *,
    skip_part: str | None = None,
) -> list[o3d.geometry.Geometry]:
    geoms: list[o3d.geometry.Geometry] = []
    screw_names = get_screw_part_names(asm)
    palette_idx = 0

    for part in asm.parts.values():
        if skip_part is not None and part.name == skip_part:
            continue

        mesh = getattr(part, "mesh", None)
        if mesh is None:
            continue

        if part_color_map is not None and part.name in part_color_map:
            color = part_color_map[part.name]
        elif part.name in screw_names:
            color = WBK_GREY
        else:
            color = ASSEMBLY_PALETTE[palette_idx % len(ASSEMBLY_PALETTE)]
            palette_idx += 1

        geoms.append(paint_mesh_copy(mesh, color))

    return geoms


def compute_scene_bounds_from_geoms(
    geoms: list[o3d.geometry.Geometry],
    *,
    include_lines: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mins = []
    maxs = []

    for g in geoms:
        if isinstance(g, o3d.geometry.TriangleMesh):
            aabb = g.get_axis_aligned_bounding_box()
        elif include_lines and isinstance(g, o3d.geometry.LineSet):
            aabb = g.get_axis_aligned_bounding_box()
        else:
            continue

        mins.append(np.asarray(aabb.min_bound, dtype=float))
        maxs.append(np.asarray(aabb.max_bound, dtype=float))

    if not mins:
        raise ValueError("No geometry bounds available")

    min_bound = np.min(np.vstack(mins), axis=0)
    max_bound = np.max(np.vstack(maxs), axis=0)
    center = 0.5 * (min_bound + max_bound)
    diag = float(np.linalg.norm(max_bound - min_bound))
    return min_bound, max_bound, center, max(diag, 1e-3)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def add_removed_gallery(
    frame: list[o3d.geometry.Geometry],
    removed_items: list[tuple[str, o3d.geometry.TriangleMesh, bool]],
    reference_geoms: list[o3d.geometry.Geometry],
) -> list[o3d.geometry.Geometry]:
    """
    Keep removed screws on the left and normal parts on the right.
    This is intentionally simpler than the GIF-gallery implementation,
    because it should remain robust in an interactive presentation window.
    """
    if not removed_items:
        return frame

    # At the very last disassembly step the current assembly can be empty.
    # In that case ``reference_geoms`` may contain no TriangleMesh, but the
    # removed-part gallery still needs a non-empty anchor for layout.
    # Prefer the provided reference; fall back to the currently visible frame;
    # finally fall back to the removed item meshes themselves.
    anchor_geoms = list(reference_geoms or [])
    if not anchor_geoms:
        anchor_geoms = list(frame or [])
    if not anchor_geoms:
        anchor_geoms = [mesh for _, mesh, _ in removed_items]

    _, _, center, diag = compute_scene_bounds_from_geoms(
        anchor_geoms, include_lines=False
    )
    diag = max(diag, 1.0)

    eye = center + diag * np.array([1.2, 1.2, 0.8])
    view_dir = normalize(center - eye)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = normalize(np.cross(view_dir, up))
    up_cam = normalize(np.cross(right, view_dir))

    screws = [(name, mesh) for name, mesh, is_screw in removed_items if is_screw]
    parts = [(name, mesh) for name, mesh, is_screw in removed_items if not is_screw]

    out = [copy.deepcopy(g) for g in frame]

    def place(items, side_sign: float, max_rows: int, target_scale: float) -> None:
        if not items:
            return
        row_spacing = diag * 0.18
        col_spacing = diag * 0.22
        side_offset = diag * 1.15
        target_extent = max(diag * target_scale, 5.0)
        plane_center = center + view_dir * (0.25 * diag)

        for idx, (_, mesh) in enumerate(items):
            row = idx % max_rows
            col = idx // max_rows
            row_offset = (0.5 * (max_rows - 1) - row) * row_spacing

            m = o3d.geometry.TriangleMesh(mesh)
            m.compute_vertex_normals()
            aabb = m.get_axis_aligned_bounding_box()
            part_center = np.asarray(aabb.get_center(), dtype=float)
            max_extent = float(np.max(aabb.get_extent()))
            if max_extent > target_extent and max_extent > 1e-9:
                m.scale(target_extent / max_extent, center=part_center)
                aabb = m.get_axis_aligned_bounding_box()
                part_center = np.asarray(aabb.get_center(), dtype=float)

            target_center = (
                plane_center
                + side_sign * right * (side_offset + col * col_spacing)
                + up_cam * row_offset
            )
            m.translate(target_center - part_center)
            out.append(m)

    place(screws, side_sign=-1.0, max_rows=8, target_scale=0.07)
    place(parts, side_sign=1.0, max_rows=5, target_scale=0.22)
    return out


def infer_removed_part_name(op) -> str | None:
    if op is None or not isinstance(getattr(op, "data", None), dict):
        return None
    data = op.data
    if isinstance(data.get("part_name"), str):
        return data["part_name"]
    conn = data.get("connection", None)
    if conn is not None:
        part_name = getattr(conn, "part_name1", None)
        if isinstance(part_name, str):
            return part_name
    return None


def is_removed_part_a_screw(assembly_before: Assembly, part_name: str) -> bool:
    lname = part_name.lower()
    if "screw" in lname or "schraube" in lname:
        return True
    for c in assembly_before.connections:
        if getattr(c, "connection_type", None) == "screw":
            if getattr(c, "part_name1", None) == part_name:
                return True
    return False


def fallback_animation_frames(
    assembly_before: Assembly,
    op,
    removed_part_name: str,
    base_color_map: dict[str, list[float]],
    *,
    num_frames: int,
) -> list[list[o3d.geometry.Geometry]]:
    """Fallback animation if an operation has no animation_frames_o3d method."""
    moving_part = assembly_before.get_part_by_name(removed_part_name)
    if moving_part is None or getattr(moving_part, "mesh", None) is None:
        return []

    base_geoms = assembly_to_o3d_geoms(
        copy.deepcopy(assembly_before),
        part_color_map=base_color_map,
        skip_part=removed_part_name,
    )
    _, _, _, diag = compute_scene_bounds_from_geoms(base_geoms, include_lines=False)

    direction = np.array([0.0, 0.0, 1.0], dtype=float)
    data = getattr(op, "data", {}) if op is not None else {}
    if isinstance(data, dict) and "direction" in data:
        try:
            direction = normalize(np.asarray(data["direction"], dtype=float).reshape(3))
        except Exception:
            pass

    distance = max(0.35 * diag, 20.0)
    color = base_color_map.get(removed_part_name, WBK_GREY)

    frames: list[list[o3d.geometry.Geometry]] = []
    for i in range(max(num_frames, 2)):
        alpha = i / max(num_frames - 1, 1)
        moving = paint_mesh_copy(moving_part.mesh, color)
        moving.translate(direction * distance * alpha)
        frames.append([copy.deepcopy(g) for g in base_geoms] + [moving])

    return frames


# -----------------------------------------------------------------------------
# Rollout frame collection for interactive presentation
# -----------------------------------------------------------------------------
def collect_rollout_frames_for_window(
    env: DisassemblyPOMDPEnv,
    *,
    seed: int,
    max_steps: int | None,
    num_frames_per_step: int,
    hold_frames: int,
    final_hold_frames: int,
    target_order: list[str] | None,
    use_gallery: bool,
    verbose: bool,
) -> tuple[bool, list[list[o3d.geometry.Geometry]]]:
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    if max_steps is None:
        max_steps = env.max_steps

    base_color_map = build_part_color_map(copy.deepcopy(env.base_assembly))
    removed_items: list[tuple[str, o3d.geometry.TriangleMesh, bool]] = []
    frames: list[list[o3d.geometry.Geometry]] = []

    initial_geoms = assembly_to_o3d_geoms(
        copy.deepcopy(env.assembly), part_color_map=base_color_map
    )
    # Keep a stable non-empty scene anchor for gallery layout. This remains
    # valid even after the final target has been removed and env.assembly is empty.
    scene_anchor_geoms = [copy.deepcopy(g) for g in initial_geoms]
    initial_combined = (
        add_removed_gallery(initial_geoms, removed_items, scene_anchor_geoms)
        if use_gallery
        else initial_geoms
    )
    for _ in range(max(hold_frames, 1)):
        frames.append([copy.deepcopy(g) for g in initial_combined])

    if verbose:
        print("=" * 80)
        print("[DEFENSE DEMO] Interactive rollout frame collection")
        print(f"assembly: {env.base_assembly.name}")
        print(f"targets:  {env.target_part_names}")
        print(f"parts:    {sorted([p.name for p in env.assembly.parts.values()])}")
        print("=" * 80)

    order_queue = list(target_order or [])

    for t in range(max_steps):
        remaining_targets = [
            name
            for name in env.target_part_names
            if env.assembly.get_part_by_name(name) is not None
        ]

        if not remaining_targets:
            if verbose:
                print("[DONE] all targets removed")
            for _ in range(max(final_hold_frames, 0)):
                frames.append([copy.deepcopy(g) for g in frames[-1]])
            return True, frames

        if order_queue:
            target = None
            while order_queue and target is None:
                candidate = order_queue.pop(0)
                if candidate in remaining_targets:
                    target = candidate
            if target is None:
                target = str(rng.choice(remaining_targets))
        else:
            target = str(rng.choice(remaining_targets))

        assembly_before = copy.deepcopy(env.assembly)
        action = find_action_for_target(env, target)
        prepare_toolchange_for_legacy_env(env, action)
        obs, reward, terminated, truncated, info = env.step(action)

        if verbose:
            print(
                f"[{t:03d}] target={target:<16} action={action.tolist()} "
                f"reward={reward:8.3f} success={info.get('success')} "
                f"term={terminated} trunc={truncated}"
            )

        if info.get("success", False):
            semantic_action = info.get("semantic_action", None)
            op = getattr(semantic_action, "operation", None)
            removed_part_name = infer_removed_part_name(op)

            step_frames: list[list[o3d.geometry.Geometry]] = []
            if op is not None and hasattr(op, "animation_frames_o3d"):
                step_frames = op.animation_frames_o3d(
                    assembly_before,
                    num_frames=num_frames_per_step,
                )

            if not step_frames and removed_part_name is not None:
                step_frames = fallback_animation_frames(
                    assembly_before,
                    op,
                    removed_part_name,
                    base_color_map,
                    num_frames=num_frames_per_step,
                )

            if use_gallery and step_frames:
                ref_geoms = assembly_to_o3d_geoms(
                    copy.deepcopy(assembly_before),
                    part_color_map=base_color_map,
                )
                step_frames = [
                    add_removed_gallery(frame, removed_items, ref_geoms)
                    for frame in step_frames
                ]

            frames.extend(step_frames)

            if removed_part_name is not None:
                removed_part_obj = assembly_before.get_part_by_name(removed_part_name)
                if (
                    removed_part_obj is not None
                    and getattr(removed_part_obj, "mesh", None) is not None
                ):
                    color = base_color_map.get(removed_part_name, WBK_GREY)
                    is_screw = is_removed_part_a_screw(
                        assembly_before, removed_part_name
                    )
                    removed_items.append(
                        (
                            removed_part_name,
                            paint_mesh_copy(removed_part_obj.mesh, color),
                            is_screw,
                        )
                    )

            post_geoms = assembly_to_o3d_geoms(
                copy.deepcopy(env.assembly), part_color_map=base_color_map
            )
            # If the rollout has finished, post_geoms can be empty. Use the
            # initial assembly as the stable gallery anchor in that situation.
            gallery_reference = post_geoms if post_geoms else scene_anchor_geoms
            post_combined = (
                add_removed_gallery(post_geoms, removed_items, gallery_reference)
                if use_gallery
                else post_geoms
            )
            for _ in range(max(hold_frames, 0)):
                frames.append([copy.deepcopy(g) for g in post_combined])

        if terminated:
            if verbose:
                print("[DONE] environment terminated")
            for _ in range(max(final_hold_frames, 0)):
                frames.append([copy.deepcopy(g) for g in frames[-1]])
            return True, frames

        if truncated:
            if verbose:
                print("[WARN] environment truncated")
            for _ in range(max(final_hold_frames, 0)):
                frames.append([copy.deepcopy(g) for g in frames[-1]])
            return False, frames

    if verbose:
        print("[WARN] max_steps reached")
    for _ in range(max(final_hold_frames, 0)):
        frames.append([copy.deepcopy(g) for g in frames[-1]])
    return False, frames


# -----------------------------------------------------------------------------
# Interactive Open3D window player
# -----------------------------------------------------------------------------
def configure_camera(
    vis: o3d.visualization.Visualizer,
    reference_geoms: list[o3d.geometry.Geometry],
    *,
    zoom: float,
) -> None:
    _, _, center, diag = compute_scene_bounds_from_geoms(
        reference_geoms, include_lines=False
    )
    eye = center + diag * np.array([1.2, 1.2, 0.8], dtype=float)
    front = normalize(center - eye)

    vc = vis.get_view_control()
    vc.set_lookat(center.tolist())
    vc.set_front(front.tolist())
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_zoom(float(zoom))


def play_o3d_frames_window(
    frames: list[list[o3d.geometry.Geometry]],
    *,
    title: str,
    fps: float,
    width: int,
    height: int,
    loop: bool,
    start_paused: bool,
    zoom: float,
) -> None:
    if not frames:
        raise ValueError("No frames to play")

    print("=" * 80)
    print("[OPEN3D WINDOW CONTROLS]")
    print("  Space : pause / resume")
    print("  N     : next frame")
    print("  B     : previous frame")
    print("  R     : restart")
    print("  Q     : close window")
    print("=" * 80)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=width, height=height)

    state = {
        "idx": 0,
        "paused": bool(start_paused),
        "last_time": time.perf_counter(),
        "geoms": [],
        "should_close": False,
    }

    frame_period = 1.0 / max(float(fps), 1e-6)

    def clear_current_geometries() -> None:
        for g in state["geoms"]:
            vis.remove_geometry(g, reset_bounding_box=False)
        state["geoms"] = []

    def load_frame(idx: int, *, reset_camera_once: bool = False) -> None:
        idx = int(np.clip(idx, 0, len(frames) - 1))
        state["idx"] = idx
        clear_current_geometries()
        current = [copy.deepcopy(g) for g in frames[idx]]
        for g in current:
            vis.add_geometry(g, reset_bounding_box=reset_camera_once)
        state["geoms"] = current
        if reset_camera_once:
            configure_camera(vis, frames[0], zoom=zoom)
        vis.poll_events()
        vis.update_renderer()

    def step(delta: int) -> None:
        new_idx = state["idx"] + delta
        if new_idx >= len(frames):
            new_idx = 0 if loop else len(frames) - 1
            if not loop:
                state["paused"] = True
        elif new_idx < 0:
            new_idx = len(frames) - 1 if loop else 0
        load_frame(new_idx)
        print(f"[FRAME] {state['idx'] + 1}/{len(frames)}")

    def on_space(vis_cb):
        state["paused"] = not state["paused"]
        print("[PLAYER] paused" if state["paused"] else "[PLAYER] playing")
        return False

    def on_next(vis_cb):
        state["paused"] = True
        step(1)
        return False

    def on_prev(vis_cb):
        state["paused"] = True
        step(-1)
        return False

    def on_restart(vis_cb):
        state["paused"] = False
        load_frame(0)
        print("[PLAYER] restarted")
        return False

    def on_quit(vis_cb):
        state["should_close"] = True
        return False

    def animation_callback(vis_cb):
        if state["should_close"]:
            vis_cb.close()
            return False

        now = time.perf_counter()
        if not state["paused"] and now - state["last_time"] >= frame_period:
            state["last_time"] = now
            step(1)
        return False

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("N"), on_next)
    vis.register_key_callback(ord("B"), on_prev)
    vis.register_key_callback(ord("R"), on_restart)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_animation_callback(animation_callback)

    load_frame(0, reset_camera_once=True)
    vis.run()
    vis.destroy_window()


def save_o3d_frames_video(
    frames: list[list[o3d.geometry.Geometry]],
    *,
    output_path: str | Path,
    title: str,
    fps: float,
    width: int,
    height: int,
    zoom: float,
    codec: str = "libx264",
    crf: int = 18,
) -> None:
    """Render Open3D frames directly to an mp4/mkv video via ffmpeg.

    This avoids screen recording. The same camera setup as the interactive
    player is used, so the result is deterministic for a fixed rollout seed,
    window size, fps, and zoom.
    """
    if not frames:
        raise ValueError("No frames to save")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        str(codec),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    print("=" * 80)
    print("[VIDEO EXPORT]")
    print(f"  output : {output_path}")
    print(f"  frames : {len(frames)}")
    print(f"  fps    : {fps}")
    print(f"  size   : {width}x{height}")
    print(f"  codec  : {codec}, crf={crf}")
    print("=" * 80)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=int(width),
        height=int(height),
        visible=False,
    )

    current_geoms: list[o3d.geometry.Geometry] = []

    def clear_current_geometries() -> None:
        nonlocal current_geoms
        for g in current_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        current_geoms = []

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        if proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")

        try:
            for idx, frame in enumerate(frames):
                clear_current_geometries()

                current_geoms = [copy.deepcopy(g) for g in frame]
                for g in current_geoms:
                    vis.add_geometry(g, reset_bounding_box=(idx == 0))

                if idx == 0:
                    configure_camera(vis, frames[0], zoom=zoom)

                vis.poll_events()
                vis.update_renderer()

                img = vis.capture_screen_float_buffer(do_render=True)
                arr = np.asarray(img)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise RuntimeError(f"Unexpected captured image shape: {arr.shape}")
                if arr.shape[0] != int(height) or arr.shape[1] != int(width):
                    raise RuntimeError(
                        f"Captured image has shape {arr.shape}, expected "
                        f"({int(height)}, {int(width)}, 3)."
                    )

                rgb = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                proc.stdin.write(rgb.tobytes())

                if (idx + 1) % 50 == 0 or idx + 1 == len(frames):
                    print(f"[VIDEO] frame {idx + 1}/{len(frames)}")

        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    finally:
        clear_current_geometries()
        vis.destroy_window()

    print(f"[VIDEO] Done: {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def resolve_xml_path(args: argparse.Namespace) -> Path:
    if args.xml_path is not None:
        return Path(args.xml_path).expanduser().resolve()

    if args.case_root is not None:
        case_root = Path(args.case_root).expanduser().resolve()
        candidate = case_root / "xml" / f"{case_root.name}.xml"
        if candidate.exists():
            return candidate
        xmls = sorted((case_root / "xml").glob("*.xml"))
        if xmls:
            return xmls[0]
        raise FileNotFoundError(f"No XML found under {case_root / 'xml'}")

    if args.converted_root is None or args.group is None or args.case_id is None:
        raise ValueError(
            "Provide either --xml-path, --case-root, or all of "
            "--converted-root --group --case-id."
        )

    case_id = str(args.case_id)
    if case_id.isdigit():
        case_id = case_id.zfill(5)

    case_name = f"{args.group}_{case_id}"
    xml_path = (
        Path(args.converted_root).expanduser().resolve()
        / case_name
        / "xml"
        / f"{case_name}.xml"
    )
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    return xml_path


def parse_comma_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_target_order(raw: str | None) -> list[str] | None:
    values = parse_comma_list(raw)
    return values or None


def emo_sr_v2_rotor_preset() -> tuple[tuple[str, ...], list[str], list[str]]:
    """Defense-friendly EMO_SR_V2 sequence: free the lids, then remove the rotor.

    The stator is treated as the fixture/base. Missing parts are skipped at runtime.
    """
    base_parts = ("02_stator", "08_lower_lid")
    sequence = [
        "04_screw_1",
        "05_screw_2",
        "06_screw_3",
        "07_screw_4",
        "01_upper_lid",
        "03_rotor_short",
    ]
    target_parts = list(sequence)
    return base_parts, target_parts, sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Open3D thesis-defense renderer for POMDP disassembly rollouts."
    )

    source = parser.add_argument_group("case source")
    source.add_argument(
        "--xml-path", default=None, help="Direct path to a converted case XML"
    )
    source.add_argument(
        "--case-root", default=None, help="Case directory containing xml/<case>.xml"
    )
    source.add_argument(
        "--converted-root", default=None, help="Root of converted external cases"
    )
    source.add_argument("--group", default=None, help="Case group, e.g. multi_assembly")
    source.add_argument("--case-id", default=None, help="Case id, e.g. 00643 or 643")

    rollout = parser.add_argument_group("rollout")
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--max-steps", type=int, default=200)
    rollout.add_argument("--frames-per-step", type=int, default=25)
    rollout.add_argument("--hold-frames", type=int, default=8)
    rollout.add_argument("--final-hold-frames", type=int, default=30)
    rollout.add_argument(
        "--target-order", default=None, help="Optional comma-separated target order"
    )
    rollout.add_argument(
        "--target-parts",
        default="",
        help="Optional comma-separated target parts; defaults to all non-base parts",
    )
    rollout.add_argument(
        "--base-parts", default="", help="Optional comma-separated base parts"
    )
    rollout.add_argument(
        "--emo-sr-v2-preset",
        action="store_true",
        help="Use a defense-friendly EMO_SR_V2 rotor-removal preset",
    )
    rollout.add_argument("--disable-screwing", action="store_true")
    rollout.add_argument("--observer-debug", action="store_true")
    rollout.add_argument(
        "--no-gallery",
        action="store_true",
        help="Do not park removed parts on the sides",
    )

    window = parser.add_argument_group("window")
    window.add_argument("--fps", type=float, default=12.0)
    window.add_argument("--width", type=int, default=1280)
    window.add_argument("--height", type=int, default=900)
    window.add_argument("--loop", action="store_true")
    window.add_argument(
        "--manual",
        action="store_true",
        help="Open paused; use N/B to step through frames",
    )
    window.add_argument("--zoom", type=float, default=0.62)
    window.add_argument(
        "--video-path",
        default=None,
        help="Optional output path for direct mp4/mkv export",
    )
    window.add_argument(
        "--no-window",
        action="store_true",
        help="Export video without opening the interactive window",
    )
    window.add_argument(
        "--video-codec", default="libx264", help="ffmpeg video codec for --video-path"
    )
    window.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="ffmpeg CRF quality value; lower means higher quality",
    )

    args = parser.parse_args()

    xml_path = resolve_xml_path(args)

    if args.emo_sr_v2_preset:
        preset_base_parts, preset_target_parts, preset_sequence = (
            emo_sr_v2_rotor_preset()
        )
        base_parts = preset_base_parts
        target_parts = preset_target_parts
        target_order = parse_target_order(args.target_order) or preset_sequence
    else:
        base_parts = tuple(parse_comma_list(args.base_parts))
        target_parts = parse_comma_list(args.target_parts) or None
        target_order = parse_target_order(args.target_order)

    print(f"[CASE] xml_path={xml_path}")
    if base_parts:
        print(f"[CASE] base_parts={list(base_parts)}")
    if target_parts:
        print(f"[CASE] target_parts={target_parts}")
    if target_order:
        print(f"[CASE] target_order={target_order}")

    observer = make_external_case_observer(debug=args.observer_debug)
    env = make_env_from_imported_xml_auto(
        xml_path,
        observer=observer,
        base_parts=base_parts,
        target_part_names=target_parts,
        max_steps=args.max_steps,
        enable_screwing_if_available=not args.disable_screwing,
    )

    success, frames = collect_rollout_frames_for_window(
        env,
        seed=args.seed,
        max_steps=args.max_steps,
        num_frames_per_step=args.frames_per_step,
        hold_frames=args.hold_frames,
        final_hold_frames=args.final_hold_frames,
        target_order=target_order,
        use_gallery=not args.no_gallery,
        verbose=True,
    )

    print(f"[RESULT] rollout_success={success}")
    print(f"[RESULT] frames={len(frames)}")

    title = f"Defense Demo: {env.base_assembly.name} | success={success}"

    if args.video_path is not None:
        save_o3d_frames_video(
            frames,
            output_path=args.video_path,
            title=title,
            fps=args.fps,
            width=args.width,
            height=args.height,
            zoom=args.zoom,
            codec=args.video_codec,
            crf=args.video_crf,
        )

    if not args.no_window:
        play_o3d_frames_window(
            frames,
            title=title,
            fps=args.fps,
            width=args.width,
            height=args.height,
            loop=args.loop,
            start_paused=args.manual,
            zoom=args.zoom,
        )


if __name__ == "__main__":
    main()
