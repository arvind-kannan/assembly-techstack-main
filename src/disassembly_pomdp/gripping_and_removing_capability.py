"""gripping_and_removing capability implementation for the disassembly POMDP gym.

The capability exposes gripper-based part-removal actions, maps them to
semantic actions, optionally delegates geometric operation planning, and
provides lightweight feasibility, transition, timing, and visualization hooks.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot
from gymnasium import spaces

from disassembly_pomdp.capability_base import (
    Capability,
    TargetRef,
    planned_operation,
    suppress_stdout,
)
from disassembly_pomdp.config import CellSettings
from motion_planning.removal_planning import part_removed

# The many defensive try/except blocks below are intentional.  This module is
# used with multiple Assembly/Operation implementations during experiments.
# pylint: disable=broad-exception-caught,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-public-methods,too-many-return-statements
# pylint: disable=too-many-branches,too-many-locals,too-few-public-methods

BUFFER_MILLED_PREFIX = "buffer_milled_"
DEFAULT_BASE_PARTS: tuple[str, ...] = ()


# Planner signature: (Assembly, Operation) -> Operation
PlanFunc = Callable[[Assembly, Any], Any]


@dataclass
class GrippingAndRemovingAction:
    """Semantic action: Use a gripper to grab a part."""

    tool_name: str  # gripper name
    part_name: str  # Target object (original non-screw or milled part placeholder name)
    gripper_id: int  # Tool index (tool_idx)
    part_index: int  # Position in the global part index (part_idx)
    operation: Any | None = None  # Stage2: attached Operation


class GrippingAndRemovingCapability(Capability):
    """
    gripping_and_removing capability: Use the gripper to grab/attempt to remove a part.

    Action is encoded as:
        raw_action = [tool_idx, part_idx]    (MultiDiscrete)
    where:
        tool_idx   ∈ [0, ..., n_tools)
        part_idx   ∈ [0, ..., N_global_parts)

    The second dimension, part_idx, is the "global part index,"
    corresponding to ``self.global_part_names``.
    """

    process = Process.GRIPPING_AND_REMOVING

    def __init__(
        self,
        name: str,
        available_grippers: Sequence[str],
        global_part_names: Sequence[str],
        cell_settings: CellSettings | None = None,
        operation_cls: Any | None = None,
        plan_func: PlanFunc | None = None,
        base_parts: Sequence[str] = DEFAULT_BASE_PARTS,
        robots: Sequence[Robot] | None = None,
    ):
        super().__init__(name, robots=robots)
        self.available_grippers: list[str] = list(available_grippers)

        self.global_part_names: list[str] = list(global_part_names)
        self.name_to_part_index: dict[str, int] = {
            n: i for i, n in enumerate(self.global_part_names)
        }

        self.cell_settings = cell_settings

        # Stage2 wiring: Operation class + planner function
        self._operation_cls = operation_cls  # e.g. GrippingAndRemovingOperation
        self._plan_func = plan_func  # e.g. plan_gripping_and_removing

        # Optional cache for visualization scenes (for render())
        self._last_scene = None

        # Optional: mounted tool name, for C_k constraints
        self.current_tool_name: str | None = None

        # The action space will be built in build_action_space.
        self._action_space: spaces.MultiDiscrete | None = None

        self._ever_existed: set[str] = set()

        self.base_parts: set[str] = set(base_parts)

    # ----------------------------------------------------------------------
    # Capability API implementations
    # ----------------------------------------------------------------------

    def build_action_space(self, assembly: Assembly) -> spaces.Space:
        """Build the local ``MultiDiscrete([n_tools, n_parts])`` action space."""
        del assembly
        n_tools = len(self.available_grippers)
        n_parts = len(self.global_part_names)
        nvec = np.array([n_tools, n_parts], dtype=np.int64)

        self._action_space = spaces.MultiDiscrete(nvec)
        return self._action_space

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Return the built local action space."""
        if self._action_space is None:
            raise RuntimeError(
                "Action space has not been built yet. "
                "Call build_action_space(assembly) first."
            )

        return self._action_space

    def decode_action(
        self,
        assembly: Assembly,
        raw_action: np.ndarray | Sequence[int],
    ) -> GrippingAndRemovingAction:
        """Decode ``raw_action = [tool_idx, part_idx]`` into a semantic action."""
        del assembly
        tool_idx, part_idx = int(raw_action[0]), int(raw_action[1])

        if tool_idx < 0 or tool_idx >= len(self.available_grippers):
            raise IndexError(f"tool_idx {tool_idx} out of range")
        if part_idx < 0 or part_idx >= len(self.global_part_names):
            raise IndexError(f"part_idx {part_idx} out of range")

        tool_name = self.available_grippers[tool_idx]
        part_name = self.global_part_names[part_idx]

        return GrippingAndRemovingAction(
            tool_name=tool_name,
            part_name=part_name,
            gripper_id=tool_idx,
            part_index=part_idx,
            operation=None,
        )

    def get_target_ref(self, assembly: Assembly, raw_action) -> TargetRef | None:
        """Resolve a raw action to a stable environment-side target reference."""
        try:
            part_idx = int(raw_action[1])
        except Exception:
            return None
        if part_idx < 0 or part_idx >= len(self.global_part_names):
            return None

        part_name = self.global_part_names[part_idx]
        is_buffer_name = part_name.startswith(BUFFER_MILLED_PREFIX)

        getter = getattr(assembly, "get_part_by_name", None)
        exists_now = False
        if callable(getter):
            try:
                exists_now = getter(part_name) is not None
            except Exception:
                exists_now = False

        exists_now = exists_now or (part_name in getattr(assembly, "parts", {}))

        # Track ever existed (episode-local)
        if exists_now:
            self._ever_existed.add(part_name)
            # Normal case: the part exists -> it's a normal part target.
            return TargetRef(kind="part", name=part_name)

        # Pure buffer slot: reserved name that has never existed so far
        if (
            is_buffer_name
            and (not exists_now)
            and (part_name not in self._ever_existed)
        ):
            return TargetRef(kind="buffer_slot", name=part_name)

        # Part missing:
        # - If it never existed in this episode -> classify as "part_never_existed"
        # - Else it existed before -> treat as a normal part target that is now missing
        if part_name not in self._ever_existed:
            return TargetRef(kind="part_never_existed", name=part_name)

        # Otherwise treat as a real part target (including milled parts once they appear)
        return TargetRef(kind="part", name=part_name)

    def is_target_present(self, assembly: Assembly, target: TargetRef) -> bool:
        """Return whether the referenced part exists in the current assembly."""
        if target.kind == "buffer_slot":
            return True
        # if target.kind != "part" or target.name is None:
        #     return True
        if target.name is None:
            return True

        if target.kind not in {"part", "part_never_existed"}:
            return True

        getter = getattr(assembly, "get_part_by_name", None)
        if callable(getter):
            try:
                return getter(target.name) is not None
            except Exception:
                return target.name in getattr(assembly, "parts", {})

        return target.name in getattr(assembly, "parts", {})

    # ----------------------------------------------------------------------
    # Create operation from action
    # ----------------------------------------------------------------------

    def make_operation(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
    ) -> Any:
        """
        Construct a GrippingAndRemovingOperation from a symbolic GrippingAndRemovingAction.

        At this stage we only fill symbolic / topological information.
        Geometric fields such as 'direction', 'path', 'path_duration'
        will be filled by the planner (plan_gripping_and_removing) later.
        """
        del assembly

        if self._operation_cls is None:
            raise RuntimeError(
                "GrippingAndRemovingCapability has no operation_cls. "
                "Pass GrippingAndRemovingOperation via constructor to enable geometric behavior."
            )

        # Default gripping cost (can be refined later)
        default_cost = 0.0
        if self.cell_settings is not None and hasattr(
            self.cell_settings, "cost_gripping_and_removing"
        ):
            default_cost = float(self.cell_settings.cost_gripping_and_removing)

        data = {
            "type": "gripping",
            "tool_name": action.tool_name,
            "part_name": action.part_name,
            "part_index": action.part_index,
            "cost": default_cost,
            # planner will later set direction / path / path_length / path_duration
            "planned": False,
        }
        return self._operation_cls(data)

    def _ensure_operation_planned(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
    ) -> GrippingAndRemovingAction:
        """
        Ensure action.operation exists and is planned.

        The plan function (ADR-0004) returns ``op`` or ``(op, new_assembly)``; the
        removal plan core judges that output once (failure, unit direction, whether
        the part is really gone) and ``planned_operation`` translates it into
        ``op.data`` (#234).
        """
        if self._operation_cls is None or self._plan_func is None:
            # Stage1 mode: no geometric planner connected
            return action

        # 1) Get or create the operation shell
        op = action.operation or self.make_operation(assembly, action)
        data = getattr(op, "data", None)
        planned_flag = (
            bool(data.get("planned", False)) if isinstance(data, dict) else False
        )

        # 2) Call planner only if not planned yet
        if not planned_flag:
            op = planned_operation(
                op, self._plan_func(assembly, op), action.part_name, assembly
            )

        action.operation = op
        return action

    def plan_motion(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
        rng: np.random.Generator | None = None,
    ) -> GrippingAndRemovingAction:
        """
        Motion Policy π_gripping_and_removing.

        Current behavior:
          - Delegate to `_ensure_operation_planned`, which
            attaches a GrippingAndRemovingOperation to `action.operation`
            and calls the installed `plan_func` once (if provided); the
            removal plan core sets the unit direction (#234).

        TODO:
          - Fill motion-specific fields in operation.data (path, velocities, etc.).
          - Optionally use rng to sample among candidate motions.
        """
        # 1) Ensure we have a planned operation
        updated = self._ensure_operation_planned(assembly, action)
        op = getattr(updated, "operation", None)
        if op is None:
            return updated

        data = getattr(op, "data", None)
        if not isinstance(data, dict):
            return updated

        # 2) Attach a minimal motion program:
        #    core gripping path (planner output) + end->dropoff (no return here).
        # --- core duration ---
        cost_t = 0.0
        try:
            cost_t = float(data.get("cost", 0.0))
        except Exception:
            cost_t = 0.0

        path_t: float | None = None
        if "path_duration" in data:
            try:
                path_t = float(data["path_duration"])
            except Exception:
                path_t = None

        core_dur: float | None = None if path_t is None else (cost_t + path_t)

        # --- core path / anchors ---
        path = None
        anchors: dict[str, np.ndarray] = {}
        if "path" in data:
            try:
                path = np.asarray(data["path"], dtype=float)
                if path.ndim == 2 and path.shape[1] == 3 and path.shape[0] >= 2:
                    anchors["start"] = path[0].copy()
                    anchors["mid"] = (
                        path[1].copy() if path.shape[0] >= 3 else path[0].copy()
                    )
                    anchors["end"] = path[-1].copy()
                else:
                    path = None
            except Exception:
                path = None

        # --- dropoff travel time (end -> dropoff) ---
        t_to_dropoff = 0.0
        if self.cell_settings is not None and hasattr(
            self.cell_settings, "traveltime_assembly_to_dropoff"
        ):
            try:
                t_to_dropoff = float(self.cell_settings.traveltime_assembly_to_dropoff)
            except Exception:
                t_to_dropoff = 0.0

        if core_dur is not None:
            program: list[dict] = []

            core_seg = {"type": "grip_core", "duration": core_dur}
            if path is not None:
                core_seg["path"] = path
            if anchors:
                core_seg["anchors"] = anchors
            program.append(core_seg)

            if t_to_dropoff > 0.0:
                program.append(
                    {
                        "type": "transit_to_dropoff",
                        "duration": t_to_dropoff,
                        "anchor_from": anchors.get("end", None),
                        "anchor_to": "dropoff",
                    }
                )
                program.append(
                    {
                        "type": "dropoff",
                        "duration": 0.0,
                        "where": "dropoff",
                    }
                )

            data["motion_program"] = program
            data["motion_duration"] = float(
                sum(float(seg.get("duration", 0.0)) for seg in program)
            )

        # TODO: in future, compare `data["direction"]` before/after and record
        #       any large discrepancies as "geometric inconsistency" flags.

        # TODO: use `rng` here to sample between multiple candidate DOF directions
        #       if you want a stochastic motion policy.
        del rng

        return updated

    # --------------------------------------------------------------------------------------------
    # Construction aids: automatically generate N_global_parts & mappings based on the Assembly.
    # --------------------------------------------------------------------------------------------

    @classmethod
    def from_assembly(
        cls,
        assembly: Assembly,
        available_grippers: Sequence[str],
        name: str = "gripping_and_removing",
        buffer_fraction: float = 0.5,
        cell_settings: CellSettings | None = None,
        operation_cls: Any | None = None,
        plan_func: PlanFunc | None = None,
        base_parts: Sequence[str] = DEFAULT_BASE_PARTS,
        robots: Sequence[Robot] | None = None,
    ) -> "GrippingAndRemovingCapability":
        """Create a gripping_and_removing capability from an assembly.

        ``global_part_names`` contains all non-screw original parts plus a
        configurable number of buffer slots for future milled parts.
        """
        all_part_names = [p.name for p in assembly.parts.values()]

        # 1) Find all screw part names.
        screw_part_names = cls._collect_screw_part_names(assembly)

        # 2) Non-screw parts that physically exist.
        non_screw_part_names = [
            name for name in all_part_names if name not in screw_part_names
        ]

        # 3) Reserve buffer slots for future milled parts.
        total_parts = len(all_part_names)
        buffer_size = max(1, math.ceil(buffer_fraction * total_parts))
        buffer_names: list[str] = [
            f"{BUFFER_MILLED_PREFIX}{i + 1}" for i in range(buffer_size)
        ]

        global_part_names: list[str] = non_screw_part_names + buffer_names

        return cls(
            name=name,
            available_grippers=available_grippers,
            global_part_names=global_part_names,
            cell_settings=cell_settings,
            operation_cls=operation_cls,
            plan_func=plan_func,
            base_parts=base_parts,
            robots=robots,
        )

    # ----------------------------------------------------------------------
    # Helpers for accessing tools and screw/grabbable parts
    # ----------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return the available gripper names."""
        return self.available_grippers

    def _is_gripper_busy(self, cell_state: CellState | None, tool_name: str) -> bool:
        """Return True if the robot acting with ``tool_name`` already holds a part.

        This is the busy-lock: while an arm holds a part (recorded in
        ``cell_state.holds``), its further operations are blocked.
        """
        return cell_state is not None and self.robot_for(tool_name) in cell_state.holds

    @staticmethod
    def _is_target_part_held_by_other(
        cell_state: CellState | None, part_name: str
    ) -> bool:
        """Return True if ``part_name`` is currently held by ANY agent.

        Only relevant for gripping the very part that a different arm already
        grasps. Other arms may still work freely on different parts.
        """
        return cell_state is not None and part_name in cell_state.holds.values()

    @staticmethod
    def _collect_screw_part_names(assembly: Assembly) -> set[str]:
        return {
            conn.part_name1
            for conn in assembly.connections
            if isinstance(conn, Screw_Connection)
            or getattr(conn, "connection_type", None) == "screw"
        }

    # ----------------------------------------------------------------------
    # Tool compatibility (Cₖ)
    # ----------------------------------------------------------------------

    def _mounted_tool(
        self,
        current_tool_name: str | None,
        cell_state: CellState | None = None,
        robot: str | None = None,
    ) -> str | None:
        """Prefer the cell state; then the explicit arg; then the cached self.current_tool_name."""
        if cell_state is not None and robot in cell_state.mounted_tools:
            return cell_state.mounted_tools[robot]
        return (
            current_tool_name
            if current_tool_name is not None
            else self.current_tool_name
        )

    def set_current_tool(self, tool_name: str | None) -> None:
        """Update which tool is currently mounted on the end-effector."""
        self.current_tool_name = tool_name

    def _nominal_tool_axis(self, tool_name: str) -> np.ndarray | None:
        """
        Return the nominal torque axis of the given tool in the base frame.

        In a real setup this would come from:
            - robot forward kinematics
            - tool transformation in the cell
            - or a database of tool poses.

        For now we return None, meaning: "no information".
        """
        del tool_name
        # TODO: connect to real robot / tool pose model
        #  --- Optional: check alignment between tool axis and motion direction ---
        # tool_axis = self._nominal_tool_axis(action.tool_name)
        # if tool_axis is not None:
        #     tool_axis = np.asarray(tool_axis, dtype=float)
        #     if np.linalg.norm(tool_axis) > 1e-6:
        #         tool_axis = tool_axis / np.linalg.norm(tool_axis)
        #         # Try to get a direction either from operation or from DOF
        #         op = getattr(action, "operation", None)
        #         op_dir = None
        #         if op is not None:
        #             data = getattr(op, "data", {})
        #             if isinstance(data, dict) and "direction" in data:
        #                 op_dir = np.asarray(data["direction"], dtype=float)
        #         if op_dir is None:
        #             op_dir = None  # the removal plan core now owns the DOF fallback (#234)
        #         if op_dir is not None and np.linalg.norm(op_dir) > 1e-6:
        #             op_dir = op_dir / np.linalg.norm(op_dir)
        #             cos_angle = float(np.clip(np.dot(tool_axis, op_dir), -1.0, 1.0))
        #             # TODO: tune this threshold based on real cell tolerance
        #             if cos_angle < np.cos(np.deg2rad(20.0)):
        #                 # Tool is too misaligned with the required torque direction.
        #                 return False
        return None

    def tool_supports_part(self, tool_idx: int, part_name: str) -> bool:
        """
        Tool-part compatibility inside the gripping and removing domain.

        Precondition:
            part_name must be in global_part_names (non-screw or buffer).
        Current policy:
            - All such parts are supported by all grippers.

        In the future, extend this using a tool-part compatibility matrix.
        """
        del tool_idx, part_name
        return True

    # ----------------------------------------------------------------------
    # Geometric feasibility (ψ / DOF / removal direction)
    # ----------------------------------------------------------------------
    def _get_part_and_dof(
        self,
        assembly: Assembly,
        part_name: str,
    ) -> tuple[Any | None, list[np.ndarray] | None, bool]:
        """Return ``(part, dof, has_dof_api)`` for a named part."""
        part = None
        try:
            part = assembly.get_part_by_name(part_name)
        except Exception:
            part = None

        if part is None:
            return None, None, False

        get_dof = getattr(assembly, "get_part_dof", None)
        if not callable(get_dof):
            return part, None, False

        try:
            dof = get_dof(part)
        except Exception:
            dof = None

        # Normalize representation: empty -> None
        if not dof:
            return part, None, True

        return part, dof, True

    @staticmethod
    def _normalize_dirs(
        dirs: Sequence[np.ndarray], eps: float = 1e-6
    ) -> list[np.ndarray]:
        """
        Normalize direction vectors; drop near-zero vectors.
        """
        out: list[np.ndarray] = []
        for d in dirs:
            v = np.asarray(d, dtype=float).reshape(-1)
            if v.shape[0] != 3:
                continue
            n = float(np.linalg.norm(v))
            if n > eps:
                out.append(v / n)
        return out

    def predecode_mask_reason(
        self,
        assembly: Assembly,
        raw_action: np.ndarray,
        current_tool_name: str | None = None,
        cell_state: CellState | None = None,
    ) -> tuple[bool, str | None, dict]:
        """
        Capability-specific pre-decode masking.
        Reasons:
          - base_part
          - no_robot
          - agent_busy
          - wrong_tool
          - geometrically_infeasible
        """
        try:
            tool_idx, part_idx = int(raw_action[0]), int(raw_action[1])
        except Exception:
            return False, None, {"mask_error": "bad_raw_action"}

        if not (0 <= tool_idx < len(self.available_grippers)):
            return False, None, {"mask_error": "tool_idx_out_of_range"}
        if not (0 <= part_idx < len(self.global_part_names)):
            return False, None, {"mask_error": "part_idx_out_of_range"}

        tool_name = self.available_grippers[tool_idx]
        part_name = self.global_part_names[part_idx]

        # --- base_part policy (EMO V2) ---
        if part_name in self.base_parts:
            return True, "base_part", {"part": part_name}

        # --- no_robot: no core robot of the cell grips and removes with this tool ---
        robot = self.robot_for(tool_name)
        if robot is None:
            return True, "no_robot", {"tool": tool_name, "part": part_name}

        # --- Busy-lock (agent_busy) ---
        # 1) Own arm already holds a part -> all further gripping blocked.
        if self._is_gripper_busy(cell_state, tool_name):
            return True, "agent_busy", {"agent": robot, "part": part_name}

        # 2) Target part is already grasped by (any) other arm -> cannot grasp it.
        if self._is_target_part_held_by_other(cell_state, part_name):
            return (
                True,
                "agent_busy",
                {"agent": robot, "part": part_name, "why": "part_held_by_other"},
            )

        # --- C_k (wrong_tool) ---
        # Only enforce when a mounted tool is known (from arg or cache).
        mounted = self._mounted_tool(current_tool_name, cell_state, robot)
        if mounted is not None and mounted != tool_name:
            return True, "wrong_tool", {"mounted": mounted, "required": tool_name}

        # --- ψ cheap gate (geometrically_infeasible) ---
        part, dof, has_dof_api = self._get_part_and_dof(assembly, part_name)
        if part is None:
            # Normally env catches target_missing, but keep it safe.
            return True, "geometrically_infeasible", {"why": "part_missing"}

        # If DOF API exists and returned empty -> infeasible
        if has_dof_api and not dof:
            return True, "geometrically_infeasible", {"why": "empty_dof"}

        check_vis = getattr(assembly, "check_visibility", None)
        if callable(check_vis) and dof:
            view_dirs = self._normalize_dirs(dof)
            if not view_dirs:
                return True, "geometrically_infeasible", {"why": "no_view_dirs"}
            try:
                with suppress_stdout():
                    ok = bool(check_vis(part, view_dirs))
            except Exception:
                ok = False
            if not ok:
                return True, "geometrically_infeasible", {"why": "not_visible"}

        return False, None, {}

    # ----------------------------------------------------------------------
    # Deterministic transition
    # ----------------------------------------------------------------------

    def deterministic_transition(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
    ) -> Assembly:
        """
        Success-conditional state update for gripping_and_removing.
        Caller must only call this after the orchestrator has classified the
        attempt as successful.  The returned assembly may still be unchanged.

        The operation's ``result_assembly`` is the removal plan core's result,
        set only when the part is really gone (#234). Without it, the state is
        left unchanged.

        The release of the arm after a successful removal is cell state, see
        ``cell_state_transition``.
        """
        op = getattr(action, "operation", None)

        out: Assembly | None = None

        # We always work on copies unless planner already produced an Assembly.
        # This avoids mutating the input assembly accidentally.
        if op is not None:
            # Cache scene for visualization (optional)
            try:
                self._update_scene_cache(assembly, action)
            except Exception:
                pass

            data = getattr(op, "data", None)
            if isinstance(data, dict) and isinstance(
                data.get("result_assembly"), Assembly
            ):
                # Keep the reference identity of the planner result (existing
                # tests assert ``out is result_assembly``).
                out = data["result_assembly"]

        if out is None:
            # Strict: if nothing removed the target, do not change state
            out = copy.deepcopy(assembly)

        return out

    def cell_state_transition(
        self,
        cell_state: CellState,
        action: GrippingAndRemovingAction,
        next_assembly: Assembly,
    ) -> CellState:
        """Busy-lock release: a successful removal frees the arm.

        Only when the target part was actually removed from ``next_assembly``,
        as the removal plan core judges it.
        """
        if not part_removed(next_assembly, action.part_name):
            return cell_state
        robot = self.robot_for(action.tool_name)
        return dataclasses.replace(
            cell_state, holds={r: p for r, p in cell_state.holds.items() if r != robot}
        )

    # ----------------------------------------------------------------------
    # Execution success factor
    # ----------------------------------------------------------------------

    def execution_success_factor(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
    ) -> float:
        """
        Tool/capability-level execution success factor in [0,1].

        Semantics:
          - Models non-EOL execution uncertainty, e.g. trajectory not reached,
            force-control not in contact, tool not engaged.
          - Must NOT depend on EOL belief / stuck latent variables.

        Default:
          - Return 1.0 if you don't want any extra execution stochasticity.
        """
        del assembly, action
        return float(np.clip(1.0, 0.0, 1.0))

    # ----------------------------------------------------------------------
    # Time estimation
    # ----------------------------------------------------------------------

    def estimate_action_time(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
        settings: CellSettings,
        # current_tool_name: str | None,
        tool_before: str | None = None,
    ) -> float:
        """
        Estimate time from *action start* onward.

        IMPORTANT:
          - Toolchange / start-pose travel is handled by the ENV (prefix_time).
          - This method must NOT add any toolchange time.
          - It may include "end -> dropoff" if your motion_program includes it.
        """
        del assembly, tool_before

        # ----- Use planner-provided motion duration if available -----
        op = getattr(action, "operation", None)
        if op is not None:
            data = getattr(op, "data", None)
            if isinstance(data, dict) and "motion_duration" in data:
                try:
                    return float(data["motion_duration"])
                except Exception:
                    pass

                # if base is not None:
                #    t_change = 0.0
                #    if current_tool_name is not None and current_tool_name != action.tool_name:
                #        t_change = (
                #            settings.toolchange_time_assembly_to_storage
                #            + settings.toolchange_time_at_toolstorage
                #            + settings.toolchange_time_dropoff_to_storage
                #        )
                #    return base + t_change

        # ----- Symbolic fallback (symbolic approximation) -----
        def pick(primary: str, fallback: str) -> float:
            if hasattr(settings, primary):
                return getattr(settings, primary)
            if hasattr(settings, fallback):
                return getattr(settings, fallback)
            raise AttributeError(
                f"Neither '{primary}' nor '{fallback}' is defined in settings."
            )

        # 1) Distances
        approach_len = pick(
            "gripping_and_removing_start_offset", "gripping_start_offset"
        )
        minimal_pullout = pick(
            "gripping_and_removing_minimal_pullout_value",
            "gripping_minimal_pullout_value",
        )
        pullout_factor = pick(
            "gripping_and_removing_pullout_factor", "gripping_pullout_factor"
        )

        pullout_len = max(
            minimal_pullout,
            pullout_factor * approach_len,
        )

        # 2) Feed rates
        feed_rate_approach = pick(
            "feed_rate_gripping_and_removing_approaching",
            "feed_rate_gripping_and_removing_approaching",
        )
        feed_rate_leave = pick(
            "feed_rate_gripping_and_removing_leaving",
            "feed_rate_gripping_and_removing_leaving",
        )

        t_approach = approach_len / feed_rate_approach
        t_leave = pullout_len / feed_rate_leave

        # 3) End -> dropoff (only if your policy always goes to dropoff after success)
        # t_dropoff = float(getattr(settings, "traveltime_assembly_to_dropoff", 0.0))

        ## 4) Tool change time
        # t_change = 0.0
        # if current_tool_name is not None and current_tool_name != action.tool_name:
        #    t_change = (
        #        settings.toolchange_time_assembly_to_storage
        #        + settings.toolchange_time_at_toolstorage
        #        + settings.toolchange_time_dropoff_to_storage
        #    )

        # t_dropoff = 0.0
        # if hasattr(settings, "traveltime_assembly_to_dropoff"):
        #    t_dropoff = float(settings.traveltime_assembly_to_dropoff)

        return (
            float(getattr(settings, "cost_gripping_and_removing", 0.0))
            + t_approach
            + t_leave
        )

    # ----------------------------------------------------------------------
    # Scene
    # ----------------------------------------------------------------------

    def _update_scene_cache(
        self,
        assembly: Assembly,
        action: GrippingAndRemovingAction,
    ) -> None:
        """
        Compute and cache a o3d.Scene for visualization.

        - If the underlying operation exposes a `.scene(assembly)` method,
          we call it and store the result in:
              * self._last_scene
              * action.operation.data["scene"]  (if data dict exists)

        - If no scene is available or an error occurs, we fall back to None.
        """
        self._last_scene = None

        op = getattr(action, "operation", None)
        if op is None:
            return

        scene = None
        if hasattr(op, "scene_o3d"):
            try:
                scene = op.scene_o3d(assembly)
            except Exception:
                # Robust against incomplete geometry or dummy operations
                scene = None

        self._last_scene = scene

        data = getattr(op, "data", None)
        if isinstance(data, dict) and scene is not None:
            data["scene"] = scene

    def get_last_scene(self):
        """
        Optional helper for visualization:
        return the cached trimesh.Scene (or None if not available).
        """
        return self._last_scene
