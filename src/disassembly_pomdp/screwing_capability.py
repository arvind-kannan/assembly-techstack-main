"""Screwing capability implementation for the disassembly POMDP environment."""

from __future__ import annotations

import copy
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

# The capability deliberately keeps broad guard rails around optional geometry,
# visualization, and legacy Assembly APIs.
# pylint: disable=too-many-instance-attributes,too-many-public-methods
# pylint: disable=too-many-locals,too-many-return-statements,broad-exception-caught
# pylint: disable=invalid-name


# Planner signature: (Assembly, Operation) -> Operation
PlanFunc = Callable[[Assembly, Any], Any]


@dataclass
class ScrewingAction:
    """Semantic action: use a screwdriver to operate on a particular screw connection."""

    tool_name: str
    connection: Screw_Connection | None
    screw_part_name: str
    base_part_name: str
    operation: Any | None = None


class ScrewingCapability(Capability):
    """
    Capability for removing screws with a screwdriver.

    Action is encoded as:
        raw_action = [tool_idx, conn_idx]    (MultiDiscrete)
    where:
        tool_idx  ∈ [0, n_tools)
        conn_idx ∈ [0, n_screw_connections)
    """

    process = Process.UNSCREWING

    def __init__(
        self,
        available_screwdrivers: Sequence[str],
        name: str = "screwing",
        cell_settings: CellSettings | None = None,
        operation_cls: Any | None = None,
        plan_func: PlanFunc | None = None,
        robots: Sequence[Robot] | None = None,
    ):
        super().__init__(name=name, robots=robots)
        self.available_screwdrivers: list[str] = list(available_screwdrivers)

        # Mapping: tool_name → set of screw heads it supports.
        # For now, this is a trivial one-to-one mapping.
        self.tool_to_heads = {name: {name} for name in self.available_screwdrivers}

        self.cell_settings = cell_settings

        # Wiring: Operation class + planner function
        self._operation_cls = operation_cls
        self._plan_func = plan_func

        # Optional cache for visualization scenes (for render())
        self._last_scene = None

        # Mounted tool name, for C_k constraints
        self.current_tool_name: str | None = None

        self._action_space: spaces.MultiDiscrete | None = None

        # Global, fixed screw-connection keys built from the base assembly
        self._global_screw_conn_keys: list[tuple[str, str, str | None]] = []

        self._ever_existed: set[str] = set()

    # ----------------------------------------------------------------------
    # Capability API implementations
    # ----------------------------------------------------------------------

    def build_action_space(self, assembly: Assembly) -> spaces.Space:
        """
        Returns MultiDiscrete([n_tools, n_screw_connections]).
        """
        n_tools = len(self.available_screwdrivers)

        # Build fixed global screw-connection index from the *given* assembly (base assembly)
        if not self._global_screw_conn_keys:
            conns = self.list_screw_connections(assembly)
            keys = [self._conn_key(c) for c in conns]
            self._global_screw_conn_keys = sorted(keys)

        n_conns = len(self._global_screw_conn_keys)
        nvec = np.array([n_tools, n_conns], dtype=np.int64)

        self._action_space = spaces.MultiDiscrete(nvec)
        return self._action_space

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        if self._action_space is None:
            raise RuntimeError(
                "Action space has not been built yet. "
                "Call build_action_space(assembly) first."
            )
        return self._action_space

    def decode_action(self, assembly: Assembly, raw_action) -> ScrewingAction:
        tool_idx = int(raw_action[0])
        conn_idx = int(raw_action[1])

        if tool_idx >= len(self.available_screwdrivers) or tool_idx < 0:
            raise IndexError(f"tool_idx {tool_idx} out of range")

        if conn_idx < 0 or conn_idx >= len(self._global_screw_conn_keys):
            raise IndexError(f"conn_idx {conn_idx} out of range (global index)")

        tool_name = self.available_screwdrivers[tool_idx]
        key = self._global_screw_conn_keys[conn_idx]

        conn = self._find_conn_by_key(assembly, key)
        if conn is None:
            screw, base, _head = key
            return ScrewingAction(
                tool_name=tool_name,
                connection=None,
                screw_part_name=screw,
                base_part_name=base,
                operation=None,
            )

        screw_name, base_name = self._get_screw_and_base_names(conn)

        return ScrewingAction(
            tool_name=tool_name,
            connection=conn,
            screw_part_name=screw_name,
            base_part_name=base_name,
            operation=None,
        )

    def get_target_ref(self, assembly: Assembly, raw_action) -> TargetRef | None:
        try:
            conn_idx = int(raw_action[1])
        except Exception:
            return None
        if conn_idx < 0 or conn_idx >= len(self._global_screw_conn_keys):
            return None

        screw, _base, _head = self._global_screw_conn_keys[conn_idx]

        # We track presence of the screw PART (not the connection).
        getter = getattr(assembly, "get_part_by_name", None)
        exists_now = False
        if callable(getter):
            try:
                exists_now = getter(screw) is not None
            except Exception:
                exists_now = False

        exists_now = exists_now or (screw in getattr(assembly, "parts", {}))

        if exists_now:
            self._ever_existed.add(screw)
            # Normal case: the screw exists -> it's a normal part target.
            return TargetRef(kind="part", name=screw)

        # Screw missing:
        # - If it never existed in this episode -> classify as "part_never_existed"
        # - Else it existed before -> treat as a normal part target that is now missing
        if screw not in self._ever_existed:
            return TargetRef(kind="part_never_existed", name=screw)

        return TargetRef(kind="part", name=screw)

    def is_target_present(self, assembly: Assembly, target: TargetRef) -> bool:
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
        action: ScrewingAction,
    ) -> Any:
        """
        Construct a ScrewingOperation from a symbolic ScrewingAction.

        At this stage we only fill symbolic / topological information.
        Geometric fields such as 'direction', 'path', 'path_duration'
        will be filled by the planner (plan_screwing) later.
        """
        if self._operation_cls is None:
            raise RuntimeError(
                "ScrewingCapability has no operation_cls. "
                "Pass ScrewingOperation via constructor to enable geometric behavior."
            )

        default_cost = 0.0
        if self.cell_settings is not None and hasattr(
            self.cell_settings, "cost_screwing"
        ):
            default_cost = float(self.cell_settings.cost_screwing)

        data = {
            "type": "screwing",
            "tool_name": action.tool_name,
            "screw_part_name": action.screw_part_name,
            "base_part_name": action.base_part_name,
            "connection": action.connection,
            "cost": default_cost,
            # planner will later set direction / path / path_length / path_duration
            "planned": False,
        }
        return self._operation_cls(data)

    def _ensure_operation_planned(
        self,
        assembly: Assembly,
        action: ScrewingAction,
    ) -> ScrewingAction:
        """
        Ensure action.operation exists and is planned.

        The plan function (ADR-0004) returns ``op`` or ``(op, new_assembly)``; the
        removal plan core judges that output once (failure, unit direction, whether
        the screw is really gone) and ``planned_operation`` translates it into
        ``op.data`` (#234).
        """
        if self._operation_cls is None or self._plan_func is None:
            # Symbolic fallback mode: no geometric planner connected
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
                op, self._plan_func(assembly, op), action.screw_part_name, assembly
            )

        action.operation = op
        return action

    def plan_motion(
        self,
        assembly: Assembly,
        action: ScrewingAction,
        rng: np.random.Generator | None = None,
    ) -> ScrewingAction:
        """
        Motion Policy π_screw.

        Current behavior:
            - Delegate to `_ensure_operation_planned`, which
              attaches a ScrewingOperation to `action.operation`
              and calls the installed `plan_func` once (if provided).
              The removal plan core sets the unit direction (#234).

        TODO:
          - Fill motion-specific fields in operation.data (path, torque, velocities, etc.).
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
        #    core screwing path (planner output) + end->dropoff (no return here).
        #    This supports cross-step execution:
        #      step t ends at dropoff
        #      step t+1 begins with dropoff -> next start (handled by env)
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

        # Add screw search variant: 5–10 s
        search_t = float(rng.uniform(5.0, 10.0)) if rng is not None else 10.0
        data["screw_search_duration"] = search_t

        core_dur: float | None = (
            None if path_t is None else (cost_t + search_t + path_t)
        )
        # core_dur: float | None = None if path_t is None else (cost_t + path_t)

        # --- core path / anchors ---
        path = None
        anchors: dict[str, np.ndarray] = {}
        if "path" in data:
            try:
                path = np.asarray(data["path"], dtype=float)
                if path.ndim == 2 and path.shape[1] == 3 and path.shape[0] >= 2:
                    anchors["start"] = path[0].copy()
                    anchors["mid"] = path[1].copy()
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

            core_seg = {"type": "screw_core", "duration": core_dur}
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

        return updated

    # ----------------------------------------------------------------------
    # Helpers for accessing tools and screw connections
    # ----------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        return self.available_screwdrivers

    def _is_tool_busy(self, cell_state: CellState | None, tool_name: str) -> bool:
        """Return True if the robot acting with ``tool_name`` already holds a part.

        This references the same ``cell_state.holds`` busy-lock that the gripping
        capabilities write to, so while an arm holds a part its screwing actions
        are blocked too (single-arm mutual exclusion).
        """
        return cell_state is not None and self.robot_for(tool_name) in cell_state.holds

    def _find_conn_by_key(
        self, assembly: Assembly, key: tuple[str, str, str | None]
    ) -> Screw_Connection | None:
        for c in self.list_screw_connections(assembly):
            if self._conn_key(c) == key:
                return c
        return None

    @staticmethod
    def list_screw_connections(assembly: Assembly) -> list[Screw_Connection]:
        """Return all screw connections inside the assembly."""
        return [
            conn for conn in assembly.connections if isinstance(conn, Screw_Connection)
        ]

    @staticmethod
    def _get_screw_and_base_names(conn: Screw_Connection) -> tuple[str, str]:
        """
        Infer (screw_part_name, base_part_name) from a Screw_Connection.
        If exactly one side is named like a screw, that one is the screw part.

        This inference keeps symbolic semantics intact.
        """
        n1, n2 = conn.part_name1, conn.part_name2
        l1, l2 = n1.lower(), n2.lower()

        if "screw" in l1 and "screw" not in l2:
            return n1, n2
        if "screw" in l2 and "screw" not in l1:
            return n2, n1

        # Fallback: treat part_name1 as screw (this keeps legacy behavior stable)
        return n1, n2

    @classmethod
    def _conn_key(cls, conn: Screw_Connection) -> tuple[str, str, str | None]:
        head = getattr(conn, "screw_head", None)
        screw, base = cls._get_screw_and_base_names(conn)
        return (screw, base, head)

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
        """
        Update which tool is currently mounted on the end-effector.

        - tool_name should be something like 'H5', 'T6', etc.
        """
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
        #    tool_axis = np.asarray(tool_axis, dtype=float)
        #    if np.linalg.norm(tool_axis) > 1e-6:
        #        tool_axis = tool_axis / np.linalg.norm(tool_axis)
        #        # Try to get a direction either from operation or from DOF
        #        op = getattr(action, "operation", None)
        #        op_dir = None
        #        if op is not None:
        #            data = getattr(op, "data", {})
        #            if isinstance(data, dict) and "direction" in data:
        #                op_dir = np.asarray(data["direction"], dtype=float)
        #        if op_dir is None:
        #            op_dir = None  # the removal plan core now owns the DOF fallback (#234)
        #        if op_dir is not None and np.linalg.norm(op_dir) > 1e-6:
        #            op_dir = op_dir / np.linalg.norm(op_dir)
        #            cos_angle = float(np.clip(np.dot(tool_axis, op_dir), -1.0, 1.0))
        #            # TODO: tune this threshold based on real cell tolerance
        #            if cos_angle < np.cos(np.deg2rad(20.0)):
        #                # Tool is too misaligned with the required torque direction.
        #                return False
        return None

    def tool_supports_connection(self, tool_idx: int, conn: Screw_Connection) -> bool:
        """Return True if the screwdriver at tool_idx supports the screw head type."""
        tool_name = self.available_screwdrivers[tool_idx]
        supported = self.tool_to_heads.get(tool_name, set())
        head = getattr(conn, "screw_head", None)
        return head in supported

    def compatible_tool_indices(self, conn: Screw_Connection) -> list[int]:
        """Return all tool indices capable of operating on this screw head."""
        head = getattr(conn, "screw_head", None)
        if head is None:
            return []

        result = []
        for idx, tool in enumerate(self.available_screwdrivers):
            if head in self.tool_to_heads.get(tool, set()):
                result.append(idx)
        return result

    # ----------------------------------------------------------------------
    # Geometric feasibility (ψ / DOF / removal direction)
    # ----------------------------------------------------------------------

    def _get_part_and_dof(
        self,
        assembly: Assembly,
        part_name: str,
    ) -> tuple[Any | None, list[np.ndarray] | None, bool]:
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
          - no_robot
          - agent_busy
          - wrong_tool
          - geometrically_infeasible
        """
        try:
            tool_idx, conn_idx = int(raw_action[0]), int(raw_action[1])
        except Exception:
            return False, None, {"mask_error": "bad_raw_action"}

        if not (0 <= tool_idx < len(self.available_screwdrivers)):
            return False, None, {"mask_error": "tool_idx_out_of_range"}
        if not (0 <= conn_idx < len(self._global_screw_conn_keys)):
            return False, None, {"mask_error": "conn_idx_out_of_range"}

        tool_name = self.available_screwdrivers[tool_idx]
        screw, _base, head = self._global_screw_conn_keys[conn_idx]

        # --- no_robot: no core robot of the cell unscrews with this tool ---
        robot = self.robot_for(tool_name)
        if robot is None:
            return True, "no_robot", {"tool": tool_name, "screw": screw}

        # --- Busy-lock (agent_busy) ---
        # Own arm already holds a part (set by a previous successful gripping,
        # stored in ``cell_state.holds``) -> all further operations of this arm
        # are blocked until the part is released.
        if self._is_tool_busy(cell_state, tool_name):
            return True, "agent_busy", {"agent": robot, "screw": screw}

        # --- C_k (wrong_tool + head mismatch) ---
        # Only enforce when a mounted tool is known (from arg or cache).
        mounted = self._mounted_tool(current_tool_name, cell_state, robot)
        if mounted is not None and mounted != tool_name:
            return True, "wrong_tool", {"mounted": mounted, "required": tool_name}

        # head type compatibility (fast: from global key)
        if head is None:
            # Strict policy: no head info => can't choose a valid tool
            return True, "wrong_tool", {"why": "missing_head", "tool": tool_name}

        supported_heads = self.tool_to_heads.get(tool_name, set())
        if head not in supported_heads:
            return (
                True,
                "wrong_tool",
                {"why": "head_mismatch", "tool": tool_name, "head": head},
            )

        # --- ψ cheap gate (geometrically_infeasible) ---
        part, dof, has_dof_api = self._get_part_and_dof(assembly, screw)
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
        self, assembly: Assembly, action: ScrewingAction
    ) -> Assembly:
        """
        Success-conditional state update for screwing.
        Caller must only call this when the attempt is considered successful by the orchestrator.

          1) If the operation carries a result_assembly (the removal plan core's
             result, set only when the screw is really gone, #234), return it.
          2) Else NO STATE CHANGE.
        """
        op = getattr(action, "operation", None)

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
                return data["result_assembly"]

        # Strict: if nothing removed the target, do not change state
        out = copy.deepcopy(assembly)

        # Could log that success was sampled but no result_assembly was provided
        return out

    # ----------------------------------------------------------------------
    # Execution success factor
    # ----------------------------------------------------------------------

    def execution_success_factor(
        self,
        assembly: Assembly,
        action: ScrewingAction,
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
        action: ScrewingAction,
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

        # --- Use planner-provided motion duration if available ---
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

        # --- Symbolic fallback (action-start -> dropoff) ---
        L_approach = settings.screwing_start_offset
        pullout_len = max(
            settings.screwing_minimal_pullout_value,
            settings.screwing_pullout_factor * settings.screwing_start_offset,
        )
        t_approach = L_approach / settings.feed_rate_screwing_approaching
        t_leave = pullout_len / settings.feed_rate_screwing_leaving
        t_search = 10.0

        # If your policy always ends at dropoff after screwing, keep this.
        # t_dropoff = float(getattr(settings, "traveltime_assembly_to_dropoff", 0.0))

        return (
            float(getattr(settings, "cost_screwing", 0.0))
            + t_search
            + t_approach
            + t_leave
        )

    # ----------------------------------------------------------------------
    # Scene
    # ----------------------------------------------------------------------

    def _update_scene_cache(
        self,
        assembly: Assembly,
        action: ScrewingAction,
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
