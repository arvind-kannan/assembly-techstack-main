"""Gymnasium-compatible POMDP environment for robotic disassembly.

The environment coordinates capability-conditioned actions, end-of-life
sampling, stochastic transition handling, observer feedback, and reward
computation behind the standard Gymnasium reset-step interface.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Fixed_Connection
from diss_assembly_dataclasses.process import Process
from gymnasium import spaces

from disassembly_pomdp.capability_base import Capability
from disassembly_pomdp.config import CellSettings
from disassembly_pomdp.eol_operator import (
    EOLUnifiedSampler,
    InspectionPriorSampler,
    PartEOLOperatorV2,
)
from disassembly_pomdp.pomdp_observer import ObservationEncoder, Observer
from disassembly_pomdp.stuck_part_rules import StuckPartRules
from motion_planning.removal_planning import removal_plan_of


def _has_fixed(assembly: Assembly, part_a: str, part_b: str) -> bool:
    """Return whether the assembly already contains a fixed connection."""
    for connection in assembly.connections:
        if getattr(connection, "connection_type", None) != "fixed":
            continue

        first_part = getattr(connection, "part_name1", None)
        second_part = getattr(connection, "part_name2", None)
        if (first_part == part_a and second_part == part_b) or (
            first_part == part_b and second_part == part_a
        ):
            return True
    return False


def _add_fixed_if_missing(assembly: Assembly, part_a: str, part_b: str) -> None:
    """Add a fixed connection unless it already exists."""
    if not _has_fixed(assembly, part_a, part_b):
        assembly.add_connection(Fixed_Connection(part_a, part_b))


def _remove_part_and_connections_inplace(asm: Assembly, part_name: str) -> bool:
    """
    Remove a part from asm, and drop any connections that reference it.
    Returns True if something changed.
    Robust against different Assembly implementations.
    """
    changed = False

    # --- remove part ---
    getter = getattr(asm, "get_part_by_name", None)
    existed = False
    if callable(getter):
        try:
            existed = getter(part_name) is not None
        except Exception:
            existed = part_name in getattr(asm, "parts", {})

    if part_name in getattr(asm, "parts", {}):
        existed = True

    if existed:
        # Prefer official API if present
        rm = getattr(asm, "remove_part", None)
        if callable(rm):
            try:
                part_obj = getattr(asm, "get_part_by_name", lambda _: None)(part_name)
                if part_obj is not None:
                    rm(part_obj)
                    changed = True
            except Exception:
                pass

        if part_name in getattr(asm, "parts", {}):
            try:
                del asm.parts[part_name]
                changed = True
            except Exception:
                pass

    # --- remove connections referencing this part ---
    conns = getattr(asm, "connections", None)
    if isinstance(conns, list) and conns:
        new_conns = []
        dropped = 0
        for c in conns:
            p1 = getattr(c, "part_name1", None)
            p2 = getattr(c, "part_name2", None)
            if p1 == part_name or p2 == part_name:
                dropped += 1
            else:
                new_conns.append(c)
        if dropped > 0:
            asm.connections = new_conns
            changed = True

    # --- clear any cached collision objects if your Assembly uses them ---
    if hasattr(asm, "_fcl_objects"):
        try:
            asm._fcl_objects = {}
        except Exception:
            pass
    if hasattr(asm, "_fcl_cache"):
        try:
            asm._fcl_cache = None
        except Exception:
            pass

    return changed


def inject_eol_stuck_fixed_connections(
    assembly: Assembly,
    stuck_parts: set[str],
    rules: StuckPartRules,
) -> None:
    """Inject representative stuck-part effects as fixed connections, as the product's rules say."""
    for part_name in stuck_parts:
        rule = rules.rule_for(part_name)
        if rule is None:
            continue
        for other in rule.fixed_to:
            _add_fixed_if_missing(assembly, part_name, other)


class DisassemblyPOMDPEnv(gym.Env):
    """
    Disassembly POMDP Environment

    This environment models automated disassembly as a POMDP with
    capability-conditioned actions, deterministic_transition, and
    explicit action feedback observations.

    Env handles stochastic sampling / EOL update

    Core characteristics:
    ---------------------
    - State:
        The underlying system state is represented by an Assembly object
        describing parts, connections, and geometry.

    - Action space:
        Global action space is a padded MultiDiscrete:
            MultiDiscrete([n_capabilities, max_dim0, max_dim1])

        Each capability k defines its own local action space:
            MultiDiscrete([dim0_k, dim1_k])

        Actions outside a capability's valid local range are explicitly rejected.

    - Action semantics:
        Actions are interpreted in two stages:
            1) Pre-decode validation (range, mask, target existence)
            2) Capability-specific decoding and deterministic execution

        Invalid actions are rejected *before decoding* and do not alter
        the physical state, but still consume penalty and produce feedback.

    - Transition model:
        Each capability implements a deterministic transition model:
            T_k(s, a) → (s', success_flag)

        Deterministic transitions are used internally and only accepted
        if they produce the intended physical effect (e.g. part removal).

    - Reward model:
        Rewards are negative execution costs:
            reward = -(execution_time + penalty)

        Optional penalties:
            - Milling penalty for destructive actions
            - Truncation penalty if max_steps is exceeded

    - Observations:
        Observations are encoded by ObservationEncoder and currently contain:
            {
                "parts":    Binary detected-parts vector,
                "tool":     One-hot encoding of the currently mounted tool,
                "accepted": Whether the previous action was accepted by env predecode
            }
        - Info (logging / analysis channel):
        info may include EOL-related diagnostic signals that are NOT exposed to the agent, e.g.:
            - "eol_p_success_ideal_init": per-part idealized success probabilities at reset
            - "eol_p_success_ideal": per-part idealized success probabilities after each step
        Here "ideal" means p_exec=1.0 (tool/geometry/planning gates assumed to pass).

        The action feedback observation (O_a) explicitly informs the agent
        whether the last action was accepted or rejected, and for what reason.

    - Termination:
        An episode terminates when all specified target parts have been
        removed from the assembly, or when max_steps is reached.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        assembly: Assembly,
        capabilities: Sequence[Capability],
        cell_settings: CellSettings,
        milling_penalty: float = 0.0,
        target_part_names: Optional[Sequence[str]] = None,
        max_steps: int = 50,
        truncation_penalty: float = 200.0,
        rejected_penalty: float = 200.0,
        noop_penalty: float = 100.0,
        # observer: FullObservableObserver | None = None,
        belief_mode: str = "pomdp",  # "fo" or "pomdp"
        inspection_prior_sampler: Optional[InspectionPriorSampler] = None,
        world_sampler: EOLUnifiedSampler | None = None,
        eol_op: PartEOLOperatorV2 | None = None,
        expose_eol_belief: bool = False,
        observer: Optional[Observer] = None,
        initial_mounted_tool: str | None = None,
    ):
        """
        Args:
            expose_eol_belief:
                If True, include EOL belief features in observation under key "eol".
                Currently only exposes p_stuck per part.
                If False, the observation does not contain any EOL fields.
                Note: idealized success probabilities are still logged in `info`
                as "eol_p_success_ideal_*" if an EOL operator is available.
            initial_mounted_tool:
                Tool mounted at the start of each episode. This must be configured
                explicitly by the concrete robot-cell or experiment setup.
        """
        super().__init__()

        # Base assembly used for reset; working copy in self.assembly
        self.base_assembly: Assembly = copy.deepcopy(assembly)
        self.assembly: Assembly = copy.deepcopy(assembly)

        # Part-existence observation is defined over the initial parts
        self._obs_part_names: list[str] = [
            p.name for p in self.base_assembly.parts.values()
        ]

        self.capabilities: list[Capability] = list(capabilities)
        self.cell_settings = cell_settings
        self.milling_penalty = float(milling_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.rejected_penalty = float(rejected_penalty)
        self.noop_penalty = float(noop_penalty)

        # Task / episode-level settings
        self.target_part_names = (
            list(target_part_names) if target_part_names is not None else []
        )
        self._target_part_set = set(self.target_part_names)

        self.max_steps = int(max_steps)
        self.step_count = 0

        self.belief_mode = str(belief_mode).lower()
        if self.belief_mode not in {"fo", "pomdp"}:
            raise ValueError("belief_mode must be 'fo' or 'pomdp'")

        self.inspection_prior_sampler = inspection_prior_sampler
        self.world_sampler = world_sampler
        self._eol_op = eol_op
        self.expose_eol_belief = bool(expose_eol_belief)

        # Initial tool configuration belongs to the concrete robot cell.
        if not isinstance(initial_mounted_tool, str) or not initial_mounted_tool:
            raise ValueError(
                "initial_mounted_tool must be explicitly configured for the robot cell."
            )

        self.initial_mounted_tool: str = initial_mounted_tool

        # Cell state (robot holds, mounted tools) lives next to the assembly,
        # not in it. Restored on every reset().
        self.cell_state: CellState = CellState()

        # Runtime tool state in the env (synchronized to all capabilities and
        # written to the cell state). It is restored from initial_mounted_tool
        # on every reset().
        self.current_mounted_tool: Optional[str] = None

        # Cross-step robot anchor: if True, robot is assumed parked at dropoff after last step.
        self._robot_at_dropoff: bool = False

        # Optional cached scene for geometric render()
        self._last_scene = None
        self.last_info: Optional[dict] = None

        # Build action space per capability, then global action space
        self._build_action_space()

        # --- action feedback (O_a) ---
        self._feedback_reason_names = [
            "none",
            "out_of_range",
            "masked_out",
            "toolchange_override",
        ]

        self._last_feedback = {
            "accepted": 1,
            "reason": 0,
            "last_action": np.zeros(3, dtype=np.int64),
        }

        # if observer is None:
        # else:
        #    self.observer = observer

        # self.observation_space = self.observer.build_observation_space(self)
        self._last_accepted = 1

        self.encoder = ObservationEncoder()
        self.observation_space = self.encoder.build_observation_space(self)

        self.observer = observer
        if self.belief_mode == "pomdp" and observer is None:
            raise ValueError("POMDP mode requires an observer.")

        self._detected_parts: set[str] = set()

        self._last_detected_parts: Optional[set[str]] = None
        self._last_measurement_debug: Optional[dict] = None

    def _build_action_space(self) -> None:
        """
        Call `build_action_space(base_assembly)` for each capability,
        and then unify them into a global `MultiDiscrete`:
            [n_capabilities, max_dim0, max_dim1]

        Assumes each capability uses `MultiDiscrete([*, *])` as local space.
        """
        self._cap_action_spaces: list[spaces.MultiDiscrete] = []

        max_dim0 = 0  # Maximum number of tools
        max_dim1 = 0  # Maximum target object number / number of operations

        for cap in self.capabilities:
            subspace = cap.build_action_space(self.base_assembly)
            if not isinstance(subspace, spaces.MultiDiscrete):
                raise TypeError(
                    f"Capability {cap.name} must use MultiDiscrete action spaces; "
                    f"got {type(subspace)}"
                )
            if subspace.nvec.shape != (2,):
                raise ValueError(
                    f"Capability {cap.name} action space must be 2D MultiDiscrete([*, *]); "
                    f"got nvec={subspace.nvec}"
                )

            self._cap_action_spaces.append(subspace)
            max_dim0 = max(max_dim0, int(subspace.nvec[0]))
            max_dim1 = max(max_dim1, int(subspace.nvec[1]))

        n_caps = len(self.capabilities)
        self.action_space = spaces.MultiDiscrete(
            np.array([n_caps, max_dim0, max_dim1], dtype=np.int64)
        )

    def _observe(
        self,
        *,
        assembly: Assembly,
        stage: str,
    ) -> tuple[Optional[set[str]], Optional[dict]]:
        """Run the configured observer in POMDP mode."""
        if self.belief_mode != "pomdp" or self.observer is None:
            return None, None
        measurement = self.observer.observe(assembly=assembly, stage=stage)
        return set(measurement.detected_parts), measurement.debug

    def _build_observation(self) -> dict[str, Any]:
        """Build the agent-facing observation from the current environment state."""
        return self.encoder.build_observation(self)

    def _is_milling_capability(self, cap: Capability) -> bool:
        """Return whether a capability should receive the milling penalty."""
        return cap.process is Process.MILLING

    def _check_termination(self) -> bool:
        """
        Terminate when all target parts have been removed from the assembly.
        If no target parts are specified, this returns False (no task-based termination).
        """
        if not self._target_part_set:
            return False

        current_names = {p.name for p in self.assembly.parts.values()}
        # All target parts have been removed ⇒ success
        return all(name not in current_names for name in self._target_part_set)

    def get_last_scene(self):
        """
        Stage2 helper:
        Return the most recent geometric scene (e.g. trimesh.Scene) produced by
        any capability, if available.
        """
        # Prefer the scene cached during the last step call
        if self._last_scene is not None:
            return self._last_scene

        # Fallback: ask each capability explicitly
        for cap in self.capabilities:
            getter = getattr(cap, "get_last_scene", None)
            if callable(getter):
                scene = getter()
                if scene is not None:
                    return scene
        return None

    def _toolchange_prefix(
        self,
        *,
        start_pose: str,  # "assembly" or "dropoff"
        tool_before: str | None,
        tool_after: str | None,
    ) -> tuple[float, list[dict]]:
        """
        Returns (time_cost, motion_program_prefix) for toolchange,
        ending at action start (assembly start), NOT at dropoff.
        """
        if tool_after is None or tool_after == tool_before:
            return 0.0, []

        t = 0.0
        prog: list[dict] = []

        # 1) start -> storage
        if start_pose == "dropoff":
            t += float(
                getattr(self.cell_settings, "toolchange_time_dropoff_to_storage", 0.0)
            )
            prog.append({"type": "transit_dropoff_to_storage"})
        else:
            # assembly -> storage
            t += float(
                getattr(self.cell_settings, "toolchange_time_assembly_to_storage", 0.0)
            )
            prog.append({"type": "transit_assembly_to_storage"})

        # 2) toolchange (duration can be constant or per-pair)
        # Prefer: cell_settings.toolchange_time(tool_before, tool_after)
        toolchange_time = getattr(
            self.cell_settings, "toolchange_time_at_toolstorage", 0.0
        )
        if callable(toolchange_time):
            dt = float(toolchange_time(tool_before, tool_after))
        else:
            dt = float(toolchange_time)
        t += dt
        prog.append(
            {"type": "toolchange", "from": tool_before, "to": tool_after, "dt": dt}
        )

        # 3) storage -> assembly(start)
        t += float(
            getattr(self.cell_settings, "toolchange_time_storage_to_assembly", 0.0)
        )
        prog.append({"type": "transit_storage_to_assembly_start"})

        return t, prog

    def _start_to_action_start_prefix(
        self, *, start_pose: str
    ) -> tuple[float, list[dict]]:
        """
        Returns prefix cost/program to reach the action start without tool change.
        """
        if start_pose != "dropoff":
            return 0.0, []
        dt = float(getattr(self.cell_settings, "traveltime_dropoff_to_assembly", 0.0))
        return dt, [{"type": "transit_dropoff_to_assembly_start"}]

    def _set_feedback(
        self, accepted: int, reason_name: str, raw_action: list[int]
    ) -> None:
        reason_idx = 0
        if reason_name in self._feedback_reason_names:
            reason_idx = self._feedback_reason_names.index(reason_name)

        # IMPORTANT: feedback.last_action must ALWAYS lie inside observation_space.
        # Since rejects may include out-of-range sub0/sub1, we clamp into action_space bounds.
        nvec = np.asarray(
            self.action_space.nvec, dtype=np.int64
        )  # [n_caps, max_dim0, max_dim1]
        a = np.asarray(raw_action, dtype=np.int64)

        # If something weird happens (wrong shape), fall back to zeros safely.
        if a.shape != (3,):
            a = np.zeros(3, dtype=np.int64)

        # Clamp each dimension into [0, nvec[i]-1]
        hi = nvec - 1
        a = np.minimum(np.maximum(a, 0), hi)

        self._last_feedback = {
            "accepted": int(accepted),
            "reason": int(reason_idx),
            "last_action": a,
        }

    def _reject(
        self,
        *,
        cap_idx: int,
        raw_action: list[int],
        reason_name: str,
        penalty: float | None = None,
        target_ref=None,
        extra_info: dict | None = None,
    ):
        self._last_attempted_cap_idx_prev = getattr(
            self, "_last_attempted_cap_idx", None
        )
        self._last_attempted_cap_idx = cap_idx
        self._last_accepted = 0

        # Rejected before the exec gate: it was not evaluated for this step.
        cap = self.capabilities[cap_idx]
        if hasattr(cap, "set_last_exec_gate"):
            cap.set_last_exec_gate(ok=None, stage="rejected", reason=reason_name)

        # feedback for O_a
        self._set_feedback(accepted=0, reason_name=reason_name, raw_action=raw_action)

        if penalty is None:
            penalty = self.rejected_penalty

        reward = -(float(penalty))

        # consistent step_count / termination / truncation
        self.step_count += 1
        terminated = self._check_termination()
        truncated = (self.step_count >= self.max_steps) and not terminated

        trunc_pen = 0.0
        if truncated and not terminated:
            trunc_pen = self.truncation_penalty
            reward -= trunc_pen

        info = {
            "success": False,
            "noop": False,  # illegal != noop
            "illegal_action": True,
            "illegal_reason": reason_name,
            "capability_index": cap_idx,
            "cap_type": type(self.capabilities[cap_idx]).__name__,
            "raw_action": raw_action,
            "penalty": float(penalty),
            "truncation_penalty": float(trunc_pen),
            "assembly": self.assembly,
        }
        info["failure_stage"] = "exec_gate"
        info["exec_gate_ok"] = None  # rejected before the exec gate
        info["attempted"] = False
        info["target_part"] = None
        info["sampled_success"] = None
        info["p_exec"] = None
        info["p_succ"] = None

        mask_reason = None
        if isinstance(extra_info, dict):
            mask_reason = extra_info.get("mask_reason", None)

        if extra_info:
            info.update(extra_info)

        if mask_reason is not None:
            info["mask_reason"] = mask_reason

        info["truth_present_pre"] = None

        if target_ref is not None:
            name = getattr(target_ref, "name", None)
            if isinstance(name, str) and name:
                info["target_part"] = name

        self.last_info = info

        if self.belief_mode == "pomdp" and self.observer is not None:
            self._detected_parts = self._last_detected_parts
        else:
            self._detected_parts = {p.name for p in self.assembly.parts.values()}

        obs = self._build_observation()

        return obs, reward, terminated, truncated, info

    def _validate_or_reject_predecode(self, cap_idx: int, sub0: int, sub1: int):
        """
        Pre-decode validation & masking.

        Policy (unified):
          - Two top-level reject reasons:
              * "out_of_range"  : invalid global index / padding region
              * "masked_out"    : everything else (with fine reason in extra_info["mask_reason"])
          - Fine reasons are recorded in info.extra_info["mask_reason"].
          - Reject reasons: * "none"  : raw action accepted
        """
        if not (0 <= cap_idx < len(self.capabilities)):
            return self._reject(
                cap_idx=max(0, min(cap_idx, len(self.capabilities) - 1)),
                raw_action=[cap_idx, sub0, sub1],
                reason_name="out_of_range",
                extra_info={"why": "cap_idx_out_of_range"},
            )

        cap = self.capabilities[cap_idx]
        subspace = self._cap_action_spaces[cap_idx]
        n0, n1 = int(subspace.nvec[0]), int(subspace.nvec[1])
        raw_action = [cap_idx, sub0, sub1]

        # 1) Local padding / out-of-range subact
        if not (0 <= sub0 < n0) or not (0 <= sub1 < n1):
            return self._reject(
                cap_idx=cap_idx,
                raw_action=raw_action,
                reason_name="out_of_range",
                extra_info={"n0": n0, "n1": n1},
            )

        raw_sub_action = np.array([sub0, sub1], dtype=np.int64)

        # 2) TargetRef creation
        try:
            target_ref = cap.get_target_ref(self.assembly, raw_sub_action)
        except Exception:
            target_ref = None

        # If we cannot even form a TargetRef, reject early
        if target_ref is None:
            return self._reject(
                cap_idx=cap_idx,
                raw_action=raw_action,
                reason_name="masked_out",
                extra_info={"masked": True, "mask_reason": "no_target_ref"},
            )

        # 3) Buffer slots are ALWAYS masked
        if getattr(target_ref, "kind", None) == "buffer_slot":
            return self._reject(
                cap_idx=cap_idx,
                raw_action=raw_action,
                reason_name="masked_out",
                target_ref=target_ref,
                extra_info={"masked": True, "mask_reason": "buffer_slot"},
            )

        # 4) Target missing-> masked_out
        try:
            present = cap.is_target_present(self.assembly, target_ref)
        except Exception:
            present = True
        if not bool(present):
            return self._reject(
                cap_idx=cap_idx,
                raw_action=raw_action,
                reason_name="masked_out",
                target_ref=target_ref,
                extra_info={"masked": True, "mask_reason": "target_missing"},
            )

        # 5) Capability-specific predecode mask(C_k / ψ-gate / base_part etc.)
        mask_fn = getattr(cap, "predecode_mask_reason", None)
        if callable(mask_fn):
            masked, reason, details = mask_fn(
                self.assembly,
                raw_sub_action,
                current_tool_name=self.current_mounted_tool,
                cell_state=self.cell_state,
            )
            if masked:
                if str(reason) == "wrong_tool" and isinstance(details, dict):
                    required = details.get("required", None)

                    prev = getattr(cap, "current_tool_name", None)
                    if hasattr(cap, "current_tool_name"):
                        cap.current_tool_name = self.current_mounted_tool
                    try:
                        sem_tmp = cap.decode_action(self.assembly, raw_sub_action)
                        requested = getattr(sem_tmp, "tool_name", None)
                    finally:
                        if hasattr(cap, "current_tool_name"):
                            cap.current_tool_name = prev

                    mounted = details.get("mounted", None)

                    # PURE toolchange only if requested == required AND mounted != required
                    pure_toolchange = (
                        isinstance(required, str)
                        and required
                        and isinstance(requested, str)
                        and requested
                        and requested == required
                        and mounted != required
                    )

                    # if pure_toolchange:
                    #    if getattr(self, "_last_attempted_cap_idx_prev", None) != cap_idx:
                    #        self._last_toolchange_hint = {
                    #            "cap_idx": cap_idx,
                    #            "raw_action": raw_action,
                    #            "mounted": mounted,
                    #            "required": required,
                    #        }
                    #        self._set_feedback(
                    #            accepted=1,
                    #            reason_name="toolchange_override",
                    #            raw_action=raw_action,
                    #        )
                    #        return None
                    #
                    #    # same-cap repeated: strict reject
                    #    extra = {
                    #        "masked": True,
                    #        "mask_reason": "wrong_tool",
                    #        **(details or {}),
                    #        "requested": requested,
                    #    }
                    #    return self._reject(
                    #        cap_idx=cap_idx,
                    #        raw_action=raw_action,
                    #        reason_name="masked_out",
                    #        target_ref=target_ref,
                    #        extra_info=extra,
                    #    )

                    if pure_toolchange:
                        self._last_toolchange_hint = {
                            "cap_idx": cap_idx,
                            "raw_action": raw_action,
                            "mounted": mounted,
                            "required": required,
                        }
                        self._set_feedback(
                            accepted=1,
                            reason_name="toolchange_override",
                            raw_action=raw_action,
                        )
                        return None

                    # Not pure toolchange => head mismatch or ambiguous => strict reject.
                    extra = {
                        "masked": True,
                        "mask_reason": "head_mismatch",
                        **(details or {}),
                        "requested": requested,
                    }
                    return self._reject(
                        cap_idx=cap_idx,
                        raw_action=raw_action,
                        reason_name="masked_out",
                        target_ref=target_ref,
                        extra_info=extra,
                    )
                # Normalize: always provide a string reason
                reason_str = str(reason) if reason is not None else "unknown"
                extra = {"masked": True, "mask_reason": reason_str}

                # Merge details, but avoid key collisions with our normalized keys
                if isinstance(details, dict):
                    for k, v in details.items():
                        if k in ("masked", "mask_reason"):
                            extra[f"detail_{k}"] = v
                        else:
                            extra[k] = v

                return self._reject(
                    cap_idx=cap_idx,
                    raw_action=raw_action,
                    reason_name="masked_out",
                    target_ref=target_ref,
                    extra_info=extra,
                )

        self._last_target_ref = target_ref

        return None

    def _transition_orchestrate(
        self,
        cap: Capability,
        assembly: Assembly,
        semantic_action: Any,
        cell_state: CellState | None = None,
    ) -> tuple[Assembly, bool, dict, Any]:
        """
        Env-side orchestrator for stochastic transitions.

        ``cell_state`` (default: no holds) is the cell state before the action;
        the one after it is returned as ``extra["cell_state"]``.

        Returns:
            next_asm, success, extra, semantic_action
        """
        if cell_state is None:
            cell_state = CellState()
        extra: dict = {"cell_state": cell_state}

        # Reset exec gate diagnostics
        if hasattr(cap, "set_last_exec_gate"):
            cap.set_last_exec_gate(ok=None, stage="init", reason=None, details={})

        # -------- 1) Best-effort planning --------
        try:
            plan_fn = getattr(cap, "plan_motion", None)
            if callable(plan_fn):
                planned = plan_fn(assembly, semantic_action, rng=self.rng)
                if planned is not None:
                    semantic_action = planned
        except Exception as e:
            if hasattr(cap, "set_last_exec_gate"):
                cap.set_last_exec_gate(
                    ok=False,
                    stage="planning",
                    reason="planning_failed",
                    details={"err": repr(e)},
                )
            return copy.deepcopy(assembly), False, extra, semantic_action

        op = getattr(semantic_action, "operation", None)
        data = getattr(op, "data", None) if op is not None else None
        # The removal plan core's judgement of the planned operation (#234); an operation planned outside
        # the core is judged here, without a DOF fallback.
        plan = removal_plan_of(op, None, None) if isinstance(data, dict) else None
        failure = plan.failure if plan is not None else None

        if failure is not None and failure.stage == "planning":
            if hasattr(cap, "set_last_exec_gate"):
                cap.set_last_exec_gate(
                    ok=False,
                    stage="planning",
                    reason=failure.reason,
                    details=dict(failure.details),
                )
            return copy.deepcopy(assembly), False, extra, semantic_action

        # -------- 2) Exec gate (non-EOL hard gate) --------
        exec_ok = True
        exec_reason = None
        exec_details: dict = {}

        if op is not None:
            # --- A valid, unit direction, as the removal plan core decided it ---
            if failure is not None:
                exec_ok = False
                exec_reason = failure.reason
                exec_details = dict(failure.details)
            elif plan is not None:
                data["direction"] = plan.direction

            # --- Collision / feasibility check (if op provides it) ---
            if exec_ok:
                # spelling compatibility: check_feasability vs check_feasibility
                chk = getattr(op, "check_feasability", None)
                if not callable(chk):
                    chk = getattr(op, "check_feasibility", None)

                if callable(chk):
                    try:
                        if "cell_state" in inspect.signature(chk).parameters:
                            exec_ok = bool(chk(assembly, cell_state=cell_state))
                        else:
                            exec_ok = bool(chk(assembly))
                        if not exec_ok:
                            exec_reason = "check_feasability_failed"
                    except Exception as e:
                        exec_ok = False
                        exec_reason = "check_feasability_exception"
                        exec_details = {"err": repr(e)}

        # --- Hard gate: screw head must match tool head (env-level invariant) ---
        if cap.process is Process.UNSCREWING:
            screw_name = getattr(semantic_action, "screw_part_name", None)
            base_name = getattr(semantic_action, "base_part_name", None)
            tool_name = getattr(semantic_action, "tool_name", None)

            if screw_name is not None and tool_name is not None:
                # find the screw connection (prefer semantic_action.connection if present)
                conn = getattr(semantic_action, "connection", None)
                if conn is None:
                    # fallback: search in assembly connections
                    for c in getattr(assembly, "connections", []):
                        if getattr(c, "connection_type", None) != "screw":
                            continue
                        p1 = getattr(c, "part_name1", None)
                        p2 = getattr(c, "part_name2", None)
                        if (
                            p1 == screw_name and (base_name is None or p2 == base_name)
                        ) or (
                            p2 == screw_name and (base_name is None or p1 == base_name)
                        ):
                            conn = c
                            break

                required_head = (
                    getattr(conn, "screw_head", None) if conn is not None else None
                )
                # Trace missing required_head / missing conn (NO behavior change: still do not gate)
                if not (isinstance(required_head, str) and required_head):
                    # keep any earlier exec_details (e.g., feasibility exception) if present
                    exec_details.setdefault("hard_gate_trace", {})
                    exec_details["hard_gate_trace"].update(
                        {
                            "why": "missing_required_head",
                            "conn_found": bool(conn is not None),
                            "required_head": required_head,
                            "screw": screw_name,
                            "base": base_name,
                            "requested": tool_name,
                        }
                    )
                else:
                    if tool_name != required_head:
                        exec_ok = False
                        exec_reason = "head_mismatch"
                        exec_details = {
                            "required": required_head,
                            "requested": tool_name,
                            "screw": screw_name,
                            "base": base_name,
                            "conn_found": True,
                        }

        if not exec_ok:
            if hasattr(cap, "set_last_exec_gate"):
                cap.set_last_exec_gate(
                    ok=False,
                    stage="exec_gate",
                    reason=exec_reason or "exec_gate_failed",
                    details=exec_details,
                )
            return copy.deepcopy(assembly), False, extra, semantic_action

        if hasattr(cap, "set_last_exec_gate"):
            cap.set_last_exec_gate(
                ok=True,
                stage="exec_gate",
                reason=None,
                details=exec_details or {},
            )

        # -------- 3) Success model (MVP) --------
        # 3.1 p_exec: non-EOL execution uncertainty (trajectory/tool engagement/etc.)
        exec_factor_fn = getattr(cap, "execution_success_factor", None)
        if callable(exec_factor_fn):
            try:
                p_exec = float(exec_factor_fn(assembly, semantic_action))
            except Exception:
                p_exec = 1.0
        else:
            p_exec = 1.0
        p_exec = float(np.clip(p_exec, 0.0, 1.0))

        # 3.2 p_succ: total success probability
        # MVP choice: EOL operator incorporates p_exec internally.
        p_succ = p_exec
        eol_part = getattr(semantic_action, "part_name", None) or getattr(
            semantic_action, "screw_part_name", None
        )

        if self._eol_op is not None and eol_part is not None:
            try:
                p_succ = float(
                    self._eol_op.expected_success_prob(eol_part, p_exec=p_exec)
                )
            except Exception:
                # conservative fallback: keep p_succ = p_exec
                p_succ = p_exec

        p_succ = float(np.clip(p_succ, 0.0, 1.0))
        sampled_success = bool(self.rng.random() < p_succ)

        extra["p_exec"] = p_exec
        extra["p_succ"] = p_succ
        extra["sampled_success"] = sampled_success

        # -------- 4) Apply deterministic transition if sampled success --------
        next_cell_state = cell_state
        if sampled_success:
            next_asm = cap.deterministic_transition(assembly, semantic_action)
            next_cell_state = cap.cell_state_transition(
                cell_state, semantic_action, next_asm
            )
        else:
            next_asm = copy.deepcopy(assembly)

        # -------- 5) Post-check: ensure intended physical effect happened --------
        # A sampled success succeeds if it changed the assembly or the cell state; only an action
        # that changed nothing is a no-op (#61). A removal changes the assembly only with the removal
        # plan core's result, which it sets only when the part is really gone (#234). This is a loose
        # "anything changed" rule; the capability reporting its own effect (#65) is parked.
        success = sampled_success and (
            next_asm != assembly or next_cell_state != cell_state
        )

        # after computing success ...
        task_success = False
        if self._target_part_set:
            current_names = {p.name for p in next_asm.parts.values()}
            task_success = all(n not in current_names for n in self._target_part_set)

        # If planner removed the target(s), accept it as success even if action-level check failed
        if sampled_success and (not success) and (not task_success):
            # Planner didn't actually change the state → treat as failure
            next_asm = copy.deepcopy(assembly)
            next_cell_state = cell_state
            success = False

        # If task is achieved, mark success=True for consistent logs/EOL
        if task_success:
            success = True
            extra["task_success"] = task_success

        extra["cell_state"] = next_cell_state

        return next_asm, bool(success), extra, semantic_action

    @property
    def current_mounted_tool(self) -> Optional[str]:
        return self._current_mounted_tool

    @current_mounted_tool.setter
    def current_mounted_tool(self, tool: Optional[str]) -> None:
        """The env mounts one tool at a time; every core robot of the cell carries it."""
        self._current_mounted_tool = tool
        self.cell_state = dataclasses.replace(
            self.cell_state, mounted_tools={r: tool for r in self.robot_ids}
        )

    @property
    def robot_ids(self) -> tuple[str, ...]:
        """The core robots of the cell, in cell order: those the capabilities were built with."""
        return tuple(
            dict.fromkeys(r.id for cap in self.capabilities for r in cap.robots)
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        super().reset(seed=seed)
        self.rng = self.np_random

        # Reset assembly and episode counters
        self.assembly = copy.deepcopy(self.base_assembly)
        self.cell_state = CellState()
        self.step_count = 0

        # Reset tool mount state and scene cache
        self.current_mounted_tool = self.initial_mounted_tool
        self._last_toolchange_hint = None

        # Start each episode at assembly area (not at dropoff)
        self._robot_at_dropoff = False

        self._last_cap_idx = None
        self._last_attempted_cap_idx = None
        self._last_attempted_cap_idx_prev = None

        self._last_target_ref = None

        self._last_scene = None
        self.last_info = None

        # Reset / sync each capability's internal tool state
        for cap in self.capabilities:
            if hasattr(cap, "current_tool_name"):
                cap.current_tool_name = self.current_mounted_tool

        # --- WORLD sampler: sample truth missing screws (probabilistic) ---
        world = self.world_sampler or EOLUnifiedSampler()

        # 1) sample missing truth
        missing_parts = set()
        for p in self._obs_part_names:
            pm = float(
                np.clip(world.sample_missing(p, self.assembly, self.rng), 0.0, 1.0)
            )
            if self.rng.random() < pm:
                missing_parts.add(p)
        for p in sorted(missing_parts):
            _remove_part_and_connections_inplace(self.assembly, p)

        # 2) sample stuck truth (after missing removal is OK)
        stuck_parts = set()
        for p in self._obs_part_names:
            if p in missing_parts:
                continue
            p_stuck = float(
                np.clip(world.sample_stuck(p, self.assembly, self.rng), 0.0, 1.0)
            )
            if self.rng.random() < p_stuck:
                stuck_parts.add(p)

        # 3) inject stuck truth into assembly
        inject_eol_stuck_fixed_connections(self.assembly, stuck_parts, world.rules)

        if self._eol_op is not None:
            if self.belief_mode == "fo":
                belief_prior = lambda p, asm, rng: (
                    PartEOLOperatorV2.prior_from_assembly_fixed(
                        p, asm, rng, p_fixed=1.0, p_free=0.0
                    )
                )
            else:
                # POMDP: belief comes from inspection/observer prior
                belief_prior = (
                    self.inspection_prior_sampler
                )  # may be None -> default_p_stuck

            self._eol_op.reset_for_parts(
                part_names=self._obs_part_names,
                rng=self.rng,
                assembly=self.assembly,
                inspection_prior_sampler=belief_prior,
            )

            # FO-MDP: align belief of missing to truth
            if self.belief_mode == "fo":
                for p in missing_parts:
                    try:
                        self._eol_op.on_missing_observed(p, kind="missing_from_start")
                    except Exception:
                        pass

        for cap in self.capabilities:
            if hasattr(cap, "_ever_existed"):
                cap._ever_existed = set()

        self._last_feedback = {
            "accepted": 1,
            "reason": 0,
            "last_action": np.zeros(3, dtype=np.int64),
        }

        # --- observe (visibility proxy) ---
        self._last_accepted = 1
        measurement_debug_reset = None
        self._detected_parts = set()

        if self.belief_mode == "pomdp" and self.observer is not None:
            try:
                self.observer.reset_episode(assembly=self.assembly)
                detected_reset, measurement_debug_reset = self._observe(
                    assembly=self.assembly,
                    stage="reset",
                )
                self._last_detected_parts = detected_reset or set()
                self._last_measurement_debug = measurement_debug_reset

                self._detected_parts = self._last_detected_parts
            except Exception:
                self._detected_parts = set()
                measurement_debug_reset = None
        else:
            self._detected_parts = {p.name for p in self.assembly.parts.values()}

        obs = self._build_observation()
        info = {
            "assembly": self.assembly,
            "measurement_debug_reset": measurement_debug_reset,
        }

        if self._eol_op is not None:
            n = len(self._obs_part_names)
            p_success_ideal = np.ones(n, dtype=np.float32)
            for i, name in enumerate(self._obs_part_names):
                try:
                    p_success_ideal[i] = float(
                        self._eol_op.expected_success_prob(name, p_exec=1.0)
                    )
                except Exception:
                    p_success_ideal[i] = 1.0

            info["eol_p_success_ideal_init"] = p_success_ideal
            info["eol_stuck_sampled_parts"] = sorted(stuck_parts)
            info["eol_fixed_count"] = sum(
                1
                for c in self.assembly.connections
                if getattr(c, "connection_type", None) == "fixed"
            )
            info["eol_missing_sampled_parts"] = sorted(missing_parts)

        return obs, info

    def step(self, action: Any):
        """
        action = [cap_idx, sub0, sub1]:
            cap_idx ∈ [0, n_capabilities)
            sub0, sub1 are interpreted in the local MultiDiscrete([dim0_k, dim1_k])
            of the selected capability k.

        High-level semantics:

        Pre-decode validation (action rejection with feedback):
            0) Synchronize the currently mounted tool into the selected capability.
            1) Reject if (sub0, sub1) lies outside the valid local action range
               of the selected capability (padding / out_of_range).
            2) Reject if the referenced target (part) do not exists
               in the current assembly state
               “masked_out: buffer_slot”
               “target_missing: removed target”

            Rejected actions:
                - Do NOT change the assembly state.
                - Rejected Penalty 100.0
                - Produce explicit action feedback (O_a) indicating the rejection reason.

        Accepted actions:
            4) Decode raw_sub_action -> semantic_action using the selected capability.
            5) Apply capability
            6) Estimate execution time via capability.estimate_action_time,
               including possible tool-change costs.
            7) Add milling penalty if the capability is a milling capability.
            8) Compute reward = -(time_cost + penalty).
            9) Update the environment state (assembly, mounted tool, action masks).

        Additional semantics:
            - A feedback observation (O_a) is emitted for every action, indicating
              whether it was accepted and, if not, why it was rejected.
            - Episode termination is checked after every step.
        """
        self._last_target_ref = None

        arr = np.asarray(action, dtype=int).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"Expected action of length 3, got shape {arr.shape}")

        cap_idx, sub0, sub1 = int(arr[0]), int(arr[1]), int(arr[2])

        prev_attempted = getattr(self, "_last_attempted_cap_idx", None)
        self._last_attempted_cap_idx = cap_idx
        self._last_attempted_cap_idx_prev = prev_attempted

        # optimistic default feedback for this action (will be overwritten on reject)
        self._set_feedback(
            accepted=1, reason_name="none", raw_action=[cap_idx, sub0, sub1]
        )

        # Pre-decode validation (out_of_range / masked_out / target_missing)
        rej = self._validate_or_reject_predecode(cap_idx, sub0, sub1)
        if rej is not None:
            return rej

        self._last_accepted = 1
        detected_pre_set = self._last_detected_parts
        measurement_debug_pre = self._last_measurement_debug

        cap = self.capabilities[cap_idx]

        hint = getattr(self, "_last_toolchange_hint", None)
        self._last_toolchange_hint = None

        start_pose = "dropoff" if self._robot_at_dropoff else "assembly"
        tool_before = self.current_mounted_tool
        decode_tool = tool_before
        if hint is not None:
            req = hint.get("required", None)
            if isinstance(req, str) and req:
                decode_tool = req

        if hasattr(cap, "current_tool_name"):
            cap.current_tool_name = decode_tool

        raw_sub_action = np.array([sub0, sub1], dtype=np.int64)

        # Decode + transition
        semantic_action = cap.decode_action(self.assembly, raw_sub_action)

        # make semantic explicit (optional)
        if hint is not None and decode_tool is not None:
            try:
                setattr(semantic_action, "tool_name", decode_tool)
            except Exception:
                pass

        requested_tool = getattr(semantic_action, "tool_name", None)

        # --- ENV toolchange (prefix) ---
        toolchange_time, toolchange_prog = self._toolchange_prefix(
            start_pose=start_pose,
            tool_before=tool_before,
            tool_after=requested_tool,
        )

        # If toolchange executed, we end at assembly(start) and leave dropoff anchor.
        prefix_time = toolchange_time
        prefix_prog = toolchange_prog

        if toolchange_time > 0.0:
            # commit tool state
            self.current_mounted_tool = requested_tool
            for cap_all in self.capabilities:
                if hasattr(cap_all, "current_tool_name"):
                    cap_all.current_tool_name = self.current_mounted_tool
            # after toolchange, robot is at action start (assembly), not at dropoff
            self._robot_at_dropoff = False
        else:
            # No toolchange: if we were at dropoff, we must travel to assembly(start)
            dt, prog = self._start_to_action_start_prefix(start_pose=start_pose)
            prefix_time += dt
            prefix_prog += prog
            if dt > 0.0:
                self._robot_at_dropoff = False  # we moved to assembly(start)

        pre_names = {p.name for p in self.assembly.parts.values()}

        # --- Transition orchestrate (planning/exec-gate/EOL) ---
        next_assembly, success, extra, semantic_action = self._transition_orchestrate(
            cap, self.assembly, semantic_action, cell_state=self.cell_state
        )

        detected_post_set, measurement_debug_post = self._observe(
            assembly=next_assembly,
            stage="post",
        )
        self._last_detected_parts = detected_post_set or set()
        self._last_measurement_debug = measurement_debug_post

        self._detected_parts = self._last_detected_parts

        # --- Attach prefix motion program into the operation for logging/trace ---
        op = getattr(semantic_action, "operation", None)
        if op is not None and isinstance(getattr(op, "data", None), dict):
            data = op.data
            old = data.get("motion_program")
            if isinstance(old, list):
                data["motion_program"] = prefix_prog + old
            else:
                data["motion_program"] = prefix_prog

        tool_before_effective = tool_before  # the env tool at step start
        # --- Base operation time (cap no longer accounts for toolchange) ---
        base_op_time = float(
            cap.estimate_action_time(
                self.assembly,
                semantic_action,
                self.cell_settings,
                tool_before=tool_before_effective,
                # tool_before=getattr(semantic_action, "tool_name", None),
            )
        )

        time_cost = prefix_time + base_op_time

        # Detect no-op (assembly unchanged) purely by structural equality
        # TODO Need more abundant criteria
        was_noop = (not bool(success)) and (
            {p.name for p in next_assembly.parts.values()}
            == {p.name for p in self.assembly.parts.values()}
        )

        # NOOP classification (info-only)
        noop_reason: str | None = None
        failure_stage: str | None = None
        gate = cap.get_last_exec_gate() if hasattr(cap, "get_last_exec_gate") else None

        if gate is None:
            exec_gate_ok = None
        else:
            ok_val = gate.get("ok", None)
            exec_gate_ok = None if ok_val is None else bool(ok_val)

        exec_fail_reason = None if gate is None else gate.get("reason")
        exec_fail_details = None if gate is None else gate.get("details")
        exec_stage = None if gate is None else gate.get("stage")
        sampled = extra.get("sampled_success", None)

        if not bool(success):
            if exec_gate_ok is False:
                # Exec gate failed => NOT an EOL-informative failure
                failure_stage = "exec_gate"
                noop_reason = f"exec_infeasible:{exec_fail_reason or 'unknown'}"
            else:
                # Exec gate passed (or unknown): failure may be EOL-informative
                if sampled is False:
                    failure_stage = "stochastic"  # or "prob_model_miss"
                    noop_reason = "sample_miss"
                elif sampled is True:
                    failure_stage = "semantic"
                    noop_reason = "sampled_but_no_effect"
                else:
                    failure_stage = "unknown"
                    noop_reason = "missing_sample_flag"
        else:
            failure_stage = "success"
            noop_reason = None

        # If it did not change the assembly, mark explicitly as noop in info
        if was_noop and not bool(success) and noop_reason is None:
            noop_reason = "noop_unknown"

        # --- EOL belief update (#62) ---
        # Only an attempt that passed the exec gate is EOL-informative; an action that
        # changed nothing counts as a failed attempt. The update lives here, not in
        # _transition_orchestrate: graph extraction calls that and must not move the
        # belief.
        eol_target = getattr(semantic_action, "part_name", None) or getattr(
            semantic_action, "screw_part_name", None
        )
        if (
            self._eol_op is not None
            and isinstance(eol_target, str)
            and exec_gate_ok is True
        ):
            try:
                extra["eol_event"] = self._eol_op.on_attempt(
                    eol_target,
                    bool(success),
                    p_exec=float(extra.get("p_exec", 1.0)),
                )
            except Exception as exc:
                extra["eol_event_error"] = repr(exc)

        # Cache last scene
        self._last_scene = None
        getter = getattr(cap, "get_last_scene", None)
        if callable(getter):
            try:
                scene = getter()
                if scene is not None:
                    self._last_scene = scene
            except Exception:
                self._last_scene = None

        milling_pen = (
            float(self.milling_penalty) if self._is_milling_capability(cap) else 0.0
        )

        # If no-op: add noop penalty
        noop_pen = 0.0

        if was_noop:
            noop_pen = self.noop_penalty

        effective_penalty = float(milling_pen + noop_pen)

        reward = -(time_cost + effective_penalty)

        # Update state
        self.assembly = next_assembly
        self.cell_state = extra["cell_state"]

        if self.belief_mode == "pomdp":
            if isinstance(detected_post_set, set):
                self._detected_parts = detected_post_set
        else:
            self._detected_parts = {p.name for p in self.assembly.parts.values()}

        # --- Update cross-step anchor: did we end at dropoff? ---
        # Only set True on success, because only then we trust the planned motion to have completed.
        if bool(success):
            try:
                op = getattr(semantic_action, "operation", None)
                data = getattr(op, "data", None) if op is not None else None
                prog = data.get("motion_program") if isinstance(data, dict) else None
                ended_at_dropoff = False
                if isinstance(prog, list):
                    for seg in prog:
                        if (
                            isinstance(seg, dict)
                            and seg.get("type") == "transit_to_dropoff"
                        ):
                            ended_at_dropoff = True
                            break
                if ended_at_dropoff:
                    self._robot_at_dropoff = True
            except Exception:
                pass

        # Sync tool into all caps
        for cap_all in self.capabilities:
            if hasattr(cap_all, "current_tool_name"):
                cap_all.current_tool_name = self.current_mounted_tool

        self.step_count += 1
        terminated = self._check_termination()
        truncated = (self.step_count >= self.max_steps) and not terminated

        trunc_pen = 0.0
        if truncated and not terminated:
            trunc_pen = self.truncation_penalty
            reward -= trunc_pen

        info = {
            "success": bool(success),
            "noop": bool(was_noop),
            "capability_index": cap_idx,
            "cap_type": type(cap).__name__,
            "raw_action": [cap_idx, sub0, sub1],
            "semantic_action": semantic_action,
            "time_cost": float(time_cost),
            "penalty": float(effective_penalty),
            "truncation_penalty": float(trunc_pen),
            "failure_stage": failure_stage,
            "noop_reason": noop_reason,
            "exec_gate_ok": exec_gate_ok,
            "exec_fail_reason": exec_fail_reason,
            "exec_fail_details": exec_fail_details,
            "exec_stage": exec_stage,
            "assembly": self.assembly,
        }
        info["illegal_action"] = False
        info["illegal_reason"] = None
        info["mask_reason"] = None
        info["sampled_success"] = extra.get("sampled_success", None)
        info["p_exec"] = extra.get("p_exec", None)
        info["p_succ"] = extra.get("p_succ", None)

        info["prefix_time"] = float(prefix_time)
        info["toolchange_time"] = float(toolchange_time)
        info["start_to_action_start_time"] = float(prefix_time - toolchange_time)
        info["base_op_time"] = float(base_op_time)
        info["prefix_motion_program"] = prefix_prog
        info["tool_before"] = tool_before
        info["tool_after"] = self.current_mounted_tool
        info["robot_at_dropoff_after"] = bool(getattr(self, "_robot_at_dropoff", False))
        info["predecode_toolchange_hint"] = hint
        info["mask_was_toolchange_override"] = bool(hint is not None)
        info["task_success"] = bool(extra.get("task_success", terminated))
        if hint is not None:
            info["tool_required"] = hint.get("required")
            info["tool_mounted_at_mask"] = hint.get("mounted")

        target_ref = getattr(self, "_last_target_ref", None)
        info["target_ref"] = target_ref.to_info() if target_ref is not None else None

        # --- observer protocol fields ---
        # attempted: Only when exec_gate passes is it considered a "real attempt".(EOL-informative)
        attempted = exec_gate_ok is True
        info["attempted"] = bool(attempted)

        # target_part: Semantic goals (for belief index)
        target_part = getattr(semantic_action, "part_name", None) or getattr(
            semantic_action, "screw_part_name", None
        )
        info["target_part"] = target_part

        truth_present_pre = None
        if isinstance(target_part, str) and target_part:
            truth_present_pre = target_part in pre_names

        info["truth_present_pre"] = (
            truth_present_pre  # optional, but nice for debugging
        )
        info["detected_count_pre"] = (
            len(detected_pre_set) if isinstance(detected_pre_set, set) else None
        )
        info["detected_count_post"] = (
            len(detected_post_set) if isinstance(detected_post_set, set) else None
        )
        info["measurement_debug_pre"] = measurement_debug_pre
        info["measurement_debug_post"] = measurement_debug_post

        info["exec_gate_warning"] = None
        if isinstance(exec_fail_details, dict):
            # if you adopt hard_gate_trace:
            trace = exec_fail_details.get("hard_gate_trace")
            if trace and isinstance(trace, dict):
                info["exec_gate_warning"] = trace.get("why")
            # or current scheme:
            elif exec_fail_details.get("why") == "missing_required_head":
                info["exec_gate_warning"] = "missing_required_head"

        if self._eol_op is not None:
            n = len(self._obs_part_names)
            p_success_ideal = np.ones(n, dtype=np.float32)
            for i, name in enumerate(self._obs_part_names):
                try:
                    p_success_ideal[i] = float(
                        self._eol_op.expected_success_prob(name, p_exec=1.0)
                    )
                except Exception:
                    p_success_ideal[i] = 1.0

            info["eol_p_success_ideal"] = p_success_ideal

        if "eol_event" in extra:
            info["eol_event"] = extra["eol_event"]
        if "eol_event_error" in extra:
            info["eol_event_error"] = extra["eol_event_error"]

        self._last_cap_idx = cap_idx

        self.last_info = info

        obs = self._build_observation()

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        """
        Render hook:

        - First try to use the latest scene from the capabilities
          (typically coming from Operation.scene(...)).
        - If not available, print a simple textual representation.
        """
        del mode

        scene = self.get_last_scene()
        if scene is None:
            # Textual fallback: show remaining parts and last action summary
            part_names = [p.name for p in self.assembly.parts.values()]
            print(
                f"Step {self.step_count}: parts={part_names}, last_info={self.last_info}"
            )
            self._last_scene = None
            return None

        self._last_scene = scene
        return scene

    def close(self):
        """
        Close any open viewer resources.

        Current implementation uses blocking trimesh.Scene.show(),
        so there is no persistent viewer handle to close here yet.
        This method is mainly provided for API compatibility.
        """
        self._last_scene = None
        # If in the future you store a viewer handle or viewer window,
        # close it here.
