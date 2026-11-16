from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Geometric_Connection
from diss_assembly_dataclasses.process import Process
from gymnasium import spaces

from crc_disassembly_sequence_generator.operations import MillingOperation
from crc_disassembly_sequence_generator.planning import (
    HierarchicalDisassemblySequencePlanner,
)
from disassembly_pomdp.capability_base import Capability, TargetKind, TargetRef
from disassembly_pomdp.config import CellSettings


@dataclass
class MillingAction:
    """Decoded milling action: which tool corresponds to which pre-planned MillingOperation."""

    tool_name: str
    tool_index: int
    operation_index: int
    operation: MillingOperation


class MillingCapability(Capability):
    """
    MillingCapability (Lightweight Fake Version):
    Action = Select a tool + Select a pre-constructed MillingOperation stub.

    action space: MultiDiscrete([n_milling_tools, n_milling_targets])
    raw_action = [tool_idx, op_idx]
    """

    process = Process.MILLING

    def __init__(
        self,
        name: str,
        available_milling_tools: Sequence[str],
        milling_operations: Sequence[MillingOperation],
        cell_settings: Optional[CellSettings] = None,
    ):
        super().__init__(name)
        self.available_milling_tools = list(available_milling_tools)
        self._milling_operations: List[MillingOperation] = list(milling_operations)
        self.cell_settings = cell_settings

        # For easier debugging: Record keys such as (part_name1, part_name2, inverted) (in the fake version, it's just best-effort).
        self._operation_keys: List[Tuple[str, str, bool]] = []
        for op in self._milling_operations:
            conn = op.data.get("connection", None)
            inverted = bool(op.data.get("inverted", False))
            if conn is not None:
                self._operation_keys.append(
                    (conn.part_name1, conn.part_name2, inverted)
                )

        # action_space will be lazy-built.
        self._action_space: Optional[spaces.MultiDiscrete] = None

    # ---------- Constructor: Lightweightly generates several fake MillingOperations from an assembly. ----------

    @classmethod
    def from_assembly(
        cls,
        assembly: Assembly,
        available_milling_tools: Sequence[str],
        mill_diameter: float = 5.0,
        cost_milling: float = 10.0,
        name: str = "milling",
        cell_settings: Optional[CellSettings] = None,
    ) -> "MillingCapability":
        """
        Lightweight fake version:
        - No longer calls the geometry planner (to avoid excessively long calculations and None returns)
        - Constructs only some "placeholders" for MillingOperation based on the geometric connection Geometric_Connection:

            Generate one op for each Geometric_Connection:
                op.data = {
                    "type": "milling",       # Automatically added by MillingOperation.__init__
                    "connection": gc,
                    "cost": cost_milling,
                    "mill_diameter": mill_diameter,
                }

        - If there are no geometric connections at all, at least one dummy operation is created, ensuring that n_ops >= 1.
        """
        # 1) Find all geometric connections
        geometric_connections: List[Geometric_Connection] = [
            conn
            for conn in assembly.connections
            if isinstance(conn, Geometric_Connection)
            or getattr(conn, "connection_type", None) == "geometric"
        ]

        operations: List[MillingOperation] = []

        for gc in geometric_connections:
            param = {
                "connection": gc,
                "cost": cost_milling,
                "mill_diameter": mill_diameter,
                # Other fields are not needed in the fake version for now; MillingOperation will automatically add "type": "milling"
            }
            op = MillingOperation(param)
            operations.append(op)

        # Even if there are no geometric connections, you still need to ensure that there is at least one op;
        # otherwise, the action space with n_ops=0 will explode.
        if not operations:
            dummy_param = {
                "connection": None,
                "cost": cost_milling,
                "mill_diameter": mill_diameter,
            }
            operations.append(MillingOperation(dummy_param))

        return cls(
            name=name,
            available_milling_tools=available_milling_tools,
            milling_operations=operations,
            cell_settings=cell_settings,
        )

    def list_tools(self) -> List[str]:
        """Return the available milling tool names."""
        return self.available_milling_tools

    # ---------- Capability Interface: Action Space ----------

    def build_action_space(self, assembly: Assembly) -> spaces.Space:
        """
        Action Space:
        First Dimension: Selecting the milling tool (currently using ["default_milling_tool"] → n_tools=1)
        Second Dimension: Selecting a pre-constructed milling operation (discrete index)
        """
        n_tools = len(self.available_milling_tools)
        n_ops = max(
            1, len(self._milling_operations)
        )  # Extreme case: If no op is available, then at least one placeholder will be allocated.

        self._action_space = spaces.MultiDiscrete(
            np.array([n_tools, n_ops], dtype=np.int64)
        )
        return self._action_space

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        if self._action_space is None:
            raise RuntimeError(
                "Action space has not been built yet. Call build_action_space(assembly) first."
            )
        return self._action_space

    def get_target_ref(self, assembly: Assembly, raw_action) -> Optional[TargetRef]:
        """Target of a milling action: the connection its operation mills (none for the placeholder op)."""
        del assembly
        op_idx = int(np.asarray(raw_action, dtype=int).reshape(-1)[1])
        if not (0 <= op_idx < len(self._milling_operations)):
            return None
        conn = self._milling_operations[op_idx].data.get("connection")
        if conn is None:
            return None
        return TargetRef(
            kind=TargetKind.CONNECTION, key=(conn.part_name1, conn.part_name2, None)
        )

    # ---------- Capability interface: decode_action----------

    def decode_action(self, assembly: Assembly, raw_action) -> MillingAction:
        """
        Decode `raw_action = [tool_idx, op_idx]` into `MillingAction`:
        - `tool_idx` selects the `tool_name`
        - `op_idx` selects a pre-constructed `MillingOperation`
        """
        if self._action_space is None:
            # To maintain consistency with gripping_and_removing/Screwing: `build_action_space` must be performed first.
            raise RuntimeError("decode_action called before build_action_space()")

        arr = np.asarray(raw_action, dtype=int).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(
                f"Expected raw_action of shape (2,), got shape {arr.shape}"
            )

        tool_idx, op_idx = int(arr[0]), int(arr[1])

        n_tools, n_ops = (
            int(self._action_space.nvec[0]),
            int(self._action_space.nvec[1]),
        )

        if not (0 <= tool_idx < n_tools):
            raise ValueError(f"tool_idx {tool_idx} out of range [0, {n_tools})")
        if not (0 <= op_idx < n_ops):
            raise ValueError(f"operation index {op_idx} out of range [0, {n_ops})")

        tool_name = self.available_milling_tools[tool_idx]

        # If op_idx exceeds the existing number of operations (e.g., n_ops > len(milling_ops) when expanding in the future), throw an error directly now:
        if op_idx >= len(self._milling_operations):
            raise ValueError(
                f"operation index {op_idx} has no corresponding MillingOperation "
                f"(only {len(self._milling_operations)} ops created)."
            )

        op = self._milling_operations[op_idx]

        return MillingAction(
            tool_name=tool_name,
            tool_index=tool_idx,
            operation_index=op_idx,
            operation=op,
        )

    def deterministic_transition(
        self,
        assembly: Assembly,
        action: MillingAction,
    ) -> Assembly:
        """
        Current simplified version: Does not modify the Assembly structure, only returns a deepcopy.
        The actual docking geometry will be implemented here:
        - Modify the target part (e.g., milling away a screw)
        - Or create a new milled part based on the original part
        - Or adjust the Geometric_Connection structure, etc.
        """
        return copy.deepcopy(assembly)

    def get_success_probability(
        self,
        assembly: Assembly,
        action: MillingAction,
    ) -> float:
        """
        Let's start with a simple, constant success rate.
        Future models can be built based on:
        - Connection topology/depth in operation.data
        - Geometric feasibility in the ψ matrix
        Tool compatibility matrix C_k
        More complex models can be developed.
        """
        return 0.8

    def stochastic_transition(
        self,
        rng: np.random.Generator,
        assembly: Assembly,
        action: MillingAction,
    ) -> Tuple[Assembly, bool]:
        """
        Bernoulli wraps `deterministic_transition`:
        - Calls `deterministic_transition` with `p_success`
        - Otherwise returns a deepcopy of the original state.
        """
        p_succ = float(self.get_success_probability(assembly, action))
        p_succ = max(0.0, min(1.0, p_succ))  # Defensive clipping

        u = float(rng.random())
        if u < p_succ:
            next_asm = self.deterministic_transition(assembly, action)
            return next_asm, True
        else:
            return copy.deepcopy(assembly), False

    def _pick_setting(settings, primary: str, fallback: str) -> float:
        """Return settings.primary if it exists, otherwise settings.fallback."""
        if hasattr(settings, primary):
            return getattr(settings, primary)
        if hasattr(settings, fallback):
            return getattr(settings, fallback)
        raise AttributeError(
            f"Neither '{primary}' nor '{fallback}' is defined in settings."
        )

    def estimate_action_time(
        self,
        assembly: Assembly,
        action: MillingAction,
        settings: CellSettings,
        tool_before: Optional[str] = None,
    ) -> float:
        """
        Simple milling time model (current version), from action start onward:
        - Currently reuses the same scalar parameters as gripping_and_removing/gripping.
        - Compatible with both gripping_and_removing_* and gripping_* naming in CellSettings.
        - Tool change is added by the environment, not here.
        """
        del assembly, action, tool_before

        # ----- Stage1 fallback: same as before, but with safe fallbacks -----
        def pick(primary: str, fallback: str) -> float:
            if hasattr(settings, primary):
                return getattr(settings, primary)
            if hasattr(settings, fallback):
                return getattr(settings, fallback)
            raise AttributeError(
                f"Neither '{primary}' nor '{fallback}' is defined in settings."
            )

        # 1) Distances
        L_approach = pick("gripping_and_removing_start_offset", "gripping_start_offset")
        minimal_pullout = pick(
            "gripping_and_removing_minimal_pullout_value",
            "gripping_minimal_pullout_value",
        )
        pullout_factor = pick(
            "gripping_and_removing_pullout_factor", "gripping_pullout_factor"
        )

        pullout_len = max(
            minimal_pullout,
            pullout_factor * L_approach,
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

        t_approach = L_approach / feed_rate_approach
        t_leave = pullout_len / feed_rate_leave

        return t_approach + t_leave


class GeometricMillingCapability(MillingCapability):
    """
    Enhanced Geometric Milling:
    - `from_assembly_with_planning`: Calls CRC's `HierarchicalDisassemblySequencePlanner.plan_milling`
    Generates a real MillingOperation (including path/mesh/result) for each `Geometric_Connection`.
    - `deterministic_transition`: Truly alters the Assembly (adding milled parts, modifying connections, etc.) through `static_apply_to_assembly`.
    This version is heavy and is only recommended for high-performance environments and small-scale testing.
    """

    def __init__(
        self,
        name: str,
        available_milling_tools: Sequence[str],
        milling_operations: Sequence[MillingOperation],
        cost_milling: float,
        cell_settings: Optional[CellSettings] = None,
    ):
        super().__init__(
            name=name,
            available_milling_tools=available_milling_tools,
            milling_operations=milling_operations,
            cell_settings=cell_settings,
        )
        self.cost_milling = float(cost_milling)

    # ---------- Overload: Generate geometrically realistic MillingOperation through CRC planning ----------

    @classmethod
    def from_assembly_with_planning(
        cls,
        assembly: Assembly,
        available_milling_tools: Sequence[str],
        mill_diameter: float = 5.0,
        cost_milling: float = 10.0,
        name: str = "milling_geom",
        cell_settings: Optional[CellSettings] = None,
    ) -> "GeometricMillingCapability":
        """
        Heavy version constructor:
        - Call HierarchicalDisassemblySequencePlanner.plan_milling for each Geometric_Connection (forward + reverse)
        - Collect all feasible MillingOperations with cost >= 0
        - Ignore cases that return None / cost < 0 (infeasible / not calculated)
        """

        # Take geometric connection
        geometric_connections: list[Geometric_Connection] = [
            conn
            for conn in assembly.connections
            if isinstance(conn, Geometric_Connection)
            or getattr(conn, "connection_type", None) == "geometric"
        ]

        operations: list[MillingOperation] = []

        for gc in geometric_connections:
            for inverted in (False, True):
                # The static method in CRC might simply return None when the DOF is too small.
                res = HierarchicalDisassemblySequencePlanner.plan_milling(
                    assembly=assembly,
                    connection=gc,
                    mill_diameter=mill_diameter,
                    cost_milling=cost_milling,
                    invert=inverted,
                )
                if res is None:
                    # For example, the branch "not computing milling..."
                    continue

                op, asm_after = res  # res = [operation, assembly_work_copy]
                if op is None:
                    continue

                cost = op.data.get("cost", -1)
                if cost is None or cost < 0:
                    # Skip unfeasible milling solutions
                    continue

                # Make sure to leave the inverted/connection information.
                op.data["inverted"] = inverted
                op.data["connection"] = op.data.get("connection", gc)

                operations.append(op)

        if not operations:
            # If no feasible operation is calculated, throw an exception to indicate that it is a geometry configuration problem rather than a capability logic error.
            raise RuntimeError(
                "GeometricMillingCapability.from_assembly_with_planning: "
                "no feasible milling operations found for this assembly."
            )

        return cls(
            name=name,
            available_milling_tools=list(available_milling_tools),
            milling_operations=operations,
            cost_milling=cost_milling,
            cell_settings=cell_settings,
        )

    # ---------- Rewriting deterministic_transition: Truly changing the assembly ----------

    def deterministic_transition(
        self,
        assembly: Assembly,
        action: MillingAction,
    ) -> Assembly:
        """
        Geometric version:
        - Apply the selected MillingOperation to a given assembly using CRC's static_apply_to_assembly
        - Will actually:
        * Modify the milled part mesh
        * May create new milled parts (*_milled_*_2)
        * Update/delete geometric & screw connections
        """
        # Use deepcopy to prevent direct modification of the passed-in assembly.
        asm_copy = copy.deepcopy(assembly)

        # static_apply_to_assembly returns (cost, new_assembly or None)
        cost, new_asm = MillingOperation.static_apply_to_assembly(
            action.operation,
            asm_copy,
            self.cost_milling,
        )

        # If it fails (returns None), it falls back to no-op, but does not throw an exception.
        if new_asm is None:
            return asm_copy

        return new_asm
