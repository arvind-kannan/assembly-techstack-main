from __future__ import annotations

import copy
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

# ---------------------------------------------------------------------
# Make tests runnable from the project root without installing the package.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CRC_SRC = ROOT / "crc_diss_assembly_dataclasses" / "src"

for path in (ROOT, SRC, CRC_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.capability_base import STAND_IN_ROBOT


@dataclass
class FakePart:
    """Lightweight part object for capability tests without mesh/FCL."""

    name: str


class DummyScrewingOperation:
    """Minimal operation class compatible with ScrewingCapability tests."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def scene_o3d(self, assembly: Assembly):
        return None


@pytest.fixture
def screw_conn_h5() -> Screw_Connection:
    return Screw_Connection("screw_1", "base_1", "H5")


@pytest.fixture
def screw_conn_t6() -> Screw_Connection:
    return Screw_Connection("screw_2", "base_1", "T6")


@pytest.fixture
def simple_screw_assembly(screw_conn_h5: Screw_Connection) -> Assembly:
    asm = Assembly("simple_screw_assembly")
    asm.parts = {
        "screw_1": FakePart("screw_1"),
        "base_1": FakePart("base_1"),
    }
    asm.connections = [screw_conn_h5]
    asm._fcl_objects = {}
    return asm


@pytest.fixture
def two_screw_assembly(
    screw_conn_h5: Screw_Connection,
    screw_conn_t6: Screw_Connection,
) -> Assembly:
    asm = Assembly("two_screw_assembly")
    asm.parts = {
        "screw_1": FakePart("screw_1"),
        "screw_2": FakePart("screw_2"),
        "base_1": FakePart("base_1"),
    }
    asm.connections = [screw_conn_h5, screw_conn_t6]
    asm._fcl_objects = {}
    return asm


@pytest.fixture
def dummy_settings() -> SimpleNamespace:
    """
    Minimal settings object with only fields used by ScrewingCapability.
    """
    return SimpleNamespace(
        cost_screwing=2.0,
        traveltime_assembly_to_dropoff=3.0,
        screwing_start_offset=10.0,
        screwing_minimal_pullout_value=20.0,
        screwing_pullout_factor=2.0,
        feed_rate_screwing_approaching=10.0,
        feed_rate_screwing_leaving=20.0,
    )


@pytest.fixture
def capability(dummy_settings: SimpleNamespace):
    from disassembly_pomdp.screwing_capability import ScrewingCapability

    return ScrewingCapability(
        available_screwdrivers=["H5", "T6"],
        cell_settings=dummy_settings,
        operation_cls=DummyScrewingOperation,
        plan_func=None,
    )


def build_capability_with_planner(plan_func, dummy_settings: SimpleNamespace):
    from disassembly_pomdp.screwing_capability import ScrewingCapability

    return ScrewingCapability(
        available_screwdrivers=["H5", "T6"],
        cell_settings=dummy_settings,
        operation_cls=DummyScrewingOperation,
        plan_func=plan_func,
    )


def conn_index(capability, screw: str, base: str, head: str | None) -> int:
    """
    Resolve the stable global connection index after build_action_space().
    """
    key = (screw, base, head)
    return capability._global_screw_conn_keys.index(key)


def raw_action_for(capability, tool_name: str, screw: str, base: str, head: str | None):
    tool_idx = capability.available_screwdrivers.index(tool_name)
    idx = conn_index(capability, screw, base, head)
    return np.array([tool_idx, idx], dtype=np.int64)


def set_dof(assembly: Assembly, dof_vectors: list[np.ndarray]):
    """
    Override get_part_dof on one assembly instance.
    ScrewingCapability calls get_dof(part), so this function takes only part.
    """

    def _get_part_dof(part):
        return dof_vectors

    assembly.get_part_dof = _get_part_dof
    return assembly


def set_visibility(assembly: Assembly, visible: bool):
    """
    Override check_visibility on one assembly instance.
    ScrewingCapability calls check_visibility(part, view_dirs).
    """

    def _check_visibility(part, view_dirs):
        return bool(visible)

    assembly.check_visibility = _check_visibility
    return assembly


def successor_without_screw(
    assembly: Assembly, screw_name: str = "screw_1"
) -> Assembly:
    out = copy.deepcopy(assembly)
    out.parts.pop(screw_name, None)
    out.connections = [
        c
        for c in out.connections
        if c.part_name1 != screw_name and c.part_name2 != screw_name
    ]
    return out


# ---------------------------------------------------------------------
# GrippingAndRemovingCapability helpers / fixtures
# ---------------------------------------------------------------------


class DummyGrippingAndRemoving:
    """Minimal operation class compatible with GrippingAndRemovingCapability tests."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def scene_o3d(self, assembly: Assembly):
        return None


@pytest.fixture
def gripping_and_removing_settings() -> SimpleNamespace:
    """
    Minimal settings object with only fields used by GrippingAndRemovingCapability.
    """
    return SimpleNamespace(
        cost_gripping_and_removing=1.5,
        traveltime_assembly_to_dropoff=3.0,
        gripping_start_offset=10.0,
        gripping_minimal_pullout_value=20.0,
        gripping_pullout_factor=2.0,
        feed_rate_gripping_and_removing_approaching=10.0,
        feed_rate_gripping_and_removing_leaving=20.0,
    )


@pytest.fixture
def simple_manip_assembly() -> Assembly:
    asm = Assembly("simple_manip_assembly")
    asm.parts = {
        "part_1": FakePart("part_1"),
        "part_2": FakePart("part_2"),
        "base_1": FakePart("base_1"),
    }
    asm.connections = []
    asm._fcl_objects = {}
    return asm


@pytest.fixture
def manip_capability(gripping_and_removing_settings: SimpleNamespace):
    from disassembly_pomdp.gripping_and_removing_capability import (
        GrippingAndRemovingCapability,
    )

    return GrippingAndRemovingCapability(
        name="gripping_and_removing",
        available_grippers=["gripper_a", "gripper_b"],
        global_part_names=["part_1", "part_2", "base_1", "buffer_milled_1"],
        cell_settings=gripping_and_removing_settings,
        operation_cls=DummyGrippingAndRemoving,
        plan_func=None,
        base_parts=("base_1",),
    )


def build_gripping_and_removing_capability_with_planner(
    plan_func,
    gripping_and_removing_settings: SimpleNamespace,
    global_part_names: list[str] | None = None,
):
    from disassembly_pomdp.gripping_and_removing_capability import (
        GrippingAndRemovingCapability,
    )

    return GrippingAndRemovingCapability(
        name="gripping_and_removing",
        available_grippers=["gripper_a", "gripper_b"],
        global_part_names=global_part_names
        or ["part_1", "part_2", "base_1", "buffer_milled_1"],
        cell_settings=gripping_and_removing_settings,
        operation_cls=DummyGrippingAndRemoving,
        plan_func=plan_func,
        base_parts=("base_1",),
    )


def raw_manip_action_for(capability, tool_name: str, part_name: str):
    tool_idx = capability.available_grippers.index(tool_name)
    part_idx = capability.global_part_names.index(part_name)
    return np.array([tool_idx, part_idx], dtype=np.int64)


def successor_without_part(assembly: Assembly, part_name: str = "part_1") -> Assembly:
    out = copy.deepcopy(assembly)
    out.parts.pop(part_name, None)
    out.connections = [
        c
        for c in out.connections
        if getattr(c, "part_name1", None) != part_name
        and getattr(c, "part_name2", None) != part_name
    ]
    return out


# ---------------------------------------------------------------------
# DisassemblyPOMDPEnv helpers / fixtures
# ---------------------------------------------------------------------


class DummyEnvOperation:
    """Minimal operation object for environment transition tests."""

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data or {}

    def check_feasibility(self, assembly: Assembly) -> bool:
        return bool(self.data.get("feasible", True))

    def scene_o3d(self, assembly: Assembly):
        return None


@dataclass
class DummyEnvAction:
    """Minimal semantic action used by FakeEnvCapability."""

    tool_name: str
    part_name: str
    operation: DummyEnvOperation | None = None


class FakeObserver:
    """Minimal observer implementing the POMDP observer protocol."""

    def __init__(self, detected_parts: set[str] | None = None):
        self.detected_parts = set(detected_parts or set())
        self.reset_calls = 0
        self.observe_calls = []

    def reset_episode(self, *, assembly: Assembly) -> None:
        self.reset_calls += 1

    def observe(self, *, assembly: Assembly, stage: str):
        from disassembly_pomdp.pomdp_observer import MeasurementResult

        self.observe_calls.append(stage)
        return MeasurementResult(
            detected_parts=set(self.detected_parts),
            debug={
                "stage": stage,
                "detected_cnt": len(self.detected_parts),
            },
        )


class FakeEnvCapability:
    """
    Lightweight capability for DisassemblyPOMDPEnv tests.

    It supports:
    - action-space construction
    - target references
    - predecode masking
    - operation planning
    - deterministic target removal

    Its process triggers neither the screw-head gate nor the milling penalty; one robot carries all its tools.
    """

    process = Process.GRIPPING_AND_REMOVING

    def __init__(
        self,
        *,
        name: str = "fake_cap",
        tools: list[str] | None = None,
        part_names: list[str] | None = None,
        mask_reason: str | None = None,
        mask_details: dict | None = None,
        remove_on_success: bool = True,
        operation_direction=None,
        operation_feasible: bool = True,
        p_exec: float = 1.0,
        op_duration: float = 2.0,
        holds_on_success: bool = False,
    ):
        self.name = name
        self.available_tools = tools or ["H5"]
        self.holds_on_success = bool(holds_on_success)
        self.part_names = part_names or ["part_1"]
        self.mask_reason = mask_reason
        self.mask_details = mask_details or {}
        self.remove_on_success = bool(remove_on_success)
        self.operation_direction = operation_direction
        self.operation_feasible = bool(operation_feasible)
        self.p_exec = float(p_exec)
        self.op_duration = float(op_duration)

        self.current_tool_name = None
        self._action_space = None
        self._last_exec_gate = None
        self._last_scene = None

    def list_tools(self):
        return list(self.available_tools)

    @property
    def robots(self):
        return (
            Robot(
                STAND_IN_ROBOT,
                1,
                tools=tuple(self.available_tools),
                capabilities=(self.process,),
            ),
        )

    def build_action_space(self, assembly: Assembly):
        self._action_space = spaces.MultiDiscrete(
            np.array([len(self.available_tools), len(self.part_names)], dtype=np.int64)
        )
        return self._action_space

    @property
    def action_space(self):
        return self._action_space

    def get_target_ref(self, assembly: Assembly, raw_action):
        from disassembly_pomdp.capability_base import TargetRef

        part_idx = int(raw_action[1])
        if part_idx < 0 or part_idx >= len(self.part_names):
            return None
        return TargetRef(kind="part", name=self.part_names[part_idx])

    def is_target_present(self, assembly: Assembly, target):
        if target.name is None:
            return True
        return target.name in getattr(assembly, "parts", {})

    def predecode_mask_reason(
        self, assembly: Assembly, raw_action, current_tool_name=None, cell_state=None
    ):
        if self.mask_reason is not None:
            return True, self.mask_reason, dict(self.mask_details)
        return False, None, {}

    def decode_action(self, assembly: Assembly, raw_action):
        tool_idx = int(raw_action[0])
        part_idx = int(raw_action[1])
        return DummyEnvAction(
            tool_name=self.available_tools[tool_idx],
            part_name=self.part_names[part_idx],
            operation=None,
        )

    def plan_motion(self, assembly: Assembly, action: DummyEnvAction, rng=None):
        direction = self.operation_direction
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])

        action.operation = DummyEnvOperation(
            {
                "direction": np.asarray(direction, dtype=float),
                "path_duration": self.op_duration,
                "motion_duration": self.op_duration,
                "motion_program": [
                    {"type": "fake_core", "duration": self.op_duration},
                    {"type": "transit_to_dropoff", "duration": 0.0},
                ],
                "feasible": self.operation_feasible,
            }
        )
        return action

    def cell_state_transition(
        self, cell_state: CellState, action: DummyEnvAction, next_assembly: Assembly
    ):
        if self.holds_on_success:
            return dataclasses.replace(
                cell_state, holds={**cell_state.holds, "robot_1": action.part_name}
            )
        return cell_state

    def deterministic_transition(self, assembly: Assembly, action: DummyEnvAction):
        out = copy.deepcopy(assembly)
        if self.remove_on_success:
            out.parts.pop(action.part_name, None)
            out.connections = [
                c
                for c in out.connections
                if getattr(c, "part_name1", None) != action.part_name
                and getattr(c, "part_name2", None) != action.part_name
            ]
        return out

    def execution_success_factor(
        self, assembly: Assembly, action: DummyEnvAction
    ) -> float:
        return self.p_exec

    def estimate_action_time(
        self, assembly: Assembly, action: DummyEnvAction, settings, tool_before=None
    ):
        op = getattr(action, "operation", None)
        data = getattr(op, "data", None) if op is not None else None
        if isinstance(data, dict) and "motion_duration" in data:
            return float(data["motion_duration"])
        return self.op_duration

    def set_last_exec_gate(self, ok, stage=None, reason=None, details=None):
        self._last_exec_gate = {
            "ok": None if ok is None else bool(ok),
            "stage": stage,
            "reason": reason,
            "details": details or {},
        }

    def get_last_exec_gate(self):
        return self._last_exec_gate

    def get_last_scene(self):
        return self._last_scene


@pytest.fixture
def env_settings() -> SimpleNamespace:
    return SimpleNamespace(
        toolchange_time_dropoff_to_storage=1.0,
        toolchange_time_assembly_to_storage=1.0,
        toolchange_time_at_toolstorage=2.0,
        toolchange_time_storage_to_assembly=1.0,
        traveltime_dropoff_to_assembly=1.0,
    )


@pytest.fixture
def simple_env_assembly() -> Assembly:
    asm = Assembly("simple_env_assembly")
    asm.parts = {
        "part_1": FakePart("part_1"),
        "part_2": FakePart("part_2"),
    }
    asm.connections = []
    asm._fcl_objects = {}
    return asm


@pytest.fixture(autouse=True)
def _isolated_product_explorer_config(tmp_path_factory, monkeypatch):
    """The Product explorer's app setting and environment variable (#257) never come from the real user."""
    config = tmp_path_factory.mktemp("user_config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("APPDATA", str(config))
    monkeypatch.delenv("PRODUCT_EXPLORER_PRODUCT_ROOT", raising=False)
