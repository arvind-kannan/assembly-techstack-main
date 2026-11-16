"""Unit tests for shared helpers in ``disassembly_pomdp.capability_base``."""

import pytest
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from disassembly_pomdp.capability_base import suppress_stdout
from disassembly_pomdp.gripping_and_removing_capability import (
    GrippingAndRemovingCapability,
)
from disassembly_pomdp.milling_capability import (
    GeometricMillingCapability,
    MillingCapability,
)
from disassembly_pomdp.screwing_capability import ScrewingCapability


def test_suppress_stdout_silences_prints_and_restores_stdout(capsys):
    print("before")
    with suppress_stdout():
        print("noisy geometry helper output")
    print("after")

    assert capsys.readouterr().out == "before\nafter\n"


@pytest.fixture
def milling_capability():
    return MillingCapability("milling", ["mill_5mm"], milling_operations=[])


@pytest.fixture(params=["capability", "manip_capability", "milling_capability"])
def any_capability(request):
    """Each real capability the env resets the execution gate on."""
    return request.getfixturevalue(request.param)


def test_exec_gate_reset_records_not_evaluated(any_capability):
    any_capability.set_last_exec_gate(ok=None, stage="init", reason=None, details={})

    assert any_capability.get_last_exec_gate() == {
        "ok": None,
        "stage": "init",
        "reason": None,
        "details": {},
    }


def test_exec_gate_records_pass_and_fail_as_bools(any_capability):
    any_capability.set_last_exec_gate(ok=1, stage="exec_gate")
    assert any_capability.get_last_exec_gate()["ok"] is True

    any_capability.set_last_exec_gate(ok=0, stage="exec_gate", reason="bad_direction")
    assert any_capability.get_last_exec_gate()["ok"] is False
    assert any_capability.get_last_exec_gate()["reason"] == "bad_direction"


@pytest.mark.parametrize(
    "capability_cls, process",
    [
        (ScrewingCapability, Process.UNSCREWING),
        (GrippingAndRemovingCapability, Process.GRIPPING_AND_REMOVING),
        (MillingCapability, Process.MILLING),
        (GeometricMillingCapability, Process.MILLING),
    ],
)
def test_every_capability_declares_its_core_process(capability_cls, process):
    assert capability_cls.process is process


def test_a_capability_without_a_process_fails_at_construction():
    class Undeclared(MillingCapability):
        process = None

    with pytest.raises(TypeError, match="Undeclared declares no core process"):
        Undeclared("milling", ["mill_5mm"], milling_operations=[])


def test_without_a_cell_one_stand_in_robot_carries_all_tools_of_the_capability(
    capability,
):
    assert capability.robots_for("H5") == ("robot_1",)
    assert capability.robots_for("T6") == ("robot_1",)
    assert capability.robots_for("H6") == ()


def test_every_robot_of_the_cell_that_carries_the_tool_has_the_capability():
    robots = (
        Robot("robot_1", 1, tools=("H5",), capabilities=(Process.GRIPPING,)),
        Robot("robot_2", 2, tools=("H5",), capabilities=(Process.UNSCREWING,)),
        Robot("robot_3", 3, tools=("H5", "T6"), capabilities=(Process.UNSCREWING,)),
    )
    capability = ScrewingCapability(["H5", "T6"], robots=robots)

    assert capability.robots_for("H5") == ("robot_2", "robot_3")
    assert capability.robot_for("H5") == "robot_2"
    assert capability.robots_for("T6") == ("robot_3",)
