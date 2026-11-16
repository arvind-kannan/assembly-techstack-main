"""Unit tests for ``disassembly_pomdp.graph_extraction``."""

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.disassembly_pomdp_gym import DisassemblyPOMDPEnv
from disassembly_pomdp.graph_extraction import (
    DISASSEMBLED_STATES,
    FULL_TEARDOWN,
    DisassemblyGraphAgent,
    ScrewGroupError,
)
from tests.conftest import FakeEnvCapability, FakeObserver, FakePart


def test_rejection_reasons_report_the_exec_gate_reason(
    simple_env_assembly, env_settings
):
    cap = FakeEnvCapability(
        tools=["H5"],
        part_names=["part_1"],
        operation_direction=np.array([0.0, 0.0, 0.0]),
    )
    env = DisassemblyPOMDPEnv(
        assembly=simple_env_assembly,
        capabilities=[cap],
        cell_settings=env_settings,
        target_part_names=["part_1"],
        observer=FakeObserver({"part_1", "part_2"}),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)

    reasons = DisassemblyGraphAgent(env).rejection_reasons(env.base_assembly, "part_1")

    assert len(reasons) == 1
    assert reasons[0]["reason"] == "exec_gate_or_infeasible"
    assert reasons[0]["details"]["reason"] == "bad_direction"


class GrippingAndRemovingCapability(FakeEnvCapability):
    """The fake under the name the extraction reads the process of a step from. part_1 covers the others:
    they come out, in any order, only once part_1 is out."""

    def predecode_mask_reason(
        self, assembly, raw_action, current_tool_name=None, cell_state=None
    ):
        name = self.part_names[int(raw_action[1])]
        if name != "part_1" and "part_1" in assembly.parts:
            return True, "geometrically_infeasible", {}
        return False, None, {}


def _three_part_agent(env_settings, target_part_names):
    """part_1 comes out first; as the target part it is not the last removable part."""
    names = ["part_1", "part_2", "part_3"]
    assembly = Assembly("three_parts")
    assembly.parts = {n: FakePart(n) for n in names}
    assembly.connections = []
    assembly._fcl_objects = {}
    env = DisassemblyPOMDPEnv(
        assembly=assembly,
        capabilities=[GrippingAndRemovingCapability(tools=["H5"], part_names=names)],
        cell_settings=env_settings,
        target_part_names=target_part_names,
        observer=FakeObserver(set(names)),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    return DisassemblyGraphAgent(env)


def test_by_default_extraction_stops_at_the_disassembled_states(env_settings):
    graph = _three_part_agent(env_settings, ["part_1"]).build_graph(verbose=False)
    full = _three_part_agent(env_settings, ["part_1"]).build_graph(
        verbose=False, extraction_depth=FULL_TEARDOWN
    )

    disassembled = {n for n, d in graph.nodes(data=True) if d["is_goal"]}
    assert disassembled == {("part_2", "part_3")}
    assert set(graph.nodes) == {("part_1", "part_2", "part_3"), ("part_2", "part_3")}
    assert set(graph.nodes) < set(full.nodes)  # the states below it are left out
    for state in graph.nodes:
        if state in disassembled:
            assert graph.out_degree(state) == 0, state
        else:
            assert sorted(v for _, v in graph.out_edges(state)) == sorted(
                v for _, v in full.out_edges(state)
            ), state
    assert graph.graph["extraction_depth"] == DISASSEMBLED_STATES


def test_a_full_teardown_expands_every_state_until_no_step_removes_a_part(
    env_settings,
):
    full = _three_part_agent(env_settings, ["part_1"]).build_graph(
        verbose=False, extraction_depth=FULL_TEARDOWN
    )

    assert set(full.nodes) == {
        ("part_1", "part_2", "part_3"),
        ("part_2", "part_3"),
        ("part_2",),
        ("part_3",),
        (),
    }
    assert [n for n in full if full.out_degree(n) == 0] == [()]
    assert full.graph["extraction_depth"] == FULL_TEARDOWN


def test_without_target_parts_both_depths_give_the_same_graph(env_settings):
    default = _three_part_agent(env_settings, []).build_graph(verbose=False)
    full = _three_part_agent(env_settings, []).build_graph(
        verbose=False, extraction_depth=FULL_TEARDOWN
    )

    assert set(default.nodes) == set(full.nodes)
    assert default.number_of_edges() == full.number_of_edges()


def test_an_unknown_depth_is_refused(env_settings):
    with pytest.raises(ValueError, match="depth"):
        _three_part_agent(env_settings, ["part_1"]).build_graph(
            verbose=False, extraction_depth="halfway"
        )


class UnscrewingCapability(FakeEnvCapability):
    """A fake that unscrews: `held_by` maps a part to the parts that must be out before it can come out."""

    process = Process.UNSCREWING

    def __init__(self, *, held_by=None, **kwargs):
        super().__init__(**kwargs)
        self.held_by = held_by or {}

    def predecode_mask_reason(
        self, assembly, raw_action, current_tool_name=None, cell_state=None
    ):
        name = self.part_names[int(raw_action[1])]
        if any(p in assembly.parts for p in self.held_by.get(name, ())):
            return True, "geometrically_infeasible", {}
        return False, None, {}


def _screwed_agent(env_settings, held_by):
    """A base with three screws (one group: same head, same partner), a cover and a clip; `held_by` blocks parts."""
    names = ["base", "s1", "s2", "s3", "cover", "clip"]
    assembly = Assembly("screwed")
    assembly.parts = {n: FakePart(n) for n in names}
    assembly.connections = [
        Screw_Connection(s, "base", "H5") for s in ("s1", "s2", "s3")
    ]
    assembly._fcl_objects = {}
    env = DisassemblyPOMDPEnv(
        assembly=assembly,
        capabilities=[
            UnscrewingCapability(
                tools=["H5"],
                part_names=names,
                held_by={
                    "base": ["s1", "s2", "s3"],
                    **held_by,
                },  # the base stays until the screws are out
            )
        ],
        cell_settings=env_settings,
        target_part_names=["cover"],
        observer=FakeObserver(set(names)),
        initial_mounted_tool="H5",
    )
    env.reset(seed=0)
    return DisassemblyGraphAgent(env)


def test_screws_of_a_group_that_are_all_removable_pass_both_checks(env_settings):
    """The cover and the clip come out only once the screws are out: the screws are one contiguous block."""
    held = {"cover": ["s1", "s2", "s3"], "clip": ["s1", "s2", "s3"]}
    graph = _screwed_agent(env_settings, held).build_graph(
        verbose=False, collapse_screw_groups=True
    )

    assert graph.graph["collapsed_screw_groups"] == [["s1", "s2", "s3"]]


def test_a_blocked_screw_of_a_collapsed_group_fails_loudly_naming_it(env_settings):
    held = {
        "s2": ["clip"],
        "cover": ["s1", "s2", "s3"],
    }  # s2 is blocked, s1 and s3 are not
    agent = _screwed_agent(env_settings, held)

    with pytest.raises(ScrewGroupError, match="s2"):
        agent.build_graph(verbose=False, collapse_screw_groups=True)
    agent.build_graph(verbose=False)  # without collapsing there is nothing to check


def test_a_step_between_the_screws_of_a_group_fails_loudly(env_settings):
    """The clip is free at all times: a path unscrews s1, removes the clip, then unscrews s2."""
    held = {"cover": ["s1", "s2", "s3"]}
    agent = _screwed_agent(env_settings, held)

    with pytest.raises(ScrewGroupError, match="clip") as error:
        agent.build_graph(verbose=False, collapse_screw_groups=True)
    assert "s1" in str(error.value) and "s2" in str(error.value)
    agent.build_graph(verbose=False)
