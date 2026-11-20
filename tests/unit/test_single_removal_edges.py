"""Single-part tool/direction alternatives do not duplicate graph edges."""

import copy
from unittest.mock import Mock

import networkx as nx
import pytest

from disassembly_pomdp.graph_extraction import (
    DisassemblyGraphAgent,
    load_graph,
    save_graph,
)
from tests.unit.test_graph_parallel import agent_for


def alternatives(agent, first_time=12, second_time=8):
    original = agent.available_steps.side_effect

    def steps(assembly, collapse=False):
        results = original(assembly, collapse)
        return [
            (
                a,
                {
                    **attrs,
                    "action_time": duration,
                    "tool": tool,
                    "direction": [0, 0, sign],
                },
            )
            for a, attrs in results
            for duration, tool, sign in [(first_time, "H5", 1), (second_time, "H6", -1)]
        ]

    agent.available_steps = Mock(side_effect=steps)


@pytest.mark.parametrize("parallel", [False, True])
def test_one_single_edge_keeps_fastest_tool_and_metadata(parallel, tmp_path):
    agent = agent_for(["A"])
    alternatives(agent)
    graph = agent.build_graph(verbose=False, parallel=parallel)
    assert graph.number_of_edges() == 1
    edge = graph[("A", "Base")][("Base",)][0]
    assert edge["tool"] == "H6" and edge["action_time"] == 8
    assert edge["direction"] == [0, 0, -1]
    assert [a["tool"] for a in edge["alternative_actions"]] == ["H5", "H6"]
    save_graph(graph, tmp_path / "graph.pkl")
    save_graph(graph, tmp_path / "graph.json")
    assert nx.utils.graphs_equal(graph, load_graph(tmp_path / "graph.pkl"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, None])
def test_unknown_or_invalid_duration_does_not_hide_valid_action(bad):
    agent = agent_for(["A"])
    alternatives(agent, first_time=bad)
    edge = next(iter(agent.build_graph(verbose=False).edges(data=True)))[2]
    assert edge["action_time"] == 8


def test_equal_time_has_stable_choice_and_keeps_source_attributes_unmodified():
    agent = agent_for(["A"])
    alternatives(agent, first_time=8)
    steps = agent.available_steps(agent.base_assembly)
    snapshot = copy.deepcopy([attrs for _, attrs in steps])
    result = DisassemblyGraphAgent._single_removal_steps(steps)
    assert result[0][1]["tool"] == "H5"
    assert [attrs for _, attrs in steps] == snapshot


def test_geometry_still_sees_all_tool_choices_before_dedup(monkeypatch):
    agent = agent_for(["A", "B"])
    alternatives(agent)
    seen = []

    def screen(assembly, operations, settings):
        seen.append(tuple(op["tool"] for op in operations))
        return True, {"reason": None}

    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry", screen
    )
    graph = agent.build_graph(verbose=False, parallel=True)
    root = ("A", "B", "Base")
    assert len(graph[root][("B", "Base")]) == 1
    assert len(graph[root][("A", "Base")]) == 1
    assert len(graph[root][("Base",)]) == 4  # grouped alternatives remain intact
    assert set(seen) == {("H5", "H5"), ("H5", "H6"), ("H6", "H5"), ("H6", "H6")}
    assert nx.dijkstra_path_length(graph, root, ("Base",), weight="action_time") == 8
