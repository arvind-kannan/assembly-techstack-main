"""Parallel stages reuse feasible steps and mutate real Assembly copies."""

import copy
import json
from unittest.mock import Mock

import matplotlib
import networkx as nx
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection

from disassembly_pomdp.graph_extraction import (
    DisassemblyGraphAgent,
    load_graph,
    plot_parallel_graph,
    save_graph,
)
from tests.conftest import FakePart
from tests.unit.test_graph_extraction import _three_part_agent

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def geometry_screen(monkeypatch):
    # Architecture tests use FakePart without meshes; real geometry is tested
    # separately in test_parallel_geometry.py.
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry",
        lambda *args: (True, {"reason": None}),
    )


def agent_for(names):
    agent = DisassemblyGraphAgent.__new__(DisassemblyGraphAgent)
    assembly = Assembly("parallel")
    assembly.parts = {n: FakePart(n) for n in [*names, "Base"]}
    assembly.connections = [Screw_Connection(n, "Base", screw_head="H5") for n in names]
    agent.base_assembly = assembly
    agent._all_part_names = list(assembly.parts)
    agent.target_part_names = list(names)
    agent.capabilities = []
    agent.env = Mock()
    agent._rng = None

    def steps(current, collapse=False):
        results = []
        for i, name in enumerate(names):
            if current.get_part_by_name(name) is None:
                continue
            successor = copy.deepcopy(current)
            successor.remove_part(successor.get_part_by_name(name))
            results.append(
                (
                    successor,
                    dict(
                        target_part=name,
                        action_time=8 + 4 * i,
                        tool="H5",
                        capability="Fake",
                        raw_action=[0, 0, i],
                        label=name,
                    ),
                )
            )
        return results

    agent.available_steps = Mock(side_effect=steps)
    return agent


def test_two_parts_direct_stage_and_connections():
    agent = agent_for(["A", "B"])
    graph = agent.build_graph(parallel=True, verbose=False)
    root = ("A", "B", "Base")
    edge = graph[root][("Base",)][0]
    assert edge["target_parts"] == ["A", "B"]
    assert edge["parallel"] and edge["n_operations"] == 2
    assert edge["action_time"] == 12
    assert edge["operations"][0]["raw_action"] == [0, 0, 0]
    assert edge["feasibility"] == "geometry_screened"
    assert not agent.assemblies[("Base",)].connections
    assert len(agent.base_assembly.connections) == 2
    assert graph.out_degree(("Base",)) == 0


@pytest.mark.parametrize("robots,expected", [(1, 3), (2, 6), (3, 7)])
def test_all_group_sizes_and_one_feasibility_pass(robots, expected):
    agent = agent_for(["A", "B", "C"])
    steps = agent.available_parallel_steps(agent.base_assembly, n_robots=robots)
    agent.available_steps.assert_called_once_with(agent.base_assembly, False)
    assert len(steps) == expected
    pairs = {tuple(d["target_parts"]) for _, d in steps if d.get("n_operations") == 2}
    assert pairs == ({("A", "B"), ("A", "C"), ("B", "C")} if robots > 1 else set())
    assert len({tuple(a.parts) for a, _ in steps}) == expected
    if robots == 3:
        assert steps[-1][1]["n_operations"] == 3


def test_singleton_preserves_result():
    agent = agent_for(["A"])
    result = agent.available_parallel_steps(agent.base_assembly)
    assert len(result) == 1
    assert "parallel" not in result[0][1]
    assert result[0][1]["target_part"] == "A"


def test_duplicate_targets_and_missing_parts_rejected():
    agent = agent_for(["A", "B"])
    steps = agent.available_steps(agent.base_assembly)
    assert (
        agent._try_parallel_actions(agent.base_assembly, [steps[0], steps[0]]) is None
    )
    bad = (steps[1][0], {**steps[1][1], "target_part": "Missing"})
    assert agent._try_parallel_actions(agent.base_assembly, [steps[0], bad]) is None
    agent.available_steps = Mock(return_value=[steps[0], steps[0], steps[1]])
    results = agent.available_parallel_steps(agent.base_assembly)
    assert len(results) == 5  # three singletons, two alternative A+B stages


def test_failed_mutation_is_rejected(monkeypatch):
    agent = agent_for(["A", "B"])
    group = agent.available_steps(agent.base_assembly)
    monkeypatch.setattr(
        Assembly, "remove_part", Mock(side_effect=ValueError("invalid"))
    )
    assert agent._try_parallel_actions(agent.base_assembly, group) is None


@pytest.mark.parametrize("robots", [0, -1, 1.5, True])
def test_invalid_robot_counts(robots):
    agent = agent_for(["A"])
    with pytest.raises(ValueError, match="positive integer"):
        agent.available_parallel_steps(agent.base_assembly, robots)


def test_collapsing_explicitly_unsupported():
    agent = agent_for(["A"])
    with pytest.raises(ValueError, match="collapse_screw_groups"):
        agent.build_graph(parallel=True, collapse_screw_groups=True)
    agent.available_steps.assert_not_called()


def test_default_sequential_graph_and_one_robot_equivalence():
    agent = agent_for(["A", "B"])
    sequential = agent.build_graph(verbose=False)
    explicit = agent.build_graph(verbose=False, parallel=False)
    one_robot = agent.build_graph(verbose=False, parallel=True, n_robots=1)
    assert nx.utils.graphs_equal(sequential, explicit)
    assert list(sequential.edges(data=True, keys=True)) == list(
        one_robot.edges(data=True, keys=True)
    )
    assert all(len(u) - len(v) == 1 for u, v in sequential.edges())
    assert sequential.number_of_nodes() == 4
    assert sequential.number_of_edges() == 4


def test_feasibility_recomputed_after_stage(env_settings):
    agent = _three_part_agent(env_settings, [])
    graph = agent.build_graph(parallel=True, verbose=False)
    assert not any(
        d.get("parallel")
        for _, _, d in graph.out_edges(tuple(agent._all_part_names), data=True)
    )
    assert graph.has_edge(("part_2", "part_3"), ())


def test_serialization_and_plot_fallback(tmp_path, monkeypatch):
    agent = agent_for(["A", "B"])
    graph = agent.build_graph(parallel=True, verbose=False)
    save_graph(graph, tmp_path / "graph.pkl")
    assert nx.utils.graphs_equal(graph, load_graph(tmp_path / "graph.pkl"))
    save_graph(graph, tmp_path / "graph.json")
    restored = nx.node_link_graph(
        json.loads((tmp_path / "graph.json").read_text()), edges="edges"
    )
    assert nx.utils.graphs_equal(graph, restored)
    import networkx.drawing.nx_agraph as agraph

    monkeypatch.setattr(agraph, "graphviz_layout", Mock(side_effect=ImportError))
    # Keep alternative tool edges in the plot too.
    u, v, attrs = next(iter(graph.edges(data=True)))
    graph.add_edge(u, v, **attrs)
    fig = plot_parallel_graph(graph, tmp_path / "graph.png")
    assert (tmp_path / "graph.png").stat().st_size > 0
    assert any("PARALLEL" in text.get_text() for text in fig.axes[0].texts)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_stage_makespan_uses_cheapest_multiedge():
    agent = agent_for(["A", "B"])
    sequential = agent.build_graph(verbose=False)
    parallel = agent.build_graph(verbose=False, parallel=True)
    root, goal = ("A", "B", "Base"), ("Base",)
    assert nx.dijkstra_path_length(sequential, root, goal, weight="action_time") == 20
    parallel.add_edge(root, goal, action_time=100)
    path = nx.dijkstra_path(parallel, root, goal, weight="action_time")
    best = min(parallel[path[0]][path[1]].values(), key=lambda d: d["action_time"])
    assert best["action_time"] == 12
    assert best["n_operations"] == 2
