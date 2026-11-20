"""Collapse the complete screw set only after certifying every stage."""

import json
from unittest.mock import Mock

import matplotlib
import networkx as nx
import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_extraction import save_graph
from disassembly_pomdp.parallel_diagnostics import plot_rejected_parallel_graph
from disassembly_pomdp.shortest_path import plot_shortest_path
from tests.unit.test_graph_parallel import agent_for

matplotlib.use("Agg")


def screw_agent(names):
    agent = agent_for(names)
    original = agent.available_steps.side_effect

    def steps(current, collapse=False):
        results = original(current, collapse)
        for _, attrs in results:
            attrs["process"] = Process.UNSCREWING.value
            attrs["direction"] = [0, 0, 1]
        return results

    agent.available_steps = Mock(side_effect=steps)
    return agent


@pytest.fixture
def passing(monkeypatch):
    screen = Mock(return_value=(True, {"reason": None}))
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry", screen
    )
    return screen


def test_four_screws_two_robots_reduce_screw_states_to_one_edge(passing):
    agent = screw_agent(["A", "B", "C", "D"])
    full = agent.build_graph(
        verbose=False, parallel=True, n_robots=2, collapse_parallel_screws=False
    )
    reduced = agent.build_graph(verbose=False, parallel=True, n_robots=2)
    assert full.number_of_nodes() == 16
    assert full.number_of_edges() == 56
    assert reduced.number_of_nodes() == 2
    assert reduced.number_of_edges() == 1
    edge = next(iter(reduced.edges(data=True)))[2]
    assert edge["n_operations"] == 4
    assert edge["screw_macro"] and edge["n_stages"] == 2
    assert all(len(stage["operations"]) <= 2 for stage in edge["stages"])
    assert edge["action_time"] == 32  # max(20,16) + max(12,8)
    assert set(agent.assemblies[("Base",)].parts) == {"Base"}
    assert not agent.assemblies[("Base",)].connections


def test_whole_group_failure_preserves_existing_architecture(monkeypatch):
    agent = screw_agent(["A", "B", "C"])

    def screen(assembly, operations, settings):
        return (
            (False, {"reason": "moving_parts_collision"})
            if len(operations) == 3
            else (True, {"reason": None})
        )

    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry", screen
    )
    steps = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert len(steps) == 6  # three singletons and all pairs
    assert not any(d.get("screw_macro") for _, d in steps)
    assert agent.rejected_parallel_graph.number_of_edges() == 1
    edge = next(iter(agent.rejected_parallel_graph.edges(data=True)))[2]
    assert edge["reason"] == "moving_parts_collision"
    assert len(edge["operations"]) == 3


def test_stage_blocked_by_stationary_screw_falls_back(monkeypatch):
    agent = screw_agent(["A", "B", "C", "D"])

    def screen(assembly, operations, settings):
        if len(operations) == 4:
            return True, {"reason": None}
        return False, {
            "reason": "stationary_part_collision",
            "parts": [operations[0]["target_part"], "A"],
        }

    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry", screen
    )
    steps = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert len(steps) == 4  # original singleton results still available
    assert not any(d.get("screw_macro") for _, d in steps)
    assert agent.rejected_parallel_graph.number_of_edges() >= 1


def test_non_screws_and_blocked_screws_not_silently_removed(passing):
    agent = screw_agent(["A", "B", "C"])
    original = agent.available_steps.side_effect

    def mixed(current, collapse=False):
        results = original(current, collapse)
        return [
            (a, {**d, "process": "gripping"} if d["target_part"] == "C" else d)
            for a, d in results
        ]

    agent.available_steps = Mock(side_effect=mixed)
    steps = agent.available_parallel_steps(agent.base_assembly)
    assert len(steps) == 2
    assert steps[0][1]["target_part"] == "C"
    assert set(steps[1][1]["target_parts"]) == {"A", "B"}
    assert "C" in steps[1][0].parts
    agent.available_steps.assert_called_once_with(agent.base_assembly, False)


def test_one_robot_and_odd_group_have_correct_stage_times(passing):
    agent = screw_agent(["A", "B", "C"])
    macro = agent.available_parallel_steps(agent.base_assembly, n_robots=2)[0][1]
    assert macro["n_stages"] == 2 and macro["action_time"] == 24
    assert macro["stages"][-1]["target_part"] == "A"
    serial = agent.available_parallel_steps(agent.base_assembly, n_robots=1)[0][1]
    assert serial["n_stages"] == 3 and serial["action_time"] == 36


def test_only_one_screw_keeps_singletons_and_non_screw_combinations(passing):
    agent = screw_agent(["A"])
    result = agent.available_parallel_steps(agent.base_assembly)
    assert len(result) == 1 and result[0][1]["target_part"] == "A"
    assert "screw_macro" not in result[0][1]


def test_sequential_mode_and_disable_flag_are_unchanged(passing):
    agent = screw_agent(["A", "B"])
    seq = agent.build_graph(verbose=False)
    assert seq.number_of_nodes() == 4 and seq.number_of_edges() == 4
    assert agent.rejected_parallel_graph.number_of_edges() == 0


def test_all_failed_calls_record_reasons_and_do_not_enter_valid_graph(
    passing, monkeypatch, tmp_path
):
    agent = agent_for(["A", "B"])
    group = agent.available_steps(agent.base_assembly)
    assert (
        agent._try_parallel_actions(agent.base_assembly, [group[0], group[0]]) is None
    )
    missing = (group[0][0], {**group[0][1], "target_part": "Missing"})
    assert agent._try_parallel_actions(agent.base_assembly, [group[0], missing]) is None
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry",
        lambda *args: (
            False,
            {"reason": "parts_too_close", "distance": 2, "parts": ["A", "B"]},
        ),
    )
    assert agent._try_parallel_actions(agent.base_assembly, group) is None
    failures = agent.rejected_parallel_graph
    assert failures.number_of_edges() == 3
    reasons = {d["reason"] for _, _, d in failures.edges(data=True)}
    assert reasons == {
        "invalid_or_duplicate_targets",
        "target_missing",
        "parts_too_close",
    }
    save_graph(failures, tmp_path / "rejections.json")
    assert len(json.loads((tmp_path / "rejections.json").read_text())["edges"]) == 3
    fig = plot_rejected_parallel_graph(failures, tmp_path / "rejections.png")
    assert (tmp_path / "rejections.png").stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)
    agent.build_graph(verbose=False)
    assert agent.rejected_parallel_graph.number_of_edges() == 0  # reset between builds


def test_macro_plots_display_substages(passing, tmp_path):
    agent = screw_agent(["A", "B", "C", "D"])
    graph = agent.build_graph(verbose=False, parallel=True)
    fig = plot_shortest_path(graph, tmp_path / "macro.png")
    text = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert (
        "SCREW BATCH" in text
        and "Stage 1: 20.00 s" in text
        and "Stage 2: 12.00 s" in text
    )
    assert "Remove A" in text and "Remove D" in text
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_empty_rejections_plot():
    fig = plot_rejected_parallel_graph(nx.MultiDiGraph())
    assert any("No rejected" in t.get_text() for t in fig.axes[0].texts)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_macro_with_real_meshes_and_geometry_checks():
    from tests.unit.test_parallel_geometry import scene

    agent = screw_agent(["A", "B", "C", "D"])
    geometry = scene(
        {
            name: (index * 20, 0, 0)
            for index, name in enumerate(["A", "B", "C", "D", "Base"])
        }
    )
    for name, part in agent.base_assembly.parts.items():
        part.mesh = geometry.parts[name].mesh
    steps = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert len(steps) == 1
    assert steps[0][1]["screw_macro"]
    assert steps[0][1]["action_time"] == 32
    assert len(steps[0][1]["stages"]) == 2
    assert set(steps[0][0].parts) == {"Base"}


def test_real_incompatible_pair_prevents_partial_screw_collapsing():
    from tests.unit.test_parallel_geometry import scene

    agent = screw_agent(["A", "B", "C", "D"])
    geometry = scene(
        {
            "A": (0, 0, 0),
            "B": (6, 0, 0),
            "C": (30, 0, 0),
            "D": (50, 0, 0),
            "Base": (80, 0, 0),
        }
    )
    for name, part in agent.base_assembly.parts.items():
        part.mesh = geometry.parts[name].mesh
    steps = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert len(steps) == 9  # all singletons plus five geometry-compatible pairs
    assert not any(d.get("screw_macro") for _, d in steps)
    attempts = [data for _, _, data in agent.rejected_parallel_graph.edges(data=True)]
    assert any(
        len(d["operations"]) == 4 and d["reason"] == "parts_too_close" for d in attempts
    )
    assert any(
        len(d["operations"]) == 2 and d["reason"] == "parts_too_close" for d in attempts
    )


def test_alternative_actions_are_tried_without_duplicate_targets(monkeypatch):
    agent = screw_agent(["A", "B"])
    steps = agent.available_steps(agent.base_assembly)
    alternative = (steps[0][0], {**steps[0][1], "tool": "alternate"})
    agent.available_steps = Mock(return_value=[steps[0], alternative, steps[1]])

    def screen(assembly, operations, settings):
        assert len({d["target_part"] for d in operations}) == len(operations)
        if operations[0]["tool"] != "alternate":
            return False, {"reason": "removal_rays_intersect"}
        return True, {"reason": None}

    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry", screen
    )
    results = agent.available_parallel_steps(agent.base_assembly)
    assert len(results) == 1
    assert results[0][1]["screw_macro"]
    assert agent.rejected_parallel_graph.number_of_edges() == 1


def test_a_goal_reached_inside_a_macro_keeps_normal_search(passing):
    agent = screw_agent(["A", "B", "C", "D"])
    agent.target_part_names = ["D"]
    results = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert any(d.get("target_part") == "D" for _, d in results)
    assert not any(d.get("screw_macro") for _, d in results)
