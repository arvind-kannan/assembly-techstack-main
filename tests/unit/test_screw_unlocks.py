"""Expose newly feasible part removals at internal screw-stage boundaries."""

import json
from unittest.mock import Mock

import networkx as nx
import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_extraction import load_graph, save_graph
from tests.unit.test_graph_parallel import agent_for


@pytest.fixture(autouse=True)
def passing_geometry(monkeypatch):
    monkeypatch.setattr(
        "disassembly_pomdp.graph_extraction.check_parallel_geometry",
        lambda *args: (True, {"reason": None}),
    )


def unlocking_agent(required=("D", "E", "F"), initially_available=False):
    agent = agent_for(["A", "B", "C", "D", "E", "F", "Cover"])
    agent.base_assembly.connections = [
        c for c in agent.base_assembly.connections if c.part_name1 != "Cover"
    ]
    agent.target_part_names = ["Cover"]
    original = agent.available_steps.side_effect

    def steps(assembly, collapse=False):
        result = []
        for successor, attrs in original(assembly, collapse):
            attrs = {**attrs, "direction": [0, 0, 1]}
            if attrs["target_part"] == "Cover":
                if not initially_available and any(
                    name in assembly.parts for name in required
                ):
                    continue
                attrs.update(
                    process=Process.GRIPPING_AND_REMOVING.value,
                    capability="GrippingAndRemovingCapability",
                    tool="gripper",
                    action_time=7,
                )
            else:
                attrs.update(
                    process=Process.UNSCREWING.value, capability="ScrewingCapability"
                )
            result.append((successor, attrs))
        return result

    agent.available_steps = Mock(side_effect=steps)
    return agent


def test_six_screws_three_robots_expose_cover_and_screw_continuation(tmp_path):
    agent = unlocking_agent()
    graph = agent.build_graph(parallel=True, n_robots=3, verbose=False)
    root = tuple(sorted(["A", "B", "C", "D", "E", "F", "Cover", "Base"]))
    boundary = tuple(sorted(["A", "B", "C", "Cover", "Base"]))
    goal = ("A", "B", "Base", "C")
    assert graph.has_edge(root, boundary)
    prefix = graph[root][boundary][0]
    assert prefix["batch_stop_reason"] == "non_screw_unlocked"
    assert prefix["unlocked_parts"] == ["Cover"]
    assert set(prefix["target_parts"]) == {"D", "E", "F"}
    assert prefix["n_operations"] == 3 and prefix["n_stages"] == 1
    assert prefix["action_time"] == 28
    assert {op["target_part"] for op in prefix["operations"]} == {"D", "E", "F"}
    assert len(prefix["certified_screw_parts"]) == 6
    assert set(agent.assemblies[boundary].parts) == set(boundary)
    assert not graph.has_edge(root, ("Base", "Cover"))
    # The exposed state has both the freed-cover action and the remaining batch.
    assert graph.has_edge(boundary, goal)
    assert graph.has_edge(boundary, ("Base", "Cover"))
    assert nx.dijkstra_path_length(graph, root, goal, weight="action_time") == 35
    assert graph.out_degree(goal) == 0
    save_graph(graph, tmp_path / "graph.json")
    save_graph(graph, tmp_path / "graph.pkl")
    assert nx.utils.graphs_equal(graph, load_graph(tmp_path / "graph.pkl"))
    assert json.loads((tmp_path / "graph.json").read_text())["edges"]


def test_stops_at_first_unlock_without_validating_unexecuted_suffix():
    agent = unlocking_agent()
    tried = []
    real = agent._try_parallel_actions

    def track(assembly, group):
        tried.append(tuple(attrs["target_part"] for _, attrs in group))
        return real(assembly, group)

    agent._try_parallel_actions = track
    prefix = agent.available_parallel_steps(agent.base_assembly, n_robots=3)[0]
    assert len(prefix[1]["operations"]) == 3
    assert len(tried) == 2  # full-set certificate and only the executed first stage
    assert len(agent.base_assembly.parts) == 8


def test_robot_limited_stage_boundary_can_remove_one_extra_screw():
    agent = unlocking_agent()
    results = agent.available_parallel_steps(agent.base_assembly, n_robots=2)
    assert len(results) == 1
    successor, attrs = results[0]
    assert attrs["unlocked_parts"] == ["Cover"]
    assert attrs["n_operations"] == 4
    assert attrs["n_stages"] == 2
    assert all(len(stage["operations"]) == 2 for stage in attrs["stages"])
    assert set(successor.parts) == {"A", "B", "Cover", "Base"}
    assert attrs["action_time"] == 48  # stage maxima 28 + 20


def test_already_available_non_screw_does_not_split_batch():
    agent = unlocking_agent(initially_available=True)
    results = agent.available_parallel_steps(agent.base_assembly, n_robots=3)
    macro = next(attrs for _, attrs in results if attrs.get("screw_macro"))
    assert macro["n_operations"] == 6 and macro["n_stages"] == 2
    assert macro["batch_stop_reason"] == "screw_set_removed"
    assert macro["unlocked_parts"] == []
    assert any(attrs.get("target_part") == "Cover" for _, attrs in results)


def test_additional_tool_for_existing_part_is_also_an_unlock():
    agent = unlocking_agent(initially_available=True)
    original = agent.available_steps.side_effect

    def steps(assembly, collapse=False):
        available = original(assembly, collapse)
        if all(name not in assembly.parts for name in ("D", "E", "F")):
            for successor, attrs in list(available):
                if attrs["target_part"] == "Cover":
                    available.append((successor, {**attrs, "tool": "second_gripper"}))
        return available

    agent.available_steps = Mock(side_effect=steps)
    macro = next(
        attrs
        for _, attrs in agent.available_parallel_steps(agent.base_assembly, 3)
        if attrs.get("screw_macro")
    )
    assert macro["n_operations"] == 3 and macro["unlocked_parts"] == ["Cover"]


def test_non_unscrewing_actions_on_screw_parts_are_not_part_unlocks():
    agent = unlocking_agent(required=("A", "B", "C", "D", "E", "F"))
    original = agent.available_steps.side_effect

    def steps(assembly, collapse=False):
        available = original(assembly, collapse)
        if "F" not in assembly.parts:
            for successor, attrs in list(available):
                if attrs["target_part"] == "A":
                    available.append(
                        (
                            successor,
                            {
                                **attrs,
                                "process": "gripping_and_removing",
                                "tool": "gripper",
                            },
                        )
                    )
        return available

    agent.available_steps = Mock(side_effect=steps)
    macro = agent.available_parallel_steps(agent.base_assembly, n_robots=3)[0][1]
    assert macro["n_operations"] == 6
    assert macro["batch_stop_reason"] == "screw_set_removed"


def test_disable_collapse_still_exposes_every_screw_subset():
    agent = unlocking_agent()
    graph = agent.build_graph(
        parallel=True, n_robots=3, collapse_parallel_screws=False, verbose=False
    )
    assert not any(attrs.get("screw_macro") for _, _, attrs in graph.edges(data=True))
    assert tuple(sorted(["A", "B", "C", "Cover", "Base"])) in graph


def test_product_explorer_exposes_both_actions_and_unlock_metadata(monkeypatch):
    from fastapi.testclient import TestClient

    from product_explorer.server import create_app
    from product_explorer.session import ExplorerSession
    from tests.unit.test_product_explorer_parallel import generate

    agent = unlocking_agent()
    for part in agent.base_assembly.parts.values():
        part.material, part.density = "Generic", 8000
    session = ExplorerSession()
    session.assembly, session.name = agent.base_assembly, "unlock test"
    session.setup = {"base_parts": ["Base"], "target_parts": ["Cover"]}
    monkeypatch.setattr("product_explorer.session._agent_for", lambda *args: agent)
    client = TestClient(create_app(session))
    view = generate(session, client, parallel=True, n_robots=3)
    assert len(view["next_steps"]) == 1
    prefix_id = view["next_steps"][0]
    prefix = view["graph"]["edges"][prefix_id]
    assert prefix["unlocked_parts"] == ["Cover"]
    assert prefix["n_stages"] == 1 and prefix["action_time"] == 28
    updated = client.post("/api/step", json={"edge": prefix_id}).json()
    next_steps = [updated["graph"]["edges"][key] for key in updated["next_steps"]]
    assert any(step["target_part"] == "Cover" for step in next_steps)
    assert any(
        step["screw_macro"] and set(step["target_parts"]) == {"A", "B", "C"}
        for step in next_steps
    )
    assert set(updated["parts_present"]) == {"A", "B", "C", "Cover", "Base"}


def test_a_later_batch_stops_again_when_another_part_unlocks():
    agent = agent_for(["A", "B", "C", "D", "E", "F", "G", "H", "Cover", "Panel"])
    agent.base_assembly.connections = [
        c
        for c in agent.base_assembly.connections
        if c.part_name1 not in {"Cover", "Panel"}
    ]
    agent.target_part_names = ["Cover", "Panel"]
    original = agent.available_steps.side_effect
    required = {"Cover": {"G", "H"}, "Panel": {"E", "F"}}

    def steps(assembly, collapse=False):
        available = []
        for successor, attrs in original(assembly, collapse):
            name = attrs["target_part"]
            if name in required:
                if required[name] & set(assembly.parts):
                    continue
                attrs = {
                    **attrs,
                    "process": Process.GRIPPING_AND_REMOVING.value,
                    "action_time": 7,
                }
            else:
                attrs = {**attrs, "process": Process.UNSCREWING.value}
            available.append((successor, attrs))
        return available

    agent.available_steps = Mock(side_effect=steps)
    first = agent.available_parallel_steps(agent.base_assembly, n_robots=2)[0]
    assert first[1]["unlocked_parts"] == ["Cover"]
    assert set(first[1]["target_parts"]) == {"G", "H"}
    next_steps = agent.available_parallel_steps(first[0], n_robots=2)
    second = next(step for step in next_steps if step[1].get("screw_macro"))
    assert second[1]["unlocked_parts"] == ["Panel"]
    assert set(second[1]["target_parts"]) == {"E", "F"}
    assert any(attrs.get("target_part") == "Cover" for _, attrs in next_steps)
    final_steps = agent.available_parallel_steps(second[0], n_robots=2)
    assert {attrs.get("target_part") for _, attrs in final_steps} >= {"Cover", "Panel"}
    assert any(attrs.get("screw_macro") for _, attrs in final_steps)
