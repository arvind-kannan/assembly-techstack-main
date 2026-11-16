"""Collapsing interchangeable screws in the sequence graph extraction (#134), checked on EMO."""

from pathlib import Path

import pytest

from disassembly_pomdp.env_builder import build_env
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent

EMO = Path(__file__).resolve().parents[2] / "emo_sr_v2"
EMO_SCREWS = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_graph():
    return DisassemblyGraphAgent(build_env(EMO)).build_graph(verbose=False)


@pytest.fixture(scope="module")
def collapsed_graph():
    return DisassemblyGraphAgent(build_env(EMO)).build_graph(
        verbose=False, collapse_screw_groups=True
    )


def test_without_the_flag_the_graph_is_unchanged(full_graph):
    assert (full_graph.number_of_nodes(), full_graph.number_of_edges()) == (18, 34)
    assert "collapsed_screw_groups" not in full_graph.graph


def test_interchangeable_screws_come_out_in_one_order(collapsed_graph):
    assert collapsed_graph.graph["collapsed_screw_groups"] == [EMO_SCREWS]
    for state in collapsed_graph.nodes:
        unscrewed = [
            step["target_part"]
            for _, _, step in collapsed_graph.out_edges(state, data=True)
            if step["target_part"] in EMO_SCREWS
        ]
        assert len(unscrewed) <= 1, f"{state}: {unscrewed}"
    root = next(n for n, d in collapsed_graph.nodes(data=True) if d["depth"] == 0)
    assert [
        s["target_part"]
        for _, _, s in collapsed_graph.out_edges(root, data=True)
        if s["target_part"] in EMO_SCREWS
    ] == ["04_screw_1"]


def test_collapsed_graph_is_part_of_the_full_graph(full_graph, collapsed_graph):
    assert set(collapsed_graph.nodes) <= set(full_graph.nodes)
    goals = {n for n, d in collapsed_graph.nodes(data=True) if d.get("is_goal")}
    assert goals and goals <= {
        n for n, d in full_graph.nodes(data=True) if d.get("is_goal")
    }


def test_battery_extraction_passes_the_group_checks():
    """The 41 lid screws of the battery replacement product are removable together and one contiguous block (#176)."""
    battery = (
        Path(__file__).resolve().parents[2] / "cases" / "battery_replacement" / "model"
    )
    graph = DisassemblyGraphAgent(build_env(battery)).build_graph(
        verbose=False, collapse_screw_groups=True
    )  # raises ScrewGroupError if a check fails
    assert [len(g) for g in graph.graph["collapsed_screw_groups"]] == [41]
