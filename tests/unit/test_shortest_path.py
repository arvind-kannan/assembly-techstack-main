"""Shortest sequence plots retain the chosen multiedge and all parallel actions."""

import copy

import matplotlib
import networkx as nx
import pytest

from disassembly_pomdp.graph_extraction import load_graph, save_graph
from disassembly_pomdp.shortest_path import plot_shortest_path, shortest_path_graph

matplotlib.use("Agg")


def graph():
    g = nx.MultiDiGraph()
    root, middle, goal = ("A", "B", "Base"), ("B", "Base"), ("Base",)
    g.add_node(root, depth=0)
    g.add_node(middle, depth=1)
    g.add_node(goal, depth=2, is_goal=True)
    g.add_edge(root, middle, action_time=8, target_part="A")
    g.add_edge(middle, goal, action_time=12, target_part="B")
    g.add_edge(root, goal, key="slow", action_time=99, target_part="Wrong")
    g.add_edge(
        root,
        goal,
        key="chosen",
        action_time=12,
        parallel=True,
        operations=[
            dict(target_part="A", tool="H5"),
            dict(target_part="B", tool="gripper"),
        ],
        target_parts=["A", "B"],
    )
    return g


def test_keeps_exact_cheapest_multiedge_and_does_not_mutate():
    g = graph()
    before = copy.deepcopy(g)
    selected = shortest_path_graph(g)
    assert selected.number_of_nodes() == 2
    assert [key for _, _, key in selected.edges(keys=True)] == ["chosen"]
    assert selected.graph["total_action_time"] == 12
    assert selected.graph["selected_goal"] == ("Base",)
    assert nx.utils.graphs_equal(g, before)
    assert next(iter(selected.edges(data=True)))[2]["stage"] == 1


def test_searches_all_goals_and_ignores_unreachable_goals():
    g = graph()
    root = next(iter(g))
    for i in range(5):
        g.add_node((str(i),), is_goal=True)
        g.add_edge(root, (str(i),), action_time=10 - i, target_part="A")
    g.add_node(("unreachable",), is_goal=True)
    selected = shortest_path_graph(g)
    assert selected.graph["selected_goal"] == ("4",)
    assert selected.graph["total_action_time"] == 6


def test_explicit_goal_and_root():
    g = graph()
    selected = shortest_path_graph(g, root=("B", "Base"), goal=("Base",))
    assert selected.graph["total_action_time"] == 12
    with pytest.raises(nx.NodeNotFound):
        shortest_path_graph(g, root=("missing",))
    with pytest.raises(nx.NodeNotFound):
        shortest_path_graph(g, goal=("missing",))


def test_no_goals_and_no_reachable_goal():
    g = graph()
    for _, data in g.nodes(data=True):
        data.pop("is_goal", None)
    with pytest.raises(nx.NetworkXNoPath, match="No goal states"):
        shortest_path_graph(g)
    g.add_node(("unreachable",), is_goal=True)
    with pytest.raises(nx.NetworkXNoPath, match="reachable"):
        shortest_path_graph(g)
    with pytest.raises(ValueError, match="root"):
        shortest_path_graph(nx.MultiDiGraph())


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1, None, "12", True])
def test_invalid_weights_rejected(cost):
    g = graph()
    g[("A", "B", "Base")][("Base",)]["chosen"]["action_time"] = cost
    with pytest.raises(ValueError, match="finite nonnegative"):
        shortest_path_graph(g)


def test_single_node_already_goal(tmp_path):
    g = nx.MultiDiGraph()
    g.add_node(("Base",), depth=0, is_goal=True)
    selected = shortest_path_graph(g)
    assert selected.graph["total_action_time"] == 0
    fig = plot_shortest_path(g, tmp_path / "already_done.png")
    assert any("START / GOAL" in t.get_text() for t in fig.axes[0].texts)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_lists_parallel_operations_and_serializes(tmp_path):
    selected = shortest_path_graph(graph())
    path = tmp_path / "nested" / "shortest_path.png"
    fig = plot_shortest_path(graph(), path)
    assert path.stat().st_size > 0
    text = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert "PARALLEL" in text and "Remove A" in text and "Remove B" in text
    assert "Wrong" not in text
    assert "12.00 s" in fig.axes[0].get_title()
    save_graph(selected, tmp_path / "path.pkl")
    assert nx.utils.graphs_equal(selected, load_graph(tmp_path / "path.pkl"))
    save_graph(selected, tmp_path / "path.json")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_sequential_plot_has_all_stages(tmp_path):
    g = graph()
    g.remove_edges_from(
        [
            (("A", "B", "Base"), ("Base",), "chosen"),
            (("A", "B", "Base"), ("Base",), "slow"),
        ]
    )
    fig = plot_shortest_path(g, tmp_path / "sequential.png")
    text = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert "Stage 1" in text and "Stage 2" in text and "State 1" in text
    assert "PARALLEL" not in text
    assert "20.00 s" in fig.axes[0].get_title()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_example_exports_shortest_sequence_for_selected_product(tmp_path, monkeypatch):
    import sys
    from unittest.mock import Mock

    from examples import extract_disassembly_graph as example

    product = tmp_path / "product"
    output = tmp_path / "output"
    builder = Mock(return_value=object())
    agent = Mock()
    agent.build_graph.return_value = graph()
    monkeypatch.setattr(example, "build_product_env", builder)
    monkeypatch.setattr(example, "DisassemblyGraphAgent", Mock(return_value=agent))
    monkeypatch.setattr(example, "plot_parallel_graph", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract",
            "--product",
            str(product),
            "--output",
            str(output),
            "--parallel",
            "--robots",
            "2",
        ],
    )
    example.main()
    builder.assert_called_once_with(product)
    assert (output / "shortest_path.png").stat().st_size > 0
    assert (output / "shortest_path.json").is_file()
    selected = load_graph(output / "shortest_path.pkl")
    assert selected.graph["total_action_time"] == 12
    import matplotlib.pyplot as plt

    plt.close("all")
