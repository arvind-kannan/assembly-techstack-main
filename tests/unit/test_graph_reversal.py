"""Completing a sequence graph by reversal: removing steps get inverse inserting steps, milling stays one-way."""

from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_reversal import AssemblyConstants, complete_by_reversal
from tests.mrtamp_scenarios import ALL, lid_screwed_on_base_graph

CONSTANTS = AssemblyConstants(
    screw_in_time=4.0,
    supply_travel_time=6.0,
    insert_approach_time=2.0,
    insert_feed_time=3.0,
)


def inverse_steps(graph, u, v):
    return [s for s in (graph.get_edge_data(u, v) or {}).values() if s.get("inverse")]


def test_unscrewing_gets_an_inverse_screwing_in_step_with_the_screw_in_time_added():
    graph = complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS)

    [step] = inverse_steps(graph, ("base", "lid"), ALL)
    assert (step["process"], step["tool"], step["target_part"]) == (
        Process.SCREWING_IN.value,
        "T6",
        "screw",
    )
    assert step["action_time"] == 5.0 + 4.0


def test_removing_gets_an_inverse_inserting_step_with_supply_travel_approach_and_feed_added():
    graph = complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS)

    [step] = inverse_steps(graph, ("base",), ("base", "lid"))
    assert (step["process"], step["tool"], step["target_part"]) == (
        Process.INSERTING.value,
        "H5",
        "lid",
    )
    assert step["action_time"] == 8.0 + 6.0 + 2.0 + 3.0
    assert step["supply_time"] == 6.0


def test_milling_stays_one_way():
    graph = complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS)

    assert inverse_steps(graph, ("base", "screw"), ALL) == []


def test_the_removal_direction_is_reversed_and_the_input_graph_is_left_alone():
    source = lid_screwed_on_base_graph()
    source.edges[("base", "lid"), ("base",), 0]["direction"] = [0.0, 0.0, 1.0]
    edges_before = source.number_of_edges()

    graph = complete_by_reversal(source, CONSTANTS)

    [step] = inverse_steps(graph, ("base",), ("base", "lid"))
    assert step["direction"] == [0.0, 0.0, -1.0]
    assert source.number_of_edges() == edges_before
    assert all(not s.get("inverse") for _, _, s in source.edges(data=True))


def test_the_original_steps_are_kept_and_not_marked_inverse():
    graph = complete_by_reversal(lid_screwed_on_base_graph(), CONSTANTS)

    assert graph.number_of_edges() == 3 + 2
    assert [
        s["process"] for s in (graph.get_edge_data(ALL, ("base", "lid")) or {}).values()
    ] == [Process.UNSCREWING.value]
