"""The precedence relation of a sequence graph, on small hand-built graphs with a known answer."""

import networkx as nx
import pytest
from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_precedence import (
    PrecedenceError,
    Task,
    choose_goal_state,
    extract_precedence,
)
from disassembly_pomdp.graph_reversal import complete_by_reversal

GRIP = Process.GRIPPING_AND_REMOVING
UNSCREW = Process.UNSCREWING
MILL = Process.MILLING


def state(*parts):
    return tuple(sorted(parts))


class GraphBuilder:
    """A sequence graph in the disassembly direction, with the node attributes of an export."""

    def __init__(self, *start_parts):
        self.graph = nx.MultiDiGraph()
        self.root = state(*start_parts)
        self._add_state(self.root, depth=0)

    def _add_state(self, key, depth=None):
        if key not in self.graph:
            attrs = {"parts_present": list(key), "n_parts_present": len(key)}
            if depth is not None:
                attrs["depth"] = depth
            self.graph.add_node(key, **attrs)

    def step(
        self,
        source,
        part,
        process=GRIP,
        tool="H5",
        action_time=1.0,
        direction=None,
        target=None,
    ):
        target = (
            target if target is not None else state(*(p for p in source if p != part))
        )
        self._add_state(target)
        attrs = {
            "process": process.value,
            "capability": "SomeCapability",
            "tool": tool,
            "target_part": part,
            "action_time": action_time,
            "label": f"{process.value}:{part}",
        }
        if direction is not None:
            attrs["direction"] = direction
        self.graph.add_edge(source, target, **attrs)
        return target


def chain():
    """screw holds lid on base: unscrew, then grip off the lid."""
    b = GraphBuilder("base", "lid", "screw")
    s = b.step(b.root, "screw", UNSCREW, "T6", 5.0, direction=[0.0, 0.0, 1.0])
    b.step(s, "lid", GRIP, "H5", 8.0)
    return b


def diamond():
    """a and b come off in either order, then c."""
    b = GraphBuilder("base", "a", "b", "c")
    after_a = b.step(b.root, "a")
    after_b = b.step(b.root, "b")
    both = b.step(after_a, "b")
    assert b.step(after_b, "a") == both
    b.step(both, "c")
    return b


def test_a_chain_is_a_total_order_and_its_steps_are_the_tasks():
    b = chain()

    tasks, precedence = extract_precedence(b.graph, state("base"))

    assert precedence == {("screw", "lid")}
    assert tasks == {
        "screw": Task("screw", UNSCREW.value, "T6", 5.0, (0.0, 0.0, 1.0)),
        "lid": Task("lid", GRIP.value, "H5", 8.0, None),
    }


def test_two_independent_parts_have_no_pair():
    b = GraphBuilder("base", "a", "b")
    b.step(b.step(b.root, "a"), "b")
    b.step(b.step(b.root, "b"), "a")

    tasks, precedence = extract_precedence(b.graph, state("base"))

    assert set(tasks) == {"a", "b"}
    assert precedence == set()


def test_a_diamond_orders_only_the_join():
    b = diamond()

    _, precedence = extract_precedence(b.graph, state("base"))

    assert precedence == {("a", "c"), ("b", "c")}


def test_a_lid_over_a_rotor_orders_two_steps_of_the_same_process():
    b = GraphBuilder("housing", "lid", "rotor", "cable")
    no_lid = b.step(b.root, "lid")
    no_cable = b.step(b.root, "cable")
    b.step(no_cable, "lid")
    no_lid_no_cable = b.step(no_lid, "cable")
    no_rotor = b.step(no_lid, "rotor")
    b.step(no_lid_no_cable, "rotor")
    b.step(no_rotor, "cable")

    tasks, precedence = extract_precedence(b.graph, state("housing"))

    assert tasks["lid"].process == tasks["rotor"].process == GRIP.value
    assert precedence == {("lid", "rotor")}


def test_completing_by_reversal_does_not_change_the_precedence():
    for b in (chain(), diamond()):
        complete = complete_by_reversal(b.graph)
        assert complete.number_of_edges() > b.graph.number_of_edges()

        assert extract_precedence(complete, state("base")) == extract_precedence(
            b.graph, state("base")
        )
        assert choose_goal_state(complete, "full") == choose_goal_state(b.graph, "full")


def test_an_alternative_route_gives_the_cheapest_step_and_its_order():
    # The lid can be milled off with the screw still in, or gripped off after unscrewing.
    b = GraphBuilder("base", "lid", "screw")
    no_screw = b.step(b.root, "screw", UNSCREW, "T6", 5.0)
    b.step(no_screw, "lid", GRIP, "H5", 8.0)
    milled = b.step(b.root, "lid", MILL, "mill", 150.0)
    b.step(milled, "screw", UNSCREW, "T6", 5.0)

    tasks, precedence = extract_precedence(b.graph, state("base"))

    assert (tasks["lid"].process, tasks["lid"].tool, tasks["lid"].action_time) == (
        GRIP.value,
        "H5",
        8.0,
    )
    # Over all paths nothing is forced; but gripping the lid needs the screw out first.
    assert precedence == {("screw", "lid")}


def test_the_cheapest_of_one_process_is_taken_when_its_time_varies_by_state():
    b = GraphBuilder("base", "a", "b")
    b.step(b.step(b.root, "a", action_time=3.0), "b", action_time=4.0)
    b.step(b.step(b.root, "b", action_time=2.0), "a", action_time=1.0)

    tasks, _ = extract_precedence(b.graph, state("base"))

    assert (tasks["a"].action_time, tasks["b"].action_time) == (1.0, 2.0)


def test_a_task_takes_its_time_from_a_step_on_a_chosen_path():
    # a is gripped quicker after b is milled off, but milling is not b's cheapest kind of step.
    b = GraphBuilder("base", "a", "b")
    b.step(b.step(b.root, "a", action_time=5.0), "b", action_time=1.0)
    b.step(b.step(b.root, "b", MILL, "mill", 2.0), "a", action_time=1.0)

    tasks, precedence = extract_precedence(b.graph, state("base"))

    assert (tasks["a"].action_time, tasks["b"].process) == (5.0, GRIP.value)
    assert precedence == {("a", "b")}


def test_cheapest_steps_that_do_not_reach_the_goal_together_raise_and_name_the_parts():
    # Each part is gripped cheaply only while the other is still present; the second one must be milled.
    b = GraphBuilder("base", "a", "b")
    b.step(b.step(b.root, "a", action_time=1.0), "b", MILL, "mill", 10.0)
    b.step(b.step(b.root, "b", action_time=1.0), "a", MILL, "mill", 10.0)

    with pytest.raises(
        PrecedenceError,
        match=r"cheapest steps for parts \['a', 'b'\] do not reach.*\['base'\]",
    ):
        extract_precedence(b.graph, state("base"))


def test_a_part_that_comes_back_gives_contradictory_pairs_and_an_error_naming_them():
    # Non-monotone and not flagged inverse, so it is kept: removing c also drops b, removing a brings b back.
    b = GraphBuilder("a", "b", "c")
    only_a = b.step(b.root, "c", target=state("a"))
    only_b = b.step(only_a, "a", target=state("b"))
    b.step(only_b, "b", target=state())

    with pytest.raises(PrecedenceError, match=r"contradictory.*'a'.*'b'"):
        extract_precedence(b.graph, state())


def test_an_unreachable_goal_raises_and_names_the_parts():
    b = chain()
    b.graph.add_node(state("lid"), parts_present=["lid"], n_parts_present=1)

    with pytest.raises(
        PrecedenceError, match=r"\['lid'\] is not reachable.*\['base', 'lid', 'screw'\]"
    ):
        extract_precedence(b.graph, state("lid"))


def test_a_goal_not_in_the_graph_is_unreachable():
    with pytest.raises(PrecedenceError, match=r"\('nothing',\) is not reachable"):
        extract_precedence(chain().graph, ("nothing",))


def test_a_nan_action_time_on_the_way_to_the_goal_state_raises_in_choose_goal_state():
    b = chain()
    b.graph.edges[b.root, state("base", "lid"), 0]["action_time"] = float("nan")

    with pytest.raises(
        PrecedenceError, match=r"action_time.*'screw'.*\['base', 'lid', 'screw'\]"
    ):
        choose_goal_state(b.graph, "full")


def test_a_nan_action_time_raises_and_names_the_step():
    b = chain()
    b.graph.edges[state("base", "lid"), state("base"), 0]["action_time"] = float("nan")

    with pytest.raises(PrecedenceError, match=r"action_time.*lid.*\['base', 'lid'\]"):
        extract_precedence(b.graph, state("base"))


def test_the_start_state_is_found_without_depth_as_the_state_with_most_parts():
    b = chain()
    for node in b.graph.nodes:
        b.graph.nodes[node].pop("depth", None)

    _, precedence = extract_precedence(b.graph, state("base"))

    assert precedence == {("screw", "lid")}


def test_the_start_state_as_goal_has_no_tasks():
    b = chain()

    assert extract_precedence(b.graph, b.root) == ({}, set())


def test_goal_full_is_the_cheapest_state_with_fewest_parts():
    b = GraphBuilder("base", "a", "b")
    no_a = b.step(b.root, "a", action_time=1.0)
    b.step(no_a, "b", action_time=1.0)  # ('base',) after 2.0
    b.step(no_a, "base", action_time=0.5)  # ('b',) after 1.5, also one part

    assert choose_goal_state(b.graph, "full") == state("b")


def test_goal_target_is_the_cheapest_goal_state():
    b = GraphBuilder("base", "lid", "screw")
    no_screw = b.step(b.root, "screw", UNSCREW, "T6", 5.0)
    gripped = b.step(no_screw, "lid", GRIP, "H5", 8.0)
    milled = b.step(b.root, "lid", MILL, "mill", 150.0)
    for node in b.graph.nodes:
        b.graph.nodes[node]["is_goal"] = "lid" not in node

    assert choose_goal_state(b.graph, "target") == gripped
    assert milled != gripped


def test_goal_target_without_goal_states_raises():
    b = chain()

    with pytest.raises(PrecedenceError, match="is_goal"):
        choose_goal_state(b.graph, "target")


def test_an_unknown_goal_mode_raises():
    with pytest.raises(ValueError, match="'half'"):
        choose_goal_state(chain().graph, "half")
