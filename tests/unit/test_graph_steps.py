"""The typed sequence graph step: what an edge reads as, and what it writes back to the graph."""

from diss_assembly_dataclasses.process import Process

from disassembly_pomdp.graph_steps import Step


def test_an_edge_reads_as_a_step_with_its_process_and_optional_attributes_defaulted():
    step = Step.from_edge(
        {
            "process": "unscrewing",
            "tool": "T6",
            "target_part": "screw",
            "action_time": 5,
        }
    )

    assert step == Step(Process.UNSCREWING, "screw", "T6", 5.0)
    assert not step.inverse and step.direction is None and step.supply_time is None


def test_a_step_writes_back_only_the_attributes_it_has():
    step = Step(
        Process.INSERTING,
        "lid",
        "H5",
        17.0,
        inverse=True,
        direction=(0.0, 0.0, -1.0),
        supply_time=6.0,
        label="inverse:lid [H5]",
    )

    attrs = step.to_attrs()

    assert attrs == {
        "process": "inserting",
        "tool": "H5",
        "target_part": "lid",
        "action_time": 17.0,
        "inverse": True,
        "label": "inverse:lid [H5]",
        "direction": [0.0, 0.0, -1.0],
        "supply_time": 6.0,
    }
    assert Step.from_edge(attrs) == step


def test_a_plain_edge_round_trips_without_new_keys():
    attrs = {
        "process": "milling",
        "tool": "mill",
        "target_part": "lid",
        "action_time": 150.0,
    }

    assert Step.from_edge(attrs).to_attrs() == attrs
