import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.process import Process
from diss_assembly_dataclasses.resource import Robot

from motion_planning.common import PlanFailure, snapshot_at, with_share
from motion_planning.removal_planning import removal_plan_of
from tests.mrtamp_scenarios import START_XML

ROBOT = Robot("robot_1", 1, tools=("H5",), capabilities=(Process.SCREWING_IN,))
SHELL = Operation(
    "op_1", Process.SCREWING_IN, "screw", (Share("robot_1", "H5", path_duration=2.0),)
)


def judged(data):
    """The removal plan core's result for a planner operation carrying `data`."""
    return removal_plan_of(SimpleNamespace(data=data), None, "screw")


def test_a_share_gets_the_planners_path_in_metres_and_its_duration():
    data = {
        "path": [[10.0, 20.0, 300.0], [10.0, 20.0, 200.0]],
        "path_duration": 4.5,
        "direction": [0, 0, -2],
    }

    planned = with_share(SHELL, judged(data), ROBOT, "H5")

    [share] = planned.shares
    assert [pytest.approx(p) for p in ((0.01, 0.02, 0.3), (0.01, 0.02, 0.2))] == list(
        share.path
    )  # the planner's mm in metres
    assert share.path_duration == 4.5
    assert planned.direction == (0.0, 0.0, -1.0)


def test_a_failed_plan_raises_a_plan_failure_with_the_planners_reason():
    with pytest.raises(PlanFailure, match="blocked"):
        with_share(
            SHELL, judged({"failure_reason": "blocked", "path": []}), ROBOT, "H5"
        )


def test_snapshot_at_a_product_state_keeps_its_parts_and_carries_the_given_cell_state():
    cell_state = CellState(
        holds={"robot_2": "lid"}, mounted_tools={"robot_1": "T6", "robot_2": "H5"}
    )

    snapshot = snapshot_at(START_XML, ("base", "lid"), cell_state)

    assert snapshot.product_state_key == ("base", "lid")
    connections = [
        (c.attrib["part_name1"], c.attrib["part_name2"])
        for c in ET.fromstring(snapshot.assembly_xml).findall("connection")
    ]
    assert connections == [("base", "lid")]
    assert snapshot.cell_state == cell_state
