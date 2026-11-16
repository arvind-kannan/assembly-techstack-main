"""The Assembly hold API is deprecated in favour of CellState but keeps working."""

import pytest

from diss_assembly_dataclasses.assembly import Assembly


def test_hold_api_warns_and_still_works():
    assembly = Assembly("a")

    with pytest.warns(DeprecationWarning, match="assign_agent"):
        assembly.assign_agent("robot_1", "part_1", "g1")
    with pytest.warns(DeprecationWarning, match="get_part_holder"):
        assert assembly.get_part_holder("part_1") == "robot_1"
    with pytest.warns(DeprecationWarning, match="is_agent_busy"):
        assert assembly.is_agent_busy("robot_1") is True
    with pytest.warns(DeprecationWarning, match="release_agent"):
        assembly.release_agent("robot_1")
    with pytest.warns(DeprecationWarning, match="is_agent_busy"):
        assert assembly.is_agent_busy("robot_1") is False


def test_assign_agent_warns_once():
    assembly = Assembly("a")

    with pytest.warns(DeprecationWarning) as record:
        assembly.assign_agent("robot_1", "part_1", "g1")

    assert len(record) == 1
