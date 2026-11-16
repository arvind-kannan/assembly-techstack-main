"""The Goal type of the MRTAMP seam: what it reaches and which steps a run may take."""

import pytest

from mrtamp.planning import Goal, RunSettings


def test_a_goal_needs_at_least_one_part_present_or_absent():
    with pytest.raises(ValueError):
        Goal()


def test_a_part_cannot_be_both_present_and_absent():
    with pytest.raises(ValueError):
        Goal(present=("lid",), absent=("lid",))


def test_run_settings_have_no_default_goal():
    with pytest.raises(TypeError):
        RunSettings()


def test_goal_is_reached_when_absent_parts_are_out_and_present_parts_are_in():
    goal = Goal(present=("screw",), absent=("lid",))

    assert goal.reached(("base", "screw"))
    assert not goal.reached(("base", "lid", "screw"))
    assert not goal.reached(("base",))


def test_a_removing_step_is_allowed_unless_it_removes_a_present_part():
    assert Goal(absent=("lid",)).allows_step(inverse=False, effect={"lid"})
    assert Goal(absent=("lid",)).allows_step(
        inverse=False, effect={"cover"}
    )  # an intermediate removal
    assert not Goal(present=("lid",)).allows_step(inverse=False, effect={"lid"})


def test_an_inserting_step_is_allowed_only_for_present_parts():
    assert Goal(present=("lid", "base")).allows_step(inverse=True, effect={"lid"})
    assert not Goal(absent=("lid",)).allows_step(inverse=True, effect={"lid"})
    assert not Goal(absent=("lid",)).allows_step(inverse=True, effect={"cover"})


def test_goal_serialises_present_and_absent_parts():
    assert Goal(present=("screw",), absent=("lid",)).to_dict() == {
        "present": ["screw"],
        "absent": ["lid"],
    }
