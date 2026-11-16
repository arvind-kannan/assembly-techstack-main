"""The removal plan core on a real planner, and the core as the gym's capability plan functions (#128, #234)."""

from pathlib import Path

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection

from motion_planning.removal_planning import (
    REMOVAL_PLAN,
    gripping_and_removing_plan_func,
    plan_removal,
    plan_unscrewing,
    screw_connection_of,
    unscrewing_plan_func,
)

pytestmark = pytest.mark.geom

MODEL_FOLDER = Path(__file__).resolve().parents[2] / "cases" / "angle_grinder" / "model"


@pytest.fixture(scope="module")
def assembly():
    import os

    previous = os.getcwd()
    os.chdir(MODEL_FOLDER)
    try:
        return Assembly.load_from_xml("angle_grinder.xml")
    finally:
        os.chdir(previous)


class Op:
    def __init__(self, data):
        self.data = data


def a_screw(assembly):
    return next(c for c in assembly.connections if isinstance(c, Screw_Connection))


def test_unscrewing_returns_the_planned_operation_and_the_assembly_without_the_screw(
    assembly,
):
    connection = a_screw(assembly)

    plan = plan_unscrewing(assembly, connection)

    assert plan.ok
    assert plan.operation.data["path_duration"] > 0
    assert np.linalg.norm(plan.direction) == pytest.approx(1.0)
    assert plan.result.get_part_by_name(connection.part_name1) is None
    assert assembly.get_part_by_name(connection.part_name1) is not None


def test_removal_of_an_unknown_part_is_a_planning_failure(assembly):
    plan = plan_removal(assembly, "no_such_part")

    assert not plan.ok and "no_such_part" in plan.failure.message


def test_screw_connection_of_finds_the_screw_by_its_part(assembly):
    connection = a_screw(assembly)

    assert screw_connection_of(assembly, connection.part_name1) is connection
    assert screw_connection_of(assembly, "no_such_part") is None


def test_the_unscrewing_plan_func_returns_the_op_carrying_its_removal_plan(
    assembly,
):
    connection = a_screw(assembly)

    op, result = unscrewing_plan_func(
        assembly, Op({"connection": connection, "cost": 5.0})
    )

    assert op.data[REMOVAL_PLAN].result is result
    assert result.get_part_by_name(connection.part_name1) is None


def test_the_plan_funcs_leave_an_op_without_a_target_untouched(assembly):
    op = Op({})

    assert unscrewing_plan_func(assembly, op) == (op, None)
    assert gripping_and_removing_plan_func(assembly, op) == (op, None)
    assert (
        gripping_and_removing_plan_func(assembly, Op({"part_name": "no_such_part"}))[1]
        is None
    )
