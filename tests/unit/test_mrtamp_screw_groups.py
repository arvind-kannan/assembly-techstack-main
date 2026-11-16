"""Part positions for B1's minimum screw distance (#264, the one rule for a position #266), on the battery replacement product's model."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mrtamp.screw_groups import (
    MIN_SCREW_DISTANCE,
    apart,
    groups_naming_unknown_parts,
    mesh_position,
    part_positions,
    parts_without_position,
)

MODEL = Path(__file__).resolve().parents[2] / "cases" / "battery_replacement" / "model"


@pytest.mark.geom
def test_part_positions_are_in_metres_so_that_neighbouring_lid_screws_are_closer_than_the_default_distance():
    positions = part_positions(
        (MODEL / "battery_replacement.xml").read_text(encoding="utf-8"), MODEL
    )

    screws = sorted(p for p in positions if p.startswith("lid_screw"))
    assert len(screws) == 41
    nearest = {
        s: min(math.dist(positions[s], positions[t]) for t in screws if t != s)
        for s in screws
    }
    farthest = max(
        math.dist(positions[s], positions[t]) for s in screws for t in screws
    )
    assert all(0.05 < d < MIN_SCREW_DISTANCE for d in nearest.values())  # 72 to 110 mm
    assert 1.0 < farthest < 2.0  # across the 1.4 m lid


class _Mesh:
    def __init__(self, vertices):
        self.vertices = vertices


def test_a_part_has_a_position_only_with_a_non_empty_mesh_and_a_finite_vertex_mean():
    nan = float("nan")
    assert mesh_position(_Mesh([[0, 0, 0], [2000, 0, 0]])) == (1.0, 0.0, 0.0)
    assert mesh_position(_Mesh(np.empty((0, 3)))) is None
    assert mesh_position(_Mesh([[nan, 0, 0], [1, 0, 0]])) is None
    assert mesh_position(_Mesh([[math.inf, 0, 0], [1, 0, 0]])) is None


def test_a_part_whose_mesh_cannot_be_read_has_no_position():
    class Unreadable:
        @property
        def mesh(self):
            raise ValueError("Failed to load mesh")

    assembly = SimpleNamespace(
        parts={"good": SimpleNamespace(mesh=_Mesh([[0, 0, 0]])), "bad": Unreadable()}
    )
    assert parts_without_position(assembly) == ["bad"]


def test_a_screw_group_names_parts_that_are_not_in_the_product_model():
    groups = [["screw_a", "screw_b"], ["screw_c"]]
    assert groups_naming_unknown_parts(groups, ["screw_a", "lid"]) == [
        "screw_b",
        "screw_c",
    ]
    assert groups_naming_unknown_parts(groups, ["screw_a", "screw_b", "screw_c"]) == []


def test_two_screws_are_never_apart_when_one_position_is_not_finite():
    positions = {"a": (0.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0), "c": (math.nan, 0.0, 0.0)}
    assert apart("a", "b", positions, 0.2)
    assert not apart("a", "c", positions, 0.2)
    assert not apart("a", "missing", positions, 0.2)
