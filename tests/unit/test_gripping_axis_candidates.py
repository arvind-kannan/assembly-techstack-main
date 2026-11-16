"""The gripping planner offers the main axis next to a tilted free direction (AT 288)."""

from __future__ import annotations

import numpy as np

from crc_disassembly_sequence_generator.planning import add_main_axis_candidates


def _unit(*v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_tilted_free_direction_adds_its_main_axis_first():
    tilted = _unit(0.111, -0.002, 0.994)  # EMO rotor: 6.4 degrees off z

    out = add_main_axis_candidates([tilted])

    assert np.allclose(out[0], [0.0, 0.0, 1.0])
    assert np.allclose(out[1], tilted)
    assert len(out) == 2


def test_direction_beyond_fifteen_degrees_adds_no_axis():
    far = _unit(0.5, 0.0, 1.0)  # 26.6 degrees off z

    out = add_main_axis_candidates([far])

    assert len(out) == 1
    assert np.allclose(out[0], far)


def test_axis_is_added_once_and_a_free_axis_is_not_duplicated():
    free = [_unit(0.1, 0, 1), _unit(-0.1, 0, 1), np.array([0.0, 0.0, 1.0])]

    out = add_main_axis_candidates(free)

    assert sum(np.allclose(d, [0.0, 0.0, 1.0]) for d in out) == 1
    assert len(out) == 3


def test_negative_axis_and_outward_order():
    free = [_unit(0.05, 0, 1), _unit(0.05, 1, 0), _unit(-1, 0.02, 0)]
    outward = np.array([-1.0, 0.0, 0.0])

    out = add_main_axis_candidates(free, outward)

    assert [list(map(float, a)) for a in out[:3]] == [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    assert len(out) == 6  # three axes, then the three sampled directions
