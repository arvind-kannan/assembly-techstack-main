"""The gripping feasibility check must see a blocker in the removal path (AT 282).

Real `Part`s in a real `Assembly`: python-fcl returns no penetration depth unless
contacts are requested, so a check that passes `return_contacts=False` accepts
every direction.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part

from crc_disassembly_sequence_generator.operations import GrippingAndRemovingOperation


def _box(name: str, size, origin) -> Part:
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(np.asarray(origin, dtype=float))
    return Part(name, "", mesh=mesh)


def _cover_with_blocker(blocker_z: float) -> Assembly:
    asm = Assembly("cover_and_blocker")
    asm.add_part(_box("cover", (100.0, 100.0, 10.0), (0.0, 0.0, 0.0)))
    asm.add_part(_box("blocker", (100.0, 100.0, 10.0), (0.0, 0.0, blocker_z)))
    return asm


def _removal(length: float = 60.0) -> GrippingAndRemovingOperation:
    return GrippingAndRemovingOperation(
        {
            "part_name": "cover",
            "direction": [0.0, 0.0, 1.0],
            "path": np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0 + length]]
            ),
            "ray_bundle_required": False,  # cover-like: the actual-mesh checks decide
        }
    )


def test_blocker_two_mm_above_a_cover_makes_the_direction_infeasible():
    op = _removal()

    assert op.check_feasability(_cover_with_blocker(blocker_z=12.0)) is False
    assert (
        op.data["last_feasibility_debug"]["reason"]
        == "single_step_translation_collision"
    )


def test_blocker_beyond_the_first_step_is_caught_by_the_stepped_path():
    op = _removal()

    assert op.check_feasability(_cover_with_blocker(blocker_z=40.0)) is False
    assert (
        op.data["last_feasibility_debug"]["reason"]
        == "removal_path_stepped_translation_collision"
    )


def test_free_cover_stays_feasible():
    op = _removal()

    assert op.check_feasability(_cover_with_blocker(blocker_z=500.0)) is True
