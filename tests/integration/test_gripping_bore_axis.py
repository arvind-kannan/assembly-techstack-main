"""The EMO rotor is planned along its bore axis, stepped path checked (AT 288)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.geom
def test_emo_rotor_is_planned_along_its_bore_axis_with_the_stepped_path_checked():

    from disassembly_pomdp.env_builder import load_product
    from motion_planning.removal_planning import plan_removal

    root = Path(__file__).resolve().parents[2]
    asm, _b, _t = load_product(root / "emo_sr_v2")
    for s in ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]:
        asm.remove_part(asm.get_part_by_name(s))
    lid = plan_removal(asm, "01_upper_lid", 5.0)
    assert lid.failure is None

    plan = plan_removal(lid.result, "03_rotor_short", 5.0)

    assert plan.failure is None
    assert np.allclose(plan.operation.data["direction"], [0.0, 0.0, 1.0])
    dbg = plan.operation.data["last_feasibility_debug"]
    assert dbg["stepped_path_checked"] is True
    assert dbg["stepped_path_feasible"] is True
    assert dbg.get("reason") is None
