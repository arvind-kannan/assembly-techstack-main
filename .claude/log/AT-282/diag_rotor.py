import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses/src"):
    sys.path.insert(0, str(p))
from disassembly_pomdp.env_builder import load_product
from motion_planning.removal_planning import plan_removal

asm, _b, _t = load_product(ROOT / "emo_sr_v2")
for s in ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]:
    asm.remove_part(asm.get_part_by_name(s))
plan = plan_removal(asm, "01_upper_lid", 5.0)
print("lid", plan.failure)
asm = plan.result
print(sorted(asm.parts))
plan = plan_removal(asm, "03_rotor_short", 5.0)
print(plan.failure)
print(plan.operation.data.get("last_feasibility_debug") if plan.operation else None)

import numpy as np

rotor_asm = asm
for d in (
    [0, 0, 1.0],
    [0.11090972531022349, -0.0024199940064082667, 0.9938275385903833],
):
    d = np.array(d) / np.linalg.norm(d)
    hit = None
    for k in range(1, 51):
        feas, contacts = rotor_asm.is_translation_feasible(
            "03_rotor_short",
            d * 5.0 * k,
            ignore_self=True,
            return_contacts=True,
            depth_threshold=2.5,
        )
        if not feas:
            depth = max(getattr(c, "penetration_depth", 0) for c in contacts)
            hit = (k * 5.0, len(contacts), round(float(depth), 2))
            break
    print(
        "axis" if d[0] == 0 else "tilt",
        d.round(3),
        "first infeasible (dist, n_contacts, depth):",
        hit,
    )
