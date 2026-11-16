"""AT 282 blast radius: plan the removal of every non-screw part of each case, screws taken out first.

Run from the repo root: uv run --group dev python .claude/log/AT-282/blast_radius.py <out.json>
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses/src"):
    sys.path.insert(0, str(p))

from diss_assembly_dataclasses.assembly import Assembly  # noqa: E402

from motion_planning.removal_planning import plan_removal  # noqa: E402

CASES = {
    "angle_grinder": ROOT / "cases/angle_grinder/model/angle_grinder.xml",
    "battery_replacement": ROOT
    / "cases/battery_replacement/model/battery_replacement.xml",
}

out = {}
for case, xml in CASES.items():
    os.chdir(xml.parent)
    asm = Assembly.load_from_xml(xml.name)
    screws = {c.part_name1 for c in asm.connections if c.connection_type == "screw"}
    for name in screws:
        asm.remove_part(asm.get_part_by_name(name))
    out[case] = {}
    for name in sorted(asm.parts):
        t0 = time.perf_counter()
        plan = plan_removal(asm, name, 5.0)
        dt = time.perf_counter() - t0
        out[case][name] = {
            "feasible": plan.ok if hasattr(plan, "ok") else plan.failure is None,
            "seconds": round(dt, 3),
            "failure": None if plan.failure is None else plan.failure.reason,
        }
        print(case, name, out[case][name], flush=True)

Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
