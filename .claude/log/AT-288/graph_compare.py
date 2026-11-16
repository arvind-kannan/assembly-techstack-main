"""AT 288: sequence graph of each product (screws collapsed): counts, time, every removal direction.

Run from the repo root: python .claude/log/AT-288/graph_compare.py <out.json> [product ...]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses/src"):
    sys.path.insert(0, str(p))
from disassembly_pomdp.env_builder import build_env  # noqa: E402
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent  # noqa: E402

ALL = {
    "emo_sr_v2": ROOT / "emo_sr_v2",
    "angle_grinder": ROOT / "cases/angle_grinder/model",
    "battery_replacement": ROOT / "cases/battery_replacement/model",
}
names = sys.argv[2:] or list(ALL)
out = {}
for name in names:
    t0 = time.perf_counter()
    g = DisassemblyGraphAgent(build_env(ALL[name])).build_graph(
        verbose=False, collapse_screw_groups=True
    )
    edge_keys = sorted({k for _, _, s in g.edges(data=True) for k in s})
    dirs = {}
    for _, _, s in g.edges(data=True):
        d = s.get("direction")
        key = tuple(round(float(x), 3) for x in d) if d is not None else None
        dirs.setdefault(s["target_part"], set()).add(key)
    out[name] = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "goals": sum(1 for _, d in g.nodes(data=True) if d.get("is_goal")),
        "directions": {
            k: sorted([list(x) for x in v if x is not None])
            for k, v in sorted(dirs.items())
        },
        "edge_keys": edge_keys,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        name,
        {k: out[name][k] for k in ("nodes", "edges", "goals", "seconds")},
        flush=True,
    )
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
