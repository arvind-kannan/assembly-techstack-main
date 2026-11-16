"""AT 282 blast radius, second level: the sequence graph of each product (screws collapsed), nodes, edges, goals, time.

Run: uv run --group dev python .claude/log/AT-282/graph_blast_radius.py <out.json>
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

PRODUCTS = {
    "emo_sr_v2": ROOT / "emo_sr_v2",
    "angle_grinder": ROOT / "cases/angle_grinder/model",
    "battery_replacement": ROOT / "cases/battery_replacement/model",
}
out = {}
for name, folder in PRODUCTS.items():
    t0 = time.perf_counter()
    g = DisassemblyGraphAgent(build_env(folder)).build_graph(
        verbose=False, collapse_screw_groups=True
    )
    out[name] = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "goals": sum(1 for _, d in g.nodes(data=True) if d.get("is_goal")),
        "removals": sorted({s["target_part"] for _, _, s in g.edges(data=True)}),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        name,
        out[name]["nodes"],
        out[name]["edges"],
        out[name]["goals"],
        out[name]["seconds"],
        flush=True,
    )
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
