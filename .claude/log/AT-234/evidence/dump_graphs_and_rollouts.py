"""Dump extracted graphs + seeded rollout info for EMO SR V2 and bolt-and-nut (AT-234 baseline)."""

import json
import sys
from pathlib import Path

import numpy as np

import disassembly_pomdp  # noqa: F401  path setup
from disassembly_pomdp.env_builder import build_env
from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent

ROOT = Path.cwd()
CASES = {
    "emo": ROOT / "emo_sr_v2",
    "bolt_nut": ROOT / "benchmark_cases" / "benchmark_bolt_and_nut_case",
}


def ser(x):
    if isinstance(x, dict):
        return {str(k): ser(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, set):
        return sorted((ser(v) for v in x), key=repr)
    if isinstance(x, (list, tuple)):
        return [ser(v) for v in x]
    if isinstance(x, np.ndarray):
        return (
            np.round(x.astype(float), 6).tolist() if x.dtype.kind in "fiub" else repr(x)
        )
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, str) or x is None:
        return x
    return type(x).__name__


def make_env(name, folder):
    if name != "bolt_nut":
        return build_env(folder)
    import os

    from disassembly_pomdp.product_model import ProductModel

    model = ProductModel.find(folder)
    prev = os.getcwd()
    os.chdir(model.xml_path.parent)
    try:
        asm = model.assembly()
    finally:
        os.chdir(prev)
    from disassembly_pomdp.env_builder import Cell

    return build_env(
        folder,
        cell=Cell(screwdrivers=()),
        assembly=asm,
        base_parts=["part_j_elbow_pipe"],
        target_parts=["part_l_cover_lid"],
    )


out = {}
for name, folder in CASES.items():
    res = {}
    for collapse in (False, True):
        g = DisassemblyGraphAgent(make_env(name, folder)).build_graph(
            verbose=False, collapse_screw_groups=collapse
        )
        res[f"graph_collapse_{collapse}"] = {
            "nodes": sorted(
                [[str(n), ser(d)] for n, d in g.nodes(data=True)], key=lambda t: t[0]
            ),
            "edges": sorted(
                [[str(u), str(v), ser(d)] for u, v, d in g.edges(data=True)], key=repr
            ),
        }
    env = make_env(name, folder)
    steps = []
    for seed in range(3):
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(40):
            a = np.array([rng.integers(n) for n in env.action_space.nvec])
            _obs, r, term, trunc, info = env.step(a)
            steps.append(
                {
                    "a": a.tolist(),
                    "r": ser(r),
                    "term": term,
                    "trunc": trunc,
                    "info": ser(info),
                }
            )
            if term or trunc:
                break
    res["rollout"] = steps
    out[name] = res
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
print("ok")
