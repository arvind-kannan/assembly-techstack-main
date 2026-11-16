# PROTOTYPE (#78, throwaway): three layouts of the product-state viewer and editor on one page.
"""Build `viewer.html` from `template_split.html` (half/half layout of mockup.svg) or, with --template template.html, the three-variant page (`?variant=A|B|C`).

Question the prototype answers (#78): what is shown for a product state, how does the user move through
the sequence graph, how do the editing functions work, and does one framing hold for every state?

  A  Stepper       3D first; the actions of the current state beside it; the path as a strip below.
  B  Graph first   the sequence graph is the main view; click any state to jump; small 3D pane.
  C  Parts editor  part list with base/target roles (edits product_setup.json, stubbed); 3D; inspector.

Data comes from the product folder through !28's `build_env` (live, needs the gym's dependencies), or from a
page that `examples/build_disassembly_ui.py` wrote earlier (`--from-html`, no dependencies, EMO only).

    python examples/prototype_viewer/build_prototype_viewer.py --product emo_sr_v2
    python examples/prototype_viewer/build_prototype_viewer.py --from-html examples/runs/disassembly_graph/ui.html
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def data_live(product_folder: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "examples"))
    from build_disassembly_ui import build_page_data

    from disassembly_pomdp.env_builder import build_env, load_product
    from disassembly_pomdp.graph_extraction import DisassemblyGraphAgent

    agent = DisassemblyGraphAgent(build_env(product_folder))
    graph = agent.build_graph(verbose=True)
    agent.compute_inspectable_parts(verbose=True)
    data = build_page_data(graph, agent.base_assembly)
    _, base, target = load_product(product_folder)
    data["setup"] = {"base_parts": base, "target_parts": target}
    data["product"] = product_folder.name
    return data


def data_from_html(page: Path) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", page.read_text(encoding="utf-8"), re.S)
    data = json.loads(m.group(1))
    setup_file = (
        ROOT / "emo_sr_v2" / "product_setup.json"
    )  # the cached page is the EMO one
    data["setup"] = (
        json.loads(setup_file.read_text(encoding="utf-8"))
        if setup_file.exists()
        else {}
    )
    data["product"] = "emo_sr_v2 (cached page)"
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=Path, help="product folder (live build)")
    ap.add_argument(
        "--from-html", type=Path, help="reuse the data of an existing ui.html"
    )
    ap.add_argument(
        "--template",
        default="template_split.html",
        help="template.html = the three-variant version",
    )
    ap.add_argument("--out", type=Path, default=HERE / "viewer.html")
    args = ap.parse_args()
    if bool(args.product) == bool(args.from_html):
        ap.error("give exactly one of --product / --from-html")
    data = data_live(args.product) if args.product else data_from_html(args.from_html)
    html = (
        (HERE / args.template)
        .read_text(encoding="utf-8")
        .replace("__DATA_JSON__", json.dumps(data))
    )
    args.out.write_text(html, encoding="utf-8")
    print(
        f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(data['nodes'])} states)"
    )
    print(f"Open {args.out}")


if __name__ == "__main__":
    main()
