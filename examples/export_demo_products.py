# examples/export_demo_products.py
"""
Sequence graph exports of the two demo products for the MRTAMP app: the angle grinder and the battery
replacement product (`cases/<product>/model/`, product setup in `product_setup.json` there).

The graph is extracted for a full teardown: the assembly demo runs start from the base parts alone, a product
state only a full-teardown graph is sure to have.

Goes through the Product explorer's own export (load the product folder, Generate graph, export), so the
files have the one export schema: `sequence_graph.json` (complete by reversal) plus URDFs and `manifest.json`.
Load one with its cell file, `cases/transformer_cell.json` for the real cell or `cases/transformer_cell_simulation.json`
for the cell simulation (both robots may screw in):

    load_export("examples/runs/demo_products/angle_grinder/sequence_graph.json", cell_file="cases/transformer_cell.json")

The angle grinder's reassembly baseline (AT 270) is the other export: the default depth (stop at the disassembled
states), the one a reassembly starts from. It is stored under `tests/test_products/angle_grinder_baseline/`, the
graph and the manifest only (the meshes and URDFs are in the product folder), so `provenance.product_folder`
resolves inside the repository. Regenerate it after any change to the grinder's model or product setup.

Usage:
    python examples/export_demo_products.py                  # both products, about 2 minutes
    python examples/export_demo_products.py angle_grinder    # one product
    python examples/export_demo_products.py --baseline       # the angle grinder reassembly baseline
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from disassembly_pomdp.graph_extraction import (  # noqa: E402
    DISASSEMBLED_STATES,
    FULL_TEARDOWN,
)
from product_explorer.session import ExplorerSession  # noqa: E402

PRODUCTS = ("angle_grinder", "battery_replacement")
OUT_DIR = ROOT / "examples" / "runs" / "demo_products"
BASELINE_DIR = ROOT / "tests" / "test_products" / "angle_grinder_baseline"


def export_demo_product(
    product: str, out_dir: Path, extraction_depth: str = FULL_TEARDOWN
) -> Path:
    """Generate and export the sequence graph of `cases/<product>/model`; returns the `sequence_graph.json`.

    The demo runs assemble from scratch, so the default is a full teardown."""
    session = ExplorerSession()
    session.load(ROOT / "cases" / product / "model")
    session.generate(extraction_depth).join()
    if session.job.error:
        raise RuntimeError(f"{product}: Generate graph failed: {session.job.error}")
    return session.export(out_dir) / "sequence_graph.json"


def export_reassembly_baseline(out_dir: Path = BASELINE_DIR) -> Path:
    """The angle grinder's default-depth export (AT 270) without the meshes and URDFs; returns the `sequence_graph.json`."""
    graph_file = export_demo_product("angle_grinder", out_dir, DISASSEMBLED_STATES)
    for subfolder in ("mesh", "urdf"):
        shutil.rmtree(out_dir / subfolder)
    return graph_file


def main() -> None:
    if sys.argv[1:] == ["--baseline"]:
        print(f"angle_grinder baseline: {export_reassembly_baseline()}")
        return
    for product in sys.argv[1:] or PRODUCTS:
        print(f"{product}: {export_demo_product(product, OUT_DIR / product)}")


if __name__ == "__main__":
    main()
