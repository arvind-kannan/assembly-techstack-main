# examples/run_product_explorer.py
"""
Start the Product explorer: a local server, used in the browser (#75, #134).

Usage:
    python examples/run_product_explorer.py                                   # load a product folder in the UI
    python examples/run_product_explorer.py --product emo_sr_v2
    python examples/run_product_explorer.py --product cases/battery_replacement/model --port 8766

Exports go to a free output folder; the default offered is runs/product_explorer/<product> (git-ignored).
"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "crc_diss_assembly_dataclasses" / "src"):
    sys.path.insert(0, str(p))

import uvicorn

from product_explorer.server import create_app
from product_explorer.session import ExplorerSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=Path, help="product folder to load at start")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    session = ExplorerSession()
    if args.product:
        session.load(args.product.resolve())
    app = create_app(session, default_out_dir=ROOT / "runs" / "product_explorer")
    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    print(f"Product explorer on {url} (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
