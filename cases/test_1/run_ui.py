"""Prepare test_1's contact cache and open the Product explorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]


def prepare_contacts() -> Path:
    """Materialize the bundled numeric contacts; no mesh detection or download is needed."""
    model = CASE / "model"
    cache = model / "test_1_data"
    cache.mkdir(exist_ok=True)
    contacts = json.loads((CASE / "contacts.json").read_text(encoding="utf-8"))
    for name, values in contacts.items():
        np.save(cache / name, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    model = prepare_contacts()
    if args.prepare_only:
        print(model)
        return
    sys.path.insert(0, str(ROOT / "examples"))
    from run_product_explorer import main as run_explorer

    sys.argv = [sys.argv[0], "--product", str(model), "--port", str(args.port)]
    if args.no_browser:
        sys.argv.append("--no-browser")
    run_explorer()


if __name__ == "__main__":
    main()
