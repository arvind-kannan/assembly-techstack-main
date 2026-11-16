"""
Battery replacement product model: builds model/battery_replacement.xml (+ _data/ contact files) from the meshes
of make_model_meshes.py. Geometric connections by contact detection; one screw connection per lid screw into
the tray (the slot nut is part of the tray). All paths in the XML are relative to model/, so load it from there.

Run from the repo root with the repo's venv (core model on the path, see tests/conftest.py):
    python cases/battery_replacement/make_model.py
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "crc_diss_assembly_dataclasses" / "src"))

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.part import Part

NAME = "battery_replacement"
MODEL_DIR = Path(__file__).parent / "model"
# Same placeholder as the angle grinder model; the real drive of the M4x16 is measured in
# "Design and print the screw nest plate".
SCREW_HEAD = "H5"
DENSITY = {
    "tray": 2700.0,
    "lid": 1400.0,
    "module": 1240.0,
    "lid_screw": 7850.0,
}  # alu, PVC, PLA, steel


def main():
    os.chdir(
        MODEL_DIR
    )  # mesh and contact paths are stored as given, i.e. relative to model/
    parts = [
        Part(
            f.stem,
            f"{NAME}/{f.name}",
            density=next(d for k, d in DENSITY.items() if f.stem.startswith(k)),
        )
        for f in sorted(Path(NAME).glob("*.stl"))
    ]
    asm = Assembly(NAME, parts=parts)
    # Assembly.find_contacts checks all 1275 pairs (hours); only parts whose bounding boxes touch can be in contact.
    tol = asm.contact_tollerance
    for a, b in itertools.combinations(parts, 2):
        if np.all(a.mesh.get_min_bound() - tol <= b.mesh.get_max_bound()) and np.all(
            b.mesh.get_min_bound() - tol <= a.mesh.get_max_bound()
        ):
            print(f"contacts {a.name} - {b.name}", flush=True)
            asm.check_pair_for_contact_new(
                (a, b),
                contact_tollerance=tol,
                dot_product_threshold=asm.dot_product_threshold,
            )
    for p in parts:
        if p.name.startswith("lid_screw"):
            asm.add_connection(Screw_Connection(p.name, "tray", SCREW_HEAD))
    asm.save_as_xml(f"{NAME}.xml")
    print(
        f"{len(parts)} parts, {len(asm.connections)} connections -> {MODEL_DIR / NAME}.xml"
    )


if __name__ == "__main__":
    main()
