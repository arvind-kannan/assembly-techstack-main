from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/hans/Uni/Bachelorarbeit/pomdp_disassembly_gym")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import Screw_Connection
from diss_assembly_dataclasses.part import load_parts_from_folder


def build_assembly_from_stl_dir(
    assembly_dir: Path,
    assembly_name: str,
    screw_specs: list[tuple[str, str, str]] | None = None,
    contact_tolerance: float = 1.0,
    dot_product_threshold: float = -0.99,
) -> Assembly:
    """
    Build an Assembly from a directory containing only STL files.

    screw_specs format:
        [
            ("03_s1_8", "00_st_8", "H5"),
            ("04_s2_8", "00_st_8", "H5"),
        ]
    meaning:
        screw part_name1 is screwed into base part_name2 with screw_head
    """
    if not assembly_dir.is_dir():
        raise FileNotFoundError(f"Assembly directory not found: {assembly_dir}")

    # 1) Load all STL files as Part objects
    parts = load_parts_from_folder(
        str(assembly_dir), depth=0, file_endings=(".stl", ".STL")
    )
    if not parts:
        raise RuntimeError(f"No STL files found in: {assembly_dir}")

    # 2) Build assembly
    asm = Assembly(
        assembly_name,
        parts=parts,
        contact_tollerance=contact_tolerance,
        dot_product_threshold=dot_product_threshold,
    )

    # 3) Automatically detect geometric contacts
    asm.find_contacts()

    # 4) Add screw semantics manually
    for screw_part, base_part, screw_head in screw_specs or []:
        if asm.get_part_by_name(screw_part) is None:
            raise ValueError(f"Screw part not found: {screw_part}")
        if asm.get_part_by_name(base_part) is None:
            raise ValueError(f"Base part not found: {base_part}")

        asm.add_connection(Screw_Connection(screw_part, base_part, screw_head))

    return asm


def export_assembly(
    asm: Assembly,
    out_xml: Path,
) -> None:
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    asm.save_as_xml(str(out_xml))
    print(f"Saved XML to: {out_xml}")
    print(f"Saved geometric npy files to: {out_xml.with_suffix('')}_data/")


def main():
    base_dir = Path(
        "/home/hans/Uni/Bachelorarbeit/pomdp_disassembly_gym/tests/internal_cases/Modell 2-Plannungssoftware/Stl"
    )

    # Example: Assembly 2
    asm2_dir = base_dir / "Angle Grinder"
    asm2 = build_assembly_from_stl_dir(
        assembly_dir=asm2_dir,
        assembly_name="Angle Grinder",
        screw_specs=[
            # Example only — replace with your real screw relations
        ],
        contact_tolerance=1.0,
        dot_product_threshold=-0.99,
    )
    export_assembly(asm2, asm2_dir / "Angle Grinder.xml")


if __name__ == "__main__":
    main()
