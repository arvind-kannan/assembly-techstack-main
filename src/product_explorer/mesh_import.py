"""Mesh folder import of the Product explorer (#243): a mesh folder becomes a product model, and the first
save of it becomes a product folder.

Everything about meshes and contacts lives here, the session only runs it. Contact detection is the core's
`Assembly.check_pair_for_contact_new`, called pair by pair so the import can report progress and be cancelled.
Meshes are taken as given: millimetres, one product frame, no unit conversion.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part, load_parts_from_folder

from disassembly_pomdp.env_builder import PRODUCT_SETUP_FILE
from mrtamp.screw_groups import parts_without_position
from product_explorer.product_folder import ProductFolderError, save_product_folder

MESH_ENDINGS = (".stl", ".STL", ".obj", ".OBJ")
SMALL_PRODUCT_MM = 5.0
LARGE_PRODUCT_MM = 5000.0


class ImportCancelled(Exception):
    """The product author cancelled the import; nothing of it is kept."""


@dataclass(frozen=True)
class ImportRequest:
    folder: Path
    depth: int
    name: str
    tolerance: float
    angle: float


def check_request(
    folder: str | Path,
    depth: int,
    name: str | None,
    tolerance: float,
    angle: float,
) -> tuple[ImportRequest, list[Part]]:
    """Validate an import request and list its parts. Raises `ProductFolderError` (a user error)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ProductFolderError(f"{folder}: not a folder")
    if depth < 0:
        raise ProductFolderError("The subfolder depth cannot be negative")
    if tolerance <= 0:
        raise ProductFolderError("The contact tolerance must be positive")
    if not -1.0 <= angle <= 1.0:
        raise ProductFolderError("The contact angle is a dot product between -1 and 1")
    name = _file_name((name or "").strip() or folder.name)
    parts = load_parts_from_folder(str(folder), depth, MESH_ENDINGS)
    if not parts:
        raise ProductFolderError(
            f"{folder}: no meshes (STL or OBJ) found at subfolder depth {depth}"
        )
    names = [p.name for p in parts]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ProductFolderError(
            f"Part names must be unique, but several meshes are named {', '.join(twice)}"
        )
    return ImportRequest(folder, depth, name, tolerance, angle), parts


def detect_product(
    request: ImportRequest,
    parts: list[Part],
    progress: Callable[[int, int], None],
    cancelled: Callable[[], bool],
) -> Assembly:
    """The product model of the parts with its geometric connections. `progress(done, total)` counts part
    pairs; `cancelled()` is polled between pairs and raises `ImportCancelled`."""
    assembly = Assembly(
        request.name,
        parts=parts,
        contact_tollerance=request.tolerance,
        dot_product_threshold=request.angle,
    )
    pairs = list(itertools.combinations(assembly.parts.values(), 2))
    progress(0, len(pairs))
    for done, pair in enumerate(pairs, 1):
        if cancelled():
            raise ImportCancelled()
        assembly.check_pair_for_contact_new(
            pair,
            contact_tollerance=request.tolerance,
            dot_product_threshold=request.angle,
        )
        progress(done, len(pairs))
    return assembly


def mesh_warnings(assembly) -> list[str]:
    """The sanity warning for parts without a usable mesh, shared by import and open (#266): one message naming
    each part once. A usable mesh is one that gives the part a position by B1's rule."""
    return _no_mesh_warning(parts_without_position(assembly))


def _no_mesh_warning(unusable: list[str]) -> list[str]:
    return [f"Parts with no usable mesh: {', '.join(unusable)}"] if unusable else []


def import_warnings(assembly) -> list[str]:
    """Sanity warnings of an imported product: parts without a usable mesh, wrong units or frames, and parts
    nothing touches."""
    unusable = parts_without_position(assembly)
    warnings = _no_mesh_warning(unusable)
    boxes = [
        p.mesh.get_axis_aligned_bounding_box()
        for n, p in assembly.parts.items()
        if n not in unusable
    ]
    if not boxes:
        return warnings
    lo = np.min([b.get_min_bound() for b in boxes], axis=0)
    hi = np.max([b.get_max_bound() for b in boxes], axis=0)
    size = float(np.max(hi - lo))
    if size < SMALL_PRODUCT_MM:
        warnings.append(
            f"The product is only {size:.3g} mm across. Meshes are taken as millimetres: "
            "are they in metres?"
        )
    elif size > LARGE_PRODUCT_MM:
        warnings.append(
            f"The product is {size / 1000:.3g} m across. Meshes are taken as millimetres: "
            "are they in micrometres or in another frame?"
        )
    touching = {
        name for c in assembly.connections for name in (c.part_name1, c.part_name2)
    }
    alone = [n for n in assembly.parts if n not in touching]
    if alone:
        warnings.append(f"Parts with no contacts: {', '.join(alone)}")
    return warnings


def write_product_folder(
    assembly: Assembly, setup: dict, target: Path, name: str, assemblies: list
) -> Path:
    """First save of an imported product into `target`: `<name>.xml`, the meshes copied to `<name>/`, the
    contact data in `<name>_data/` and `product_setup.json`. Returns the XML path.

    The meshes of `assembly` and of every assembly in `assemblies` (the edit history's) are repointed to
    the copies, so no later save refers back to the import folder."""
    target = Path(target)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ProductFolderError(f"{target}: the target folder must be new or empty")
    xml = target / f"{name}.xml"
    mesh_dir = target / name
    mesh_dir.mkdir(parents=True)
    copies = {}
    for part in assembly.parts.values():
        copy = (
            mesh_dir / Path(part.mesh_path).name
        )  # unique: part names are the file stems
        shutil.copyfile(part.mesh_path, copy)
        copies[part.mesh_path] = str(copy)
    for each in [assembly, *assemblies]:
        for part in each.parts.values():
            part.mesh_path = copies.get(part.mesh_path, part.mesh_path)
    save_product_folder(assembly, xml)
    (target / PRODUCT_SETUP_FILE).write_text(
        json.dumps(setup, indent=2) + "\n", encoding="utf-8"
    )
    return xml


def _file_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not cleaned:
        raise ProductFolderError("The product needs a name")
    return cleaned
