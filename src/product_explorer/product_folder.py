"""The product folder adapter of the Product explorer (#238): portable paths on load and save.

The product model is read and written only through the core's `Assembly.load_from_xml` and
`Assembly.save_as_xml`. Everything about paths lives here: a path in the product model XML is resolved
against the XML's folder, and one that does not resolve is repaired by file name next to the XML (meshes
also in the product's mesh folder `<stem>/`, contact data in `<stem>_data/`, both next to the XML). The XML
itself is found by the product-model type's loader rule (`ProductModel.find`, #273). Save hands the core
the real XML path, so it writes every path relative to the folder with forward slashes.

Loading writes nothing to the product folder; only `save_product_folder` does.
"""

from __future__ import annotations

import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.env_builder import PRODUCT_SETUP_FILE
from disassembly_pomdp.product_model import ProductModel, ProductModelError

_DRIVE_OR_ROOT = re.compile(r"^([A-Za-z]:[\\/]|[\\/])")
_CONTACT_FILES = ("points_filename", "normals_filename", "distances_filename")


class ProductFolderError(ValueError):
    """A product folder that cannot be loaded (shown to the user)."""


def load_product_folder(folder: Path) -> tuple[Path, Assembly, list[str], list[str]]:
    """The product model XML, the assembly and the base and target parts of a product folder."""
    folder = Path(folder)
    try:
        xml = ProductModel.find(folder).xml_path
    except ProductModelError as e:
        raise ProductFolderError(str(e)) from e
    setup_path = folder / PRODUCT_SETUP_FILE
    if not setup_path.exists():
        raise FileNotFoundError(f"{setup_path} not found")
    setup = json.loads(setup_path.read_text(encoding="utf-8"))

    tree = ET.parse(xml)
    stem = xml.stem
    for part in tree.getroot().findall("part"):
        name = part.attrib["name"]
        part.set(
            "mesh_path",
            _repair(
                part.attrib["mesh_path"], xml.parent, [xml.parent / stem], name, "mesh"
            ),
        )
    for conn in tree.getroot().findall("connection"):
        label = f"{conn.attrib.get('part_name1')}-{conn.attrib.get('part_name2')}"
        for attr in _CONTACT_FILES:
            if attr in conn.attrib:
                conn.set(
                    attr,
                    _repair(
                        conn.attrib[attr],
                        xml.parent,
                        [xml.parent / f"{stem}_data"],
                        label,
                        "contact data",
                    ),
                )

    # The core reads the repaired copy; its paths are all absolute, so the copy's own folder does not matter.
    with tempfile.TemporaryDirectory() as tmp:
        repaired = Path(tmp) / xml.name
        tree.write(repaired, encoding="utf-8")
        assembly = Assembly.load_from_xml(str(repaired))
    assembly.xml_path = str(xml)
    return (
        xml,
        assembly,
        list(setup.get("base_parts", [])),
        list(setup.get("target_parts", [])),
    )


def save_product_folder(assembly: Assembly, xml: Path) -> None:
    """Write the product model XML (and the contact data it refers to) in place, with portable paths."""
    assembly.save_as_xml(str(Path(xml).resolve()))


def _repair(
    value: str, xml_dir: Path, search_dirs: list[Path], label: str, kind: str
) -> str:
    """An absolute path to the file `value` names: next to the XML, else by file name in `search_dirs`."""
    normalized = value.replace("\\", "/")
    is_absolute = bool(_DRIVE_OR_ROOT.match(value))
    relative = xml_dir / normalized
    if not is_absolute and relative.is_file():
        return str(relative.resolve())
    name = Path(normalized).name
    for directory in search_dirs:
        if (directory / name).is_file():
            return str((directory / name).resolve())
    if (
        is_absolute and Path(value).is_file()
    ):  # as the core reads it: the path exists here as written
        return str(Path(value).resolve())
    where = " or ".join(str(d) for d in [xml_dir, *search_dirs])
    raise ProductFolderError(
        f"{label}: {kind} file {name} not found (looked in {where})"
    )
