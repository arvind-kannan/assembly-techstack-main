"""The product-model type (#273): a product's model XML and its product folder, found by one loader rule.

The rule: the path a graph export's provenance names, else the single XML at the folder root, else the
single XML in the folder's `xml` subfolder (the bolt-and-nut benchmark's layout). XML files at the root are
never skipped: several of them are an error, as is a folder with none in either place. Mesh and contact
paths in the XML resolve against the XML's own folder, as `Assembly.load_from_xml` resolves them.

The gym's env builder, the graph export and the Product explorer load products only through this type. The
MRTAMP planning request carries one, and its planners, plan functions and cell simulations build every
snapshot at a product state from it (#232). It wraps the snapshot and assembly helpers of `motion_planning.common` and does not
replace the core's cell-assembly snapshot (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import Cell_Assembly_Snapshot, CellState

from motion_planning.common import assembly_of, snapshot_at

XML_SUBFOLDER = "xml"


class ProductModelError(ValueError):
    """A product folder whose product model the loader rule cannot find."""


@dataclass(frozen=True)
class ProductModel:
    """The product model of a product folder: its XML file and the folder it belongs to."""

    folder: Path
    xml_path: Path

    @classmethod
    def find(
        cls, folder: str | Path, provenance_xml: str | None = None
    ) -> ProductModel:
        """The loader rule. `provenance_xml` is the path a graph export's provenance names, relative to
        `folder`; it is taken as named, also when the file is not there (the export loader warns about that)."""
        folder = Path(folder)
        if provenance_xml is not None:
            return cls(folder, folder / provenance_xml)
        root = sorted(folder.glob("*.xml"))
        subfolder = sorted((folder / XML_SUBFOLDER).glob("*.xml"))
        if len(root) == 1:
            return cls(folder, root[0])
        if not root and len(subfolder) == 1:
            return cls(folder, subfolder[0])
        raise ProductModelError(
            f"{folder}: expected exactly one assembly XML at the folder root, or none there and exactly "
            f"one in its {XML_SUBFOLDER}/ subfolder; found {[x.relative_to(folder).as_posix() for x in root + subfolder]}"
        )

    @property
    def relative_xml(self) -> str:
        """The XML's path inside the product folder, with forward slashes (as a provenance header records it)."""
        return self.xml_path.relative_to(self.folder).as_posix()

    @property
    def xml(self) -> str:
        """The XML text of the whole product."""
        return self.xml_path.read_text(encoding="utf-8")

    def assembly(self) -> Assembly:
        """The assembly of the whole product."""
        return Assembly.load_from_xml(str(self.xml_path))

    def snapshot_at(
        self, product_state: tuple[str, ...], cell_state: CellState | None = None
    ) -> Cell_Assembly_Snapshot:
        """The cell-assembly snapshot at `product_state`: the product model reduced to its parts, with
        `cell_state` (default: no holds, no mounted tools)."""
        return snapshot_at(self.xml, product_state, cell_state or CellState())

    def assembly_of(self, snapshot: Cell_Assembly_Snapshot) -> Assembly:
        """The snapshot's assembly, with mesh and contact paths resolved against the XML's folder."""
        return assembly_of(snapshot, self.xml_path.parent)
