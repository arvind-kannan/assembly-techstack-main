"""One exploring session of the Product explorer: the loaded product folder, its product setup, the
sequence graph, the current product state with the path that led there, and the recorded sequence.

All logic lives here; the server only maps requests onto it and the browser only renders the view.
The sequence graph comes from `build_env` and `DisassemblyGraphAgent`: sequential with interchangeable
screws collapsed by default, or optional parallel analysis with robot-limited stages; next steps are its edges. Rejection reasons are checked live for one picked part.

Edits to the product model (#240) all go through `_apply_edit`: it keeps a memento of the product model and
product setup before the edit, so undo and redo swap whole states. Any edit, undo or redo makes the sequence
graph stale and re-checks the available actions of the current product state on the edited model, without
extracting a graph. The part edits (#241) are the same kind: rename by rules file, material and density, delete,
and merge into a subassembly.

The cell file (#245) is optional; its screwdrivers and grippers are the tools of the available actions, and the
default cell is used without one. Choosing it is an edit of its own kind: undoable, and it makes the graph stale
(a graph remembers the fingerprint of the cell file and its travel-time table it was built with).
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
from diss_assembly_dataclasses.assembly import (
    o3d_merge_triangle_meshes,
    o3d_save_triangle_mesh,
)
from diss_assembly_dataclasses.connection import (
    Fixed_Connection,
    Geometric_Connection,
    Safety_Ring_Connection,
    Screw_Connection,
)
from diss_assembly_dataclasses.part import Part

from disassembly_pomdp.env_builder import (
    PRODUCT_SETUP_FILE,
    Cell,
    build_env,
    cell_from_resources,
)
from disassembly_pomdp.graph_export import (
    SCHEMA_VERSION,
    export_graph,
    files_repository_root,
    fingerprint_of_cell,
    make_provenance,
)
from disassembly_pomdp.graph_extraction import (
    DISASSEMBLED_STATES,
    DisassemblyGraphAgent,
    check_extraction_depth,
)
from disassembly_pomdp.graph_reversal import complete_by_reversal
from disassembly_pomdp.parallel_geometry import ParallelGeometrySettings
from disassembly_pomdp.product_model import ProductModel
from disassembly_pomdp.rendering import _PALETTE, HIGHLIGHT_COLOR
from disassembly_pomdp.screwing_capability import ScrewingCapability
from mrtamp.cell_file import load_cell_file
from mrtamp.screw_groups import groups_naming_unknown_parts, parts_without_position
from product_explorer.graph_analysis import analysis_view, export_analysis
from product_explorer.mesh_import import (
    ImportCancelled,
    check_request,
    detect_product,
    import_warnings,
    mesh_warnings,
    write_product_folder,
)
from product_explorer.mesh_update import (
    apply_update,
    check_add,
    plan_update,
    redetect_pair,
    report_line,
)
from product_explorer.product_folder import (
    ProductFolderError,
    load_product_folder,
    save_product_folder,
)
from product_explorer.product_root import (
    ENV_VAR,
    InvalidProductRoot,
    ProductRoot,
    read_setting,
    resolve_product_root,
    write_setting,
)


class ExplorerError(Exception):
    """A request that does not fit the session state (shown to the user)."""


_UNKNOWN_SCREWS = "Screw group names parts that are not in the product: "


class Conflict(ExplorerError):
    """The request needs another state first (generate, store, wait for the job)."""


_NOT_A_FILE_NAME = re.compile(
    r'[\\/:*?"<>|]'
)  # what a part name must not hold if its mesh file is named by it


_DEFAULT_DENSITY, _DEFAULT_MATERIAL = (
    8000.0,
    "Generic",
)  # the core's defaults for a part

_REASON_TEXT = {
    "masked:base_part": "base part of the product setup",
    "masked:wrong_tool": "the tool does not fit",
    "masked:agent_busy": "the robot is busy",
    "masked:geometrically_infeasible": "geometrically blocked",
    "exec_gate_or_infeasible": "the removal path is not feasible",
}
_WHY_TEXT = {
    "empty_dof": "no free direction",
    "not_visible": "not visible along its free directions",
    "no_view_dirs": "no view direction",
    "head_mismatch": "screw head does not match",
}


_HAND_MADE = {"safety_ring": Safety_Ring_Connection, "fixed": Fixed_Connection}
_HAND_MADE_NAMES = tuple(_HAND_MADE)


@dataclass
class Job:
    """The background job (Generate graph, mesh folder import, or mesh update); read by the progress WebSocket."""

    running: bool = False
    kind: str = "graph"  # "graph", "import" or "update"
    expanded: int = 0  # import, update: part pairs checked
    discovered: int = 0  # import, update: part pairs in all
    error: str | None = None
    cancel_requested: bool = False
    cancelled: bool = False

    def as_dict(self) -> dict:
        return {
            "running": self.running,
            "kind": self.kind,
            "expanded": self.expanded,
            "discovered": self.discovered,
            "error": self.error,
            "cancelled": self.cancelled,
        }


@dataclass
class _Graph:
    graph: nx.MultiDiGraph
    agent: DisassemblyGraphAgent
    serial: int  # `ExplorerSession._serial` of the product model it was extracted from
    cell_fingerprint: str | None = (
        None  # of the cell file it was built with; None: default cell
    )
    node_ids: dict = field(default_factory=dict)  # product state (tuple) -> "s<i>"
    nodes: dict = field(default_factory=dict)  # "s<i>" -> product state
    edges: dict = field(default_factory=dict)  # "e<i>" -> (u, v, key)
    root: str = "s0"
    analysis: dict = field(default_factory=dict)

    @property
    def extraction_depth(self) -> str | None:
        """The depth the graph was extracted with (None for a graph that does not say)."""
        return self.graph.graph.get("extraction_depth")


@dataclass
class _Memento:
    """The product model and product setup as they were before (undo stack) or after (redo stack) one edit."""

    assembly: object
    setup: dict
    meshes: dict | None
    version: int
    renames: dict  # old name -> new name, as the edit made it
    cell: "CellChoice | None" = None
    merged: frozenset = frozenset()  # parts merged in this session (#244)
    geometry: int = 0  # mesh version: another number means other mesh content (#244)


@dataclass(frozen=True)
class CellChoice:
    """The cell file in use (#245): read once when chosen; its fingerprint tells graphs built with it apart."""

    path: Path
    fingerprint: str
    cell: Cell

    @property
    def name(self) -> str:
        return self.path.name


class ExplorerSession:
    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir  # None: the user's configuration directory (#257)
        self.lock = threading.RLock()
        self.job = Job()
        self.cell_choice: CellChoice | None = None  # None: the default cell
        self.folder: Path | None = (
            None  # None while the product is an import that was never saved
        )
        self.xml: Path | None = None
        self.name: str | None = None
        self.assembly = None
        self.warnings: list[
            str
        ] = []  # of the import, shown until another product is opened
        self._product_id = 0  # changes when another product is opened or imported
        self.setup = {"base_parts": [], "target_parts": []}
        self._graph: _Graph | None = None
        self._meshes: dict | None = None
        self.status_log: list[str] = []
        self._reset_history()
        self.path: list[str] = []
        self.recording = False
        self.recorded: list[str] = []

    # ------------------------------------------------------------ product folder
    def load(self, folder: str | Path) -> None:
        folder = Path(folder)
        self._need_idle()
        try:
            xml, assembly, base, target = load_product_folder(folder)
        except (ValueError, FileNotFoundError, OSError) as e:
            raise ExplorerError(str(e)) from e
        with self.lock:
            self.folder, self.xml, self.assembly = folder, xml, assembly
            self.name, self.warnings = xml.stem, mesh_warnings(assembly)
            self.setup = {"base_parts": base, "target_parts": target}
            self._open_product()

    def _open_product(self) -> None:
        """Opening or importing a product starts exploring and the edit history afresh."""
        self._product_id += 1
        self._graph = self._meshes = None
        self.path, self.recording, self.recorded = [], False, []
        self.cell_choice = None  # the default cell, like the rest of the edit history
        self._reset_history()

    def import_folder(
        self,
        folder: str | Path,
        depth: int = 0,
        name: str | None = None,
        tolerance: float = 1.0,
        angle: float = -0.99,
    ) -> threading.Thread:
        """Import a mesh folder as a new product: contact detection runs in a background thread, with progress
        in `self.job`. The open product, its edit history and its graph stay as they are until detection is
        done; cancelling or failing keeps them."""
        with self.lock:
            self._need_idle()
            try:
                request, parts = check_request(folder, depth, name, tolerance, angle)
            except ProductFolderError as e:
                raise ExplorerError(str(e)) from e
            self.job = job = Job(running=True, kind="import")
        thread = threading.Thread(
            target=self._import, args=(job, request, parts), daemon=True
        )
        thread.start()
        return thread

    def cancel(self) -> None:
        """Cancel the running import; its result is discarded."""
        with self.lock:
            if not self.job.running or self.job.kind != "import":
                raise Conflict("No import is running")
            self.job.cancel_requested = True

    def _import(self, job: Job, request, parts) -> None:
        def progress(done: int, total: int) -> None:
            job.expanded, job.discovered = done, total

        try:
            assembly = detect_product(
                request, parts, progress, lambda: job.cancel_requested
            )
            warnings = import_warnings(assembly)
            with self.lock:
                if job.cancel_requested:  # a cancel that came after the last pair
                    raise ImportCancelled()
                self.folder = self.xml = None
                self.name, self.assembly, self.warnings = (
                    request.name,
                    assembly,
                    warnings,
                )
                self.setup = {"base_parts": [], "target_parts": []}
                self._open_product()
                self._versions = self._version = (
                    1  # never saved: unsaved until the first save
                )
                job.running = False  # kept: a cancel from here on is refused
                self._log(
                    f"Imported {request.name}: {len(assembly.parts)} parts, "
                    f"{len(assembly.connections)} connections"
                )
                for warning in warnings:
                    self._log(f"Warning: {warning}")
        except ImportCancelled:
            job.cancelled = True
            self._log("Import cancelled")
        except Exception as e:  # shown in the UI; the open product stays
            job.error = f"{type(e).__name__}: {e}"
            self._log(f"Import failed: {job.error}")
        finally:
            job.running = False

    def set_setup(self, base_parts: list[str], target_parts: list[str]) -> None:
        self._need_product()
        unknown = sorted((set(base_parts) | set(target_parts)) - set(self.part_names))
        if unknown:
            raise ExplorerError(f"Unknown parts: {', '.join(unknown)}")
        if set(base_parts) & set(target_parts):
            raise ExplorerError("A part cannot be base part and target part at once")
        new = {"base_parts": list(base_parts), "target_parts": list(target_parts)}

        def edit(_assembly, setup: dict) -> dict:
            setup.update(new)
            return {}

        with self.lock:
            if new != self.setup:
                self._apply_edit(edit, model=False)

    def set_cell(self, path: str | Path) -> None:
        """Use the cell file at `path` for the available actions: one undoable edit. A file that cannot be read
        or is unusable is a user error and the cell in use stays."""
        self._need_product()
        choice = _read_cell_file(Path(path))

        def edit(_assembly, _setup: dict) -> dict:
            return {}

        with self.lock:
            if choice != self.cell_choice:
                self._apply_edit(edit, model=False, cell=choice)

    def save(self, target: str | Path | None = None) -> None:
        """Write the product model XML in place, with portable paths (#238). Besides `store` and the mesh file of a merge (#241), only
        this writes to the folder.

        The first save of an imported product needs a `target` folder (new or empty) and writes the whole
        product folder: the product model, the meshes copied into it and the product setup (#243)."""
        self._need_product()
        with self.lock:
            self._need_idle()
            try:
                if self.folder is None:
                    if not target:
                        raise ExplorerError("The first save needs a target folder")
                    self._first_save(Path(target))
                else:
                    save_product_folder(self.assembly, self.xml)
            except (ProductFolderError, OSError) as e:
                raise ExplorerError(str(e)) from e
            self._saved_version = self._version

    def _first_save(self, target: Path) -> None:
        history = [m.assembly for m in (*self._undo, *self._redo)]
        xml = write_product_folder(
            self.assembly, self.setup, target, self.name, history
        )
        self.folder, self.xml = target, xml
        self._meshes = None  # the meshes are now the copies

    def store(self) -> None:
        if self.folder is None:
            raise Conflict("Save the product first: it has no folder yet")
        path = self.folder / PRODUCT_SETUP_FILE
        # Keep the fields the explorer does not edit (e.g. the stuck-part rules).
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(
            json.dumps({**stored, **self.setup}, indent=2) + "\n", encoding="utf-8"
        )

    @property
    def part_names(self) -> list[str]:
        return (
            sorted(p.name for p in self.assembly.parts.values())
            if self.assembly
            else []
        )

    def setup_stored(self) -> bool:
        if self.folder is None:
            return False
        try:
            stored = json.loads(
                (self.folder / PRODUCT_SETUP_FILE).read_text(encoding="utf-8")
            )
        except (TypeError, OSError, ValueError):
            return False
        return [stored.get("base_parts", []), stored.get("target_parts", [])] == [
            self.setup["base_parts"],
            self.setup["target_parts"],
        ]

    def meshes(self) -> dict:
        """Decimated part meshes for the 3D view, framed on the whole product (#13)."""
        self._need_product()
        if self._meshes is None:
            from disassembly_pomdp.mesh_export import export_part_mesh

            parts = {}
            for i, name in enumerate(self.part_names):
                parts[name] = export_part_mesh(self.assembly.get_part_by_name(name))
                parts[name]["color"] = list(_PALETTE[i % len(_PALETTE)])
            lo, hi = _bounds(self.assembly)
            self._meshes = {
                "parts": parts,
                "scene_center": ((lo + hi) / 2).tolist(),
                "scene_radius": float(np.linalg.norm(hi - lo) / 2),
                "highlight_color": list(HIGHLIGHT_COLOR),
            }
        return self._meshes

    # ------------------------------------------------------------ edit history
    def _reset_history(self) -> None:
        self._undo: list[_Memento] = []
        self._redo: list[_Memento] = []
        self._serial = 0  # changes with every edit, undo and redo: a graph of another serial is stale
        self._version = self._saved_version = self._versions = (
            0  # the product model's identity, for "unsaved"
        )
        self._removed: list[str] | None = (
            None  # removal order of the current state while it is not the graph's
        )
        self._state: tuple[DisassemblyGraphAgent, object] | None = (
            None  # agent and assembly of that state
        )
        self._available: list[dict] | None = None
        self._recheck_error: str | None = None
        self._merged: frozenset = (
            frozenset()
        )  # merged in this session; undo and redo restore it
        self._report: dict | None = (
            None  # of the last mesh update, shown while `_report_serial` is current
        )
        self._report_serial = -1
        self._geometry = self._geometries = 0  # mesh version, see `_apply_edit`

    def _log(self, line: str) -> None:
        self.status_log = [*self.status_log, line][-200:]

    @property
    def unsaved(self) -> bool:
        return self._version != self._saved_version

    def rename_part(self, part: str, name: str) -> None:
        """The first product model edit: rename one part, also in the product setup."""
        self._need_product()
        name = name.strip()
        with self.lock:
            self._need_idle()
            if part not in self.part_names:
                raise ExplorerError(f"Unknown part {part}")
            if not name:
                raise ExplorerError("A part needs a name")
            if name == part:
                return
            if name in self.part_names:
                raise ExplorerError(f"Another part is already named {name}")

            def edit(assembly, setup: dict) -> dict:
                return _rename(assembly, setup, {part: name})

            self._apply_edit(edit)

    def rename_parts_by_rules(self, rules_file: str | Path) -> None:
        """Rename every part a rules file names (lines `old new`) in one edit, also in the product setup."""
        self._need_product()
        rules = _read_rules(rules_file)
        with self.lock:
            self._need_idle()
            renames = {p: rules[p] for p in self.part_names if rules.get(p, p) != p}
            if not renames:
                raise ExplorerError("No rule of the rules file renames a part")
            final = [renames.get(p, p) for p in self.part_names]
            clashes = sorted({n for n in final if final.count(n) > 1})
            if clashes:
                raise ExplorerError(
                    f"The rules would give several parts the name {', '.join(clashes)}"
                )

            self._apply_edit(lambda assembly, setup: _rename(assembly, setup, renames))

    def set_part_properties(self, part: str, material: str, density: float) -> None:
        """Set the material and density of one part; both are saved with the product model."""
        self._need_product()
        material = material.strip()
        with self.lock:
            self._need_idle()
            if part not in self.part_names:
                raise ExplorerError(f"Unknown part {part}")
            if not material:
                raise ExplorerError("A part needs a material")
            if not math.isfinite(density) or density <= 0:
                raise ExplorerError("The density must be a positive number (kg/m3)")
            current = self.assembly.parts[part]
            if (current.material, current.density) == (material, density):
                return

            def edit(assembly, _setup: dict) -> dict:
                p = assembly.parts[part]
                if p.weight is not None:
                    p.weight *= density / p.density
                p.material, p.density = material, density
                return {}

            self._apply_edit(edit)

    def delete_part(self, part: str) -> None:
        """Remove a part with its connections, also from the product setup."""
        self._need_product()
        with self.lock:
            self._need_idle()
            if part not in self.part_names:
                raise ExplorerError(f"Unknown part {part}")
            if len(self.part_names) == 1:
                raise ExplorerError("The last part of a product cannot be deleted")

            def edit(assembly, setup: dict) -> dict:
                assembly.remove_part(assembly.parts[part])
                for key, parts in setup.items():
                    setup[key] = [p for p in parts if p != part]
                return {}

            self._apply_edit(edit, parts_changed=True)

    def merge_parts(self, parts: list[str], name: str) -> None:
        """Merge two or more parts into one subassembly part. Its mesh is written to the product folder at
        once, because the product model reloads meshes by path on every copy; an undo leaves that file behind."""
        self._need_product()
        name, members = name.strip(), list(dict.fromkeys(parts))
        with self.lock:
            self._need_idle()
            if self.folder is None:
                raise Conflict("Save the product first: it has no folder yet")
            if len(members) < 2:
                raise ExplorerError("Merging needs at least two different parts")
            unknown = sorted(set(members) - set(self.part_names))
            if unknown:
                raise ExplorerError(f"Unknown parts: {', '.join(unknown)}")
            if not name or _NOT_A_FILE_NAME.search(name):
                raise ExplorerError(
                    "The subassembly needs a name that is also a file name"
                )
            if name in self.part_names:
                raise ExplorerError(f"Another part is already named {name}")
            if set(members) & set(self.setup["base_parts"]) and set(members) & set(
                self.setup["target_parts"]
            ):
                raise ExplorerError(
                    "Parts of the base and of the target cannot be merged into one"
                )
            mesh_path = self.xml.parent / self.xml.stem / f"{name}.stl"
            taken = {Path(p.mesh_path).resolve() for p in self.assembly.parts.values()}
            if mesh_path.exists() or mesh_path.resolve() in taken:
                raise ExplorerError(
                    f"{mesh_path.name} exists in the product folder: pick another name"
                )

            def edit(assembly, setup: dict) -> dict:
                _merge(assembly, members, name, mesh_path)
                for key, names in setup.items():
                    first = next((i for i, p in enumerate(names) if p in members), None)
                    if first is not None:
                        rest = [p for p in names if p not in members]
                        setup[key] = rest[:first] + [name] + rest[first:]
                return {}

            try:
                self._apply_edit(edit, parts_changed=True, merged=name)
            except OSError as e:
                raise ExplorerError(str(e)) from e

    # ------------------------------------------------------------ mesh update (#244)
    def check_update(self, folder: str | Path, depth: int = 0) -> dict:
        """What a mesh update from `folder` would do, without doing it: the product author ticks the new mesh
        files to add. Parts are matched to the meshes by file name; merged subassemblies (those made in this
        session; no merge record is kept in the product folder) are skipped."""
        with self.lock:
            return self._plan_update(folder, depth).as_dict()

    def update_meshes(
        self, folder: str | Path, depth: int = 0, add: list[str] | None = None
    ) -> threading.Thread | None:
        """Update the meshes of the open product from a new CAD export folder as one undoable edit: changed
        parts get the new mesh and fresh contacts, parts with no mesh file are removed, the `add` files join. It
        runs in a background thread on a copy, with progress in `self.job`; the edit is applied at the end, so a
        failure keeps the open product."""
        add = list(add or [])
        with self.lock:
            plan = self._plan_update(folder, depth)
            try:
                check_add(plan, add)
            except ProductFolderError as e:
                raise ExplorerError(str(e)) from e
            if not (plan.changed or plan.gone or add):
                self._report = {
                    "folder": str(plan.folder),
                    "changed": [],
                    "unchanged": len(plan.unchanged),
                    "removed": [],
                    "added": [],
                    "not_added": list(plan.new_files),
                    "merged": list(plan.merged),
                    "redetected": [],
                }
                self._report_serial = self._serial
                self.job = Job(
                    kind="update"
                )  # the browser shows this result, not an older job's
                self._log(report_line(self._report) + " (nothing to change)")
                return None
            self.job = job = Job(running=True, kind="update")
            work, setup = copy.deepcopy(self.assembly), copy.deepcopy(self.setup)
            mesh_dir = self.xml.parent / self.xml.stem
        thread = threading.Thread(
            target=self._update,
            args=(job, plan, add, work, setup, mesh_dir),
            daemon=True,
        )
        thread.start()
        return thread

    def _plan_update(self, folder, depth):
        self._need_product()
        self._need_idle()
        if self.folder is None:
            raise Conflict("Save the product first: it has no folder yet")
        try:
            return plan_update(self.assembly, self._merged, folder, depth)
        except ProductFolderError as e:
            raise ExplorerError(str(e)) from e

    def _update(self, job: Job, plan, add, work, setup, mesh_dir: Path) -> None:
        def progress(done: int, total: int) -> None:
            job.expanded, job.discovered = done, total

        try:
            report = apply_update(work, setup, plan, add, mesh_dir, progress)

            def edit(assembly, product_setup: dict) -> dict:
                assembly.parts, assembly.connections = work.parts, work.connections
                assembly._fcl_objects = {}
                product_setup.update(setup)
                return {}

            with self.lock:
                job.running = False
                self._apply_edit(edit, parts_changed=True, new_geometry=True)
                self._report, self._report_serial = report, self._serial
                self._log(report_line(report))
        except Exception as e:  # shown in the UI; the open product stays
            job.error = f"{type(e).__name__}: {e}"
            self._log(f"Mesh update failed: {job.error}")
        finally:
            job.running = False

    # ------------------------------------------------------------ connection authoring (#242)
    def screw_heads(self) -> list[str]:
        """The screw heads the screwdrivers of the cell in use support."""
        cell = Cell() if self.cell_choice is None else self.cell_choice.cell
        screwing = ScrewingCapability(cell.screwdrivers)
        return sorted(set().union(*screwing.tool_to_heads.values()))

    def connections(self) -> list[dict]:
        """The connections of the product model, numbered as `edit_connection` and `remove_connection` take them."""
        if self.assembly is None:
            return []
        return [
            {
                "index": i,
                "type": c.connection_type,
                "part1": c.part_name1,
                "part2": c.part_name2,
                "screw_head": getattr(c, "screw_head", None),
            }
            for i, c in enumerate(self.assembly.connections)
        ]

    def screw_warnings(self) -> list[str]:
        """Screw connections where "screw" appears in neither or both part names: the gym cannot tell which part
        is the screw."""
        if self.assembly is None:
            return []
        return [
            f'{c}: "screw" appears in {"both" if "screw" in c.part_name1.lower() else "neither"} part name'
            for c in self.assembly.connections
            if c.connection_type == "screw"
            and ("screw" in c.part_name1.lower()) == ("screw" in c.part_name2.lower())
        ]

    def add_connection(
        self, kind: str, part1: str, part2: str, screw_head: str | None = None
    ) -> None:
        """Add a screw, safety ring or fixed connection. A screw connection takes the screw first, whichever
        order the screw and the held part were picked in, and a screw head the cell's screwdrivers support."""
        self._need_product()
        conn = self._new_connection(kind, part1, part2, screw_head)
        with self.lock:
            self._need_idle()
            self._check_new(conn)
            self._apply_edit(lambda assembly, _setup: _add(assembly, conn))

    def edit_connection(
        self, index: int, part1: str, part2: str, screw_head: str | None = None
    ) -> None:
        """Change the parts (and the screw head) of a screw, safety ring or fixed connection."""
        self._need_product()
        with self.lock:
            self._need_idle()
            old = self._hand_made(index)
            conn = self._new_connection(old.connection_type, part1, part2, screw_head)
            self._check_new(conn, ignore=old)
            self._apply_edit(lambda assembly, _setup: _replace(assembly, index, conn))

    def remove_connection(self, index: int) -> None:
        """Remove one connection, of any type; it takes only itself (the parts it held are available again)."""
        self._need_product()
        with self.lock:
            self._need_idle()
            if not 0 <= index < len(self.assembly.connections):
                raise ExplorerError(f"Unknown connection {index}")
            self._apply_edit(lambda assembly, _setup: _remove(assembly, index))

    def redetect_contacts(self, part1: str, part2: str) -> None:
        """Replace the geometric connections of one part pair by freshly detected ones; nothing else changes."""
        self._need_product()
        with self.lock:
            self._need_idle()
            for part in (part1, part2):
                if part not in self.part_names:
                    raise ExplorerError(f"Unknown part {part}")
            if part1 == part2:
                raise ExplorerError("A contact needs two different parts")
            self._apply_edit(lambda assembly, _setup: _redetect(assembly, part1, part2))

    def _new_connection(self, kind: str, part1: str, part2: str, screw_head):
        for part in (part1, part2):
            if part not in self.part_names:
                raise ExplorerError(f"Unknown part {part}")
        if part1 == part2:
            raise ExplorerError("A connection needs two different parts")
        if kind == "screw":
            if screw_head not in self.screw_heads():
                raise ExplorerError(
                    f"Screw head {screw_head!r} is not supported by the cell's screwdrivers "
                    f"({', '.join(self.screw_heads())})"
                )
            # the screw first, whatever order the two were picked in: the gym tells the screw by its name
            screw, held = ScrewingCapability._get_screw_and_base_names(
                Screw_Connection(part1, part2, screw_head)
            )
            return Screw_Connection(screw, held, screw_head)
        if kind in _HAND_MADE:
            return _HAND_MADE[kind](part1, part2)
        raise ExplorerError(
            f"Unknown connection type {kind!r} (screw, safety_ring or fixed)"
        )

    def _hand_made(self, index: int):
        conns = self.assembly.connections
        if not 0 <= index < len(conns):
            raise ExplorerError(f"Unknown connection {index}")
        if conns[index].connection_type not in ("screw", *_HAND_MADE_NAMES):
            raise ExplorerError(
                "Only screw, safety ring and fixed connections are edited; "
                "remove a geometric connection or re-detect its pair"
            )
        return conns[index]

    def _check_new(self, conn, ignore=None) -> None:
        if any(
            c.connection_type == conn.connection_type
            and {c.part_name1, c.part_name2} == {conn.part_name1, conn.part_name2}
            for c in self.assembly.connections
            if c is not ignore
        ):
            raise ExplorerError(f"{conn} exists already")

    def undo(self) -> None:
        self._travel(self._undo, self._redo, inverse=True)

    def redo(self) -> None:
        self._travel(self._redo, self._undo, inverse=False)

    def _apply_edit(
        self,
        edit: Callable[[object, dict], dict],
        model: bool = True,
        cell: CellChoice | None = None,
        parts_changed: bool = False,
        merged: str | None = None,
        new_geometry: bool = False,
    ) -> None:
        """Run `edit(assembly, setup) -> renames` on copies of the product model and product setup, keep a
        memento of the state before it and drop the redo history. A `cell` replaces the cell in use with the
        same edit. Every edit of the product explorer comes here.
        `parts_changed`: the edit adds or removes parts, so the 3D meshes are fetched again.
        `merged`: the name of the subassembly the edit made (see `update_meshes`).
        `new_geometry`: the edit replaces mesh content, so a browser that has the meshes of the same part names
        must fetch them again (`mesh_version` in the view)."""
        self._need_product()
        with self.lock:
            self._need_idle()
            assembly, setup = copy.deepcopy(self.assembly), copy.deepcopy(self.setup)
            renames = edit(assembly, setup)
            self._undo.append(self._memento(renames))
            self._redo.clear()
            self.assembly, self.setup = assembly, setup
            self._merged = frozenset(
                {renames.get(n, n) for n in self._merged}
                | ({merged} if merged else set())
            ) & set(assembly.parts)
            if cell is not None:
                self.cell_choice = cell
            if model:
                self._versions += 1
                self._version = self._versions
            if new_geometry:
                self._geometries += 1
                self._geometry = self._geometries
            if parts_changed:
                self._meshes = None
            elif self._meshes is not None:
                self._meshes = _renamed_meshes(self._meshes, renames)
            self._changed(renames)

    def _travel(self, take: list, give: list, inverse: bool) -> None:
        with self.lock:
            self._need_product()
            self._need_idle()
            if not take:
                raise Conflict(f"Nothing to {'undo' if inverse else 'redo'}")
            memento = take.pop()
            give.append(self._memento(memento.renames))
            self.assembly, self.setup = memento.assembly, memento.setup
            self.cell_choice, self._merged = memento.cell, memento.merged
            self._meshes, self._version = memento.meshes, memento.version
            self._geometry = memento.geometry
            renames = memento.renames
            if inverse:
                renames = {new: old for old, new in renames.items()}
            self._changed(renames)

    def _memento(self, renames: dict) -> _Memento:
        return _Memento(
            self.assembly,
            self.setup,
            self._meshes,
            self._version,
            dict(renames),
            self.cell_choice,
            self._merged,
            self._geometry,
        )

    def _changed(self, renames: dict) -> None:
        """After an edit, undo or redo: the graph is stale, and the available actions are re-checked."""
        self._serial += 1
        order = [renames.get(p, p) for p in self._removal_order()]
        if not set(order) <= set(self.part_names):  # the current product state is gone
            order, self.path, self.recording = [], [], False
        self._removed = order
        self._recheck()

    def _removal_order(self) -> list[str]:
        """The parts removed in the current product state, in order."""
        if self._removed is not None:
            return list(self._removed)
        g = self._graph
        return (
            []
            if g is None
            else [
                name
                for e in self.path
                for name in _step_targets(g.graph.edges[g.edges[e]])
            ]
        )

    def _recheck(self) -> None:
        """Available actions of the current product state on the edited product model (no graph extraction)."""
        self._state = self._available = self._recheck_error = None
        try:
            agent, state = self._state_agent()
            self._available = agent.available_actions(state)
        except Exception as e:  # shown in the UI; the edit stays
            self._recheck_error = f"{type(e).__name__}: {e}"

    def _state_agent(self) -> tuple[DisassemblyGraphAgent, object]:
        """Agent on the in-memory product model and the assembly of the current product state."""
        if self._state is None:
            agent = _agent_for(self.assembly, self.setup, self.cell_choice)
            state = agent.replay(self._removal_order())
            if (
                state is None
            ):  # an edit made a removal impossible: restart from the full product
                self._removed, self.path, self.recording = [], [], False
                state = agent.base_assembly
            self._state = agent, state
        return self._state

    def _need_idle(self) -> None:
        if self.job.running:
            raise Conflict(
                {
                    "import": "Wait until the import is done",
                    "update": "Wait until the mesh update is done",
                }.get(self.job.kind, "Wait until Generate graph is done")
            )

    # ------------------------------------------------------------ sequence graph
    def generate(
        self,
        extraction_depth: str = DISASSEMBLED_STATES,
        parallel: bool = False,
        n_robots: int = 2,
        collapse_parallel_screws: bool = True,
        min_separation: float = 50.0,
    ) -> threading.Thread:
        """Start Rebuild (Generate graph) in a background thread; progress goes to `self.job`.
        `extraction_depth`: down to the disassembled states (default) or a full teardown."""
        self._need_product()
        try:
            check_extraction_depth(extraction_depth)
            DisassemblyGraphAgent._check_parallel_options(n_robots, False)
            ParallelGeometrySettings(min_separation=min_separation)
        except ValueError as e:
            raise ExplorerError(str(e)) from e
        with self.lock:
            if self.job.running:
                raise Conflict("Generate graph is already running")
            self.job = Job(running=True)
            assembly, setup = copy.deepcopy(self.assembly), copy.deepcopy(self.setup)
            serial, cell = self._serial, self.cell_choice
        thread = threading.Thread(
            target=self._generate,
            args=(
                self._product_id,
                assembly,
                setup,
                serial,
                cell,
                extraction_depth,
                parallel,
                n_robots,
                collapse_parallel_screws,
                min_separation,
            ),
            daemon=True,
        )
        thread.start()
        return thread

    def _generate(
        self,
        product_id: int,
        assembly,
        setup: dict,
        serial: int,
        cell: CellChoice | None,
        extraction_depth: str,
        parallel: bool = False,
        n_robots: int = 2,
        collapse_parallel_screws: bool = True,
        min_separation: float = 50.0,
    ) -> None:
        job = self.job
        try:
            agent = _agent_for(assembly, setup, cell)
            agent.parallel_geometry = ParallelGeometrySettings(
                min_separation=min_separation
            )

            def progress(expanded: int, discovered: int) -> None:
                job.expanded, job.discovered = expanded, discovered

            graph = agent.build_graph(
                verbose=False,
                collapse_screw_groups=not parallel,
                parallel=parallel,
                n_robots=n_robots,
                collapse_parallel_screws=collapse_parallel_screws,
                progress=progress,
                extraction_depth=extraction_depth,
            )
            if parallel:
                graph.graph["min_separation"] = min_separation
            with self.lock:
                if product_id != self._product_id:
                    return  # another product was loaded meanwhile
                g = _Graph(graph, agent, serial, _fingerprint_of(cell))
                for i, state in enumerate(graph.nodes):
                    g.node_ids[state], g.nodes[f"s{i}"] = f"s{i}", state
                for i, edge in enumerate(graph.edges(keys=True)):
                    g.edges[f"e{i}"] = edge
                g.analysis = analysis_view(graph, agent, g.node_ids, g.edges)
                g.root = g.node_ids[
                    next(n for n, d in graph.nodes(data=True) if d.get("depth") == 0)
                ]
                node = g.node_ids.get(
                    tuple(sorted(set(self.part_names) - set(self._removal_order())))
                )  # exploring continues where it was, if that product state is in the new graph
                self._graph, self._removed = g, None
                self._warn_about_unknown_group_screws()
                self._state = self._available = None
                self.path = [] if node is None else _path_to(g, node)
                self.recording, self.recorded = False, []
        except Exception as e:  # shown in the UI; the old graph stays
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.running = False

    def stale(self) -> bool:
        g = self._graph
        return g is not None and (
            g.serial != self._serial
            or g.cell_fingerprint != _fingerprint_of(self.cell_choice)
        )

    def _need_graph(self) -> _Graph:
        if self._graph is None:
            raise Conflict("Generate the sequence graph first")
        if self.stale():
            raise Conflict("The sequence graph is stale: Rebuild first")
        return self._graph

    # ------------------------------------------------------------ exploring
    @property
    def current(self) -> str | None:
        g = self._graph
        if g is None:
            return None
        return g.node_ids[g.edges[self.path[-1]][1]] if self.path else g.root

    def next_steps(self) -> list[str]:
        g = self._graph
        if g is None or self.stale():
            return []
        state = g.nodes[self.current]
        steps = [e for e, (u, _v, _k) in g.edges.items() if u == state]
        return sorted(steps, key=lambda e: g.graph.edges[g.edges[e]]["action_time"])

    def step(self, edge: str) -> None:
        with self.lock:
            if edge not in self.next_steps():
                raise ExplorerError(
                    f"{edge} is not a next step of the current product state"
                )
            self.path.append(edge)
            if self.recording:
                self.recorded.append(edge)

    def back(self) -> None:
        with self.lock:
            if self.stale():
                raise Conflict("The sequence graph is stale: Rebuild first")
            if self.path:
                edge = self.path.pop()
                if self.recording and self.recorded and self.recorded[-1] == edge:
                    self.recorded.pop()

    def restart(self) -> None:
        with self.lock:
            self.path, self.recording = [], False
            if (
                self.stale()
            ):  # exploring restarts from the full product on the edited model
                self._removed = []
                self._recheck()

    def jump(self, node: str) -> None:
        """Go to any node of the sequence graph; the path becomes a shortest one from the start state."""
        g = self._need_saved_graph()
        if node not in g.nodes:
            raise ExplorerError(f"Unknown product state {node}")
        with self.lock:
            self.path, self.recording = _path_to(g, node), False

    def parts_present(self) -> list[str]:
        g = self._graph
        if g is None:
            return self.part_names
        if (
            self.stale()
        ):  # the graph's product states carry the names of the model it was built from
            removed = set(self._removal_order())
            return [p for p in self.part_names if p not in removed]
        return list(g.nodes[self.current])

    def rejection(self, part: str) -> dict:
        """Whether `part` can be removed in the current product state and, if not, why (rejection reason)."""
        self._need_product()
        if part not in self.part_names:
            raise ExplorerError(f"Unknown part {part}")
        if part not in self.parts_present():
            return {
                "part": part,
                "available": False,
                "text": f"{part} is already removed",
            }
        g = self._graph
        if g is not None and not self.stale():
            agent, assembly = g.agent, g.agent.assemblies[g.nodes[self.current]]
        else:
            agent, assembly = self._state_agent()
        reasons = agent.rejection_reasons(assembly, part)
        accepted = [r for r in reasons if r["reason"] is None]
        if accepted:
            text = f"{part} can be removed ({', '.join(sorted({r['capability'] for r in accepted}))})"
            group = next((grp for grp in self._collapsed_groups() if part in grp), None)
            if group and not any(
                part in _step_targets(g.graph.edges[g.edges[e]])
                for e in self.next_steps()
            ):
                text += (
                    f"; it is one of {len(group)} interchangeable screws, which the sequence graph "
                    "removes in one order (by name)"
                )
            return {"part": part, "available": True, "text": text}
        texts = sorted({_reason_text(r) for r in reasons}) or [
            "no capability of the cell acts on it"
        ]
        touching = sorted(
            {
                c.part_name1 if c.part_name2 == part else c.part_name2
                for c in assembly.get_connections_by_part_name(part)
            }
        )
        holding = _holding(assembly, part)
        text = f"{part} is rejected: {'; '.join(texts)}"
        if holding:
            text += f". Held by: {'; '.join(holding)}"
        if touching:
            more = f" and {len(touching) - 8} more" if len(touching) > 8 else ""
            text += f". Touching: {', '.join(touching[:8])}{more}"
        return {
            "part": part,
            "available": False,
            "text": text,
            "reasons": reasons,
            "holding": holding,
        }

    def _warn_about_unknown_group_screws(self) -> None:
        self.warnings = [w for w in self.warnings if not w.startswith(_UNKNOWN_SCREWS)]
        unknown = groups_naming_unknown_parts(self._collapsed_groups(), self.part_names)
        if unknown:
            self.warnings.append(f"{_UNKNOWN_SCREWS}{', '.join(unknown)}")

    def _collapsed_groups(self) -> list[list[str]]:
        return (
            self._graph.graph.graph.get("collapsed_screw_groups", [])
            if self._graph
            else []
        )

    # ------------------------------------------------------------ recording and export
    def set_recording(self, on: bool) -> None:
        with self.lock:
            if on and not self.recording:
                if self._need_graph().graph.graph.get("parallel"):
                    raise ExplorerError(
                        "Parallel analysis supports graph export, not robot-planner recording"
                    )
                self.recorded = []
            self.recording = on

    def save_recording(self, path: str | Path) -> Path:
        g = self._need_exportable()
        if not self.recorded:
            raise ExplorerError("Nothing recorded")
        sequence = nx.MultiDiGraph(
            **g.graph.graph
        )  # built step by step: the file keeps the recorded order
        for e in self.recorded:
            u, v, k = g.edges[e]
            for state in (u, v):
                sequence.add_node(state, **g.graph.nodes[state])
            sequence.add_edge(u, v, **g.graph.edges[u, v, k])
        path = Path(path)
        export_graph(
            complete_by_reversal(sequence), path, self._provenance(path.parent)
        )
        return path

    def export(self, out_dir: str | Path) -> Path:
        """The sequence graph file and the per-part URDFs (+ manifest.json) into `out_dir` (#77)."""
        if self._need_graph().graph.graph.get("parallel"):
            raise ExplorerError(
                "Use analysis graph export for parallel graphs; planner export supports sequential graphs only"
            )
        g = self._need_exportable()
        out = Path(out_dir)
        provenance = self._provenance(out)
        export_graph(
            complete_by_reversal(g.graph), out / "sequence_graph.json", provenance
        )  # ADR-0005
        self.assembly.generate_urdf_files(str(out / "urdf"), mesh_dir=str(out / "mesh"))
        for urdf in (out / "urdf").glob("*.urdf"):
            urdf.write_text(
                _portable_mesh_paths(urdf.read_text(encoding="utf-8")), encoding="utf-8"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "provenance": provenance,
            "parts": self.part_names,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return out

    def export_analysis(self, out_dir: str | Path) -> Path:
        """Full graph, shortest sequence and rejection plots of the current analysis."""
        self._need_idle()
        g = self._need_graph()
        out = Path(out_dir).resolve()
        if (
            self.folder is not None and out.is_relative_to(self.folder.resolve())
        ) or files_repository_root(out) is not None:
            raise ExplorerError(
                "Choose an analysis output folder outside the product source folder"
            )
        with self.lock:
            return export_analysis(g.graph, g.agent, out)

    def _need_saved_graph(self) -> _Graph:
        g = self._need_graph()
        if self.unsaved:
            raise Conflict("Save the product first: it has unsaved edits")
        return g

    def _need_exportable(self) -> _Graph:
        unusable = parts_without_position(self.assembly) if self.assembly else []
        if unusable:
            raise ExplorerError(
                f"The export feeds the planner, which needs a usable mesh for every part; fix the mesh of {', '.join(unusable)}"
            )
        g = self._need_saved_graph()
        self._warn_about_unknown_group_screws()
        if not self.setup_stored():
            raise Conflict(
                "Store the product setup first: the export's provenance fingerprints the stored file"
            )
        return g

    def _provenance(self, export_folder: Path) -> dict:
        """The provenance for an export into `export_folder`, which must not lie in the 3D files repository
        (exports are derived files, ADR-0011)."""
        if files_repository_root(export_folder) is not None:
            raise ExplorerError(
                f"{export_folder} lies in the 3D files repository; choose another output folder"
            )
        cell = self.cell_choice
        root = self.product_root()
        return make_provenance(
            self.xml.stem,
            self.folder,
            ProductModel(self.folder, self.xml).relative_xml,
            PRODUCT_SETUP_FILE,
            None if cell is None else cell.path,
            export_folder=export_folder,
            cell_fingerprint=_fingerprint_of(cell),
            product_root=None if root is None else root.path,
            extraction_depth=self._graph.extraction_depth,
        )

    def product_root(self) -> ProductRoot | None:
        """The product root in effect (#257): app setting, else environment variable, else code default."""
        try:
            return resolve_product_root(self.folder, self.config_dir)
        except InvalidProductRoot as e:
            raise ExplorerError(str(e)) from e

    def set_product_root(self, path: str | Path | None) -> None:
        """Set the app setting to `path`, or clear it with None."""
        try:
            write_setting(path, self.config_dir)
        except InvalidProductRoot as e:
            raise ExplorerError(str(e)) from e

    def _product_root_view(self) -> dict:
        try:
            root = self.product_root()
            error = None
        except ExplorerError as e:
            root, error = None, str(e)
        return {
            "path": None if root is None else str(root.path),
            "source": None if root is None else root.source,
            "setting": read_setting(self.config_dir),
            "env_var": ENV_VAR,
            "error": error,
        }

    # ------------------------------------------------------------ view
    def view(self) -> dict:
        """Everything the browser renders, except the meshes."""
        with self.lock:
            g = self._graph
            graph = None
            if g is not None:
                graph = {
                    **g.analysis,
                    "root": g.root,
                    "stale": self.stale(),
                    "cell_fingerprint": g.cell_fingerprint,
                    "extraction_depth": g.extraction_depth,
                    "collapsed_screw_groups": self._collapsed_groups(),
                    "nodes": {
                        i: _node_view(g.graph.nodes[s]) for i, s in g.nodes.items()
                    },
                    "edges": {e: _edge_view(g, e) for e in g.edges},
                }
            return {
                "product": None
                if self.assembly is None
                else {
                    "name": self.name,
                    "mesh_version": self._geometry,
                    "folder": None if self.folder is None else str(self.folder),
                    "parts": self.part_names,
                    "properties": {
                        p.name: {"material": p.material, "density": p.density}
                        for p in self.assembly.parts.values()
                    },
                },
                "setup": self.setup,
                "connections": self.connections(),
                "screw_heads": self.screw_heads(),
                "screw_warnings": self.screw_warnings(),
                "cell": self._cell_view(),
                "product_root": self._product_root_view(),
                "warnings": list(self.warnings),
                "notices": self._setup_notices(),
                "setup_stored": self.setup_stored(),
                "graph": graph,
                "unsaved": self.unsaved,
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
                "recheck": None
                if self.assembly is None
                or self._available is None
                and not self._recheck_error
                else {"available": self._available, "error": self._recheck_error},
                "job": self.job.as_dict(),
                "update_report": self._report
                if self._report_serial == self._serial
                else None,
                "status_log": list(self.status_log),
                "current": self.current,
                "path": list(self.path),
                "parts_present": self.parts_present(),
                "next_steps": self.next_steps(),
                "recording": self.recording,
                "recorded": list(self.recorded),
            }

    def _cell_view(self) -> dict:
        """The cell in use: the cell file's name, or the default cell, and the tools that drive the actions."""
        choice = self.cell_choice
        cell = Cell() if choice is None else choice.cell
        return {
            "name": None if choice is None else choice.name,
            "path": None if choice is None else str(choice.path),
            "default": choice is None,
            "screwdrivers": list(cell.screwdrivers),
            "grippers": list(cell.grippers),
        }

    def _setup_notices(self) -> list[str]:
        if self.assembly is None:
            return []
        notices = []
        if not self.setup["base_parts"]:
            notices.append("no base parts: every part may be removed")
        if not self.setup["target_parts"]:
            notices.append("no target parts: no goal states")
        return notices

    def _need_product(self) -> None:
        if self.assembly is None:
            raise Conflict("Load a product folder or import a mesh folder first")


def _agent_for(
    assembly, setup: dict, cell: CellChoice | None = None
) -> DisassemblyGraphAgent:
    """The agent of the product model in memory (loaded through the adapter, #238, or imported, #243, and maybe
    edited since) on the cell in use (the default cell if no cell file was chosen); `build_env` reads no folder
    when it gets the assembly."""
    return DisassemblyGraphAgent(
        build_env(None, None if cell is None else cell.cell, assembly=assembly, **setup)
    )


def _fingerprint_of(cell: CellChoice | None) -> str | None:
    return None if cell is None else cell.fingerprint


def _read_cell_file(path: Path) -> CellChoice:
    """Read a cell file in the existing cell file format, or raise a user error."""
    try:
        cell_file = load_cell_file(path)
        cell = cell_from_resources(cell_file.cell, cell_file.travel_times)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ExplorerError(f"{path} is not a readable cell file: {e!r}") from e
    if not (cell.screwdrivers or cell.grippers):
        raise ExplorerError(f"{path}: the cell has neither screwdrivers nor grippers")
    digest = fingerprint_of_cell(path)
    if (
        digest is None
    ):  # the file vanished meanwhile; None would read as the default cell
        raise ExplorerError(f"{path} is not a readable cell file")
    return CellChoice(path.resolve(), digest, cell)


def _rename(assembly, setup: dict, renames: dict) -> dict:
    """Rename parts (old -> new, all at once, so swaps work) in the product model and product setup, and
    keep the order of the parts. The core's renames leave the keys of its part dict and FCL cache behind."""
    for part in assembly.parts.values():
        part.name = renames.get(part.name, part.name)
    assembly.parts = {renames.get(n, n): p for n, p in assembly.parts.items()}
    assembly._fcl_objects = {
        renames.get(n, n): o for n, o in (assembly._fcl_objects or {}).items()
    }
    for c in assembly.connections:
        c.part_name1, c.part_name2 = (
            renames.get(c.part_name1, c.part_name1),
            renames.get(c.part_name2, c.part_name2),
        )
    for key, parts in setup.items():
        setup[key] = [renames.get(p, p) for p in parts]
    return dict(renames)


def _read_rules(rules_file: str | Path) -> dict[str, str]:
    """Lines `old new` of a rules file, whole names only; the first rule for a name wins."""
    try:
        lines = Path(rules_file).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise ExplorerError(f"Cannot read the rules file {rules_file}: {e}") from e
    rules: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        words = line.split()
        if len(words) not in (0, 2):
            raise ExplorerError(
                f"Rules file line {number}: expected 'old new', found '{line.strip()}'"
            )
        if words:
            rules.setdefault(words[0], words[1])
    return rules


def _merge(assembly, members: list[str], name: str, mesh_path: Path) -> None:
    """Merge `members` into the part `name`, whose mesh is written to `mesh_path`.

    The core's `merge_parts` writes next to the first member's mesh, overwrites a file there and leaves one
    geometric connection per member and neighbour, whose contact data files would then share a name. Here the
    connections of a neighbour become one: the contact data are joined, and with the pair in the other order the
    normals are turned, because they are seen from the second part."""
    first = assembly.parts[members[0]]
    mesh = o3d_merge_triangle_meshes([assembly.parts[m].mesh for m in members])
    mesh.compute_triangle_normals()  # an STL without normals is written empty
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    o3d_save_triangle_mesh(mesh, str(mesh_path))
    same = all(
        (assembly.parts[m].material, assembly.parts[m].density)
        == (first.material, first.density)
        for m in members
    )
    merged = Part(
        name,
        str(mesh_path),
        density=first.density if same else _DEFAULT_DENSITY,
        mesh=mesh,
        material=first.material if same else _DEFAULT_MATERIAL,
    )

    inside, connections, joined, seen = set(members), [], {}, set()
    for c in assembly.connections:
        a, b = c.part_name1 in inside, c.part_name2 in inside
        if a and b:
            continue  # between the members: the connection is gone
        if a:
            c.part_name1 = name
        elif b:
            c.part_name2 = name
        if not (a or b):
            connections.append(c)
        elif isinstance(c, Geometric_Connection):
            other = c.part_name2 if a else c.part_name1
            if other in joined:
                joined[other].append(c)
            else:
                joined[other] = [c]
                connections.append(other)  # placeholder, replaced below
        else:
            key = (
                c.connection_type,
                getattr(c, "screw_head", None),
                c.part_name1,
                c.part_name2,
            )
            if key not in seen:
                seen.add(key)
                connections.append(c)
    assembly.connections = [
        _join(name, joined[c]) if isinstance(c, str) else c for c in connections
    ]
    for m in members:
        assembly.remove_part(assembly.parts[m])
    assembly.add_part(merged)


def _join(name: str, group: list) -> Geometric_Connection:
    """One geometric connection of the merged part and a neighbour, in the order of the first one."""
    if len(group) == 1:
        return group[0]
    head = group[0]
    normals = [np.asarray(head.normals)]
    for c in group[1:]:
        n = np.asarray(c.normals)
        normals.append(n if c.part_name1 == head.part_name1 else -n)
    return Geometric_Connection(
        head.part_name1,
        head.part_name2,
        np.concatenate([np.asarray(c.points) for c in group]),
        np.concatenate(normals),
        np.concatenate([np.asarray(c.distances) for c in group]),
    )


def _holding(assembly, part: str) -> list[str]:
    """The screw, safety ring and fixed connections that still hold `part` to a part that is present."""
    return [
        str(c)
        for c in assembly.get_connections_by_part_name(part)
        if c.connection_type in ("screw", *_HAND_MADE_NAMES)
        and (
            c.connection_type != "screw"
            or ScrewingCapability._get_screw_and_base_names(c)[1] == part
        )
    ]


def _add(assembly, conn) -> dict:
    assembly.add_connection(conn)
    return {}


def _replace(assembly, index: int, conn) -> dict:
    assembly.connections[index] = conn
    return {}


def _remove(assembly, index: int) -> dict:
    del assembly.connections[index]
    return {}


def _redetect(assembly, part1: str, part2: str) -> dict:
    redetect_pair(assembly, part1, part2)
    return {}


def _renamed_meshes(meshes: dict, renames: dict) -> dict:
    return meshes | {
        "parts": {renames.get(n, n): m for n, m in meshes["parts"].items()}
    }


def _path_to(g: _Graph, node: str) -> list[str]:
    """Edges of a shortest path from the start state to `node`."""
    edge_of = {(u, v): e for e, (u, v, _k) in g.edges.items()}
    prev, queue = {g.nodes[g.root]: None}, deque([g.nodes[g.root]])
    while queue:
        u = queue.popleft()
        for v in g.graph.successors(u):
            if v not in prev:
                prev[v] = u
                queue.append(v)
    path, v = [], g.nodes[node]
    while prev[v] is not None:
        path.append(edge_of[(prev[v], v)])
        v = prev[v]
    return path[::-1]


def _bounds(assembly) -> tuple[np.ndarray, np.ndarray]:
    boxes = [p.mesh.get_axis_aligned_bounding_box() for p in assembly.parts.values()]
    return (
        np.min([b.get_min_bound() for b in boxes], axis=0),
        np.max([b.get_max_bound() for b in boxes], axis=0),
    )


def _portable_mesh_paths(urdf: str) -> str:
    """Forward slashes in mesh file names: the core writes `os.path.relpath`, i.e. `..\\mesh\\...` on Windows."""
    return re.sub(
        r'filename="([^"]*)"',
        lambda m: 'filename="{}"'.format(m[1].replace("\\", "/")),
        urdf,
    )


def _reason_text(reason: dict) -> str:
    text = _REASON_TEXT.get(reason["reason"], reason["reason"])
    if reason["tool"]:
        text = f"{reason['tool']}: {text}"
    why = reason["details"].get("why") or reason["details"].get("reason")
    if why:
        text += f" ({_WHY_TEXT.get(why, why)})"
    return text


def _node_view(attrs: dict) -> dict:
    return {k: attrs.get(k) for k in ("parts_present", "parts_removed", "depth")} | {
        "is_goal": bool(attrs.get("is_goal"))
    }


def _edge_view(g: _Graph, e: str) -> dict:
    u, v, k = g.edges[e]
    step = g.graph.edges[u, v, k]
    return {
        "source": g.node_ids[u],
        "target": g.node_ids[v],
        **{
            key: step.get(key)
            for key in (
                "label",
                "capability",
                "tool",
                "target_part",
                "action_time",
                "direction",
                "target_parts",
                "operations",
                "stages",
                "parallel",
                "screw_macro",
                "n_stages",
            )
        },
    }


def _step_targets(step):
    return step.get("target_parts") or [step["target_part"]]
