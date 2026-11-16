"""The sequence graph export: node-link JSON with a `schema_version` and a provenance header.

The Sequence graph context owns the format; consumers (the MRTAMP run controller) read it with
`read_export` and refuse a `schema_version` they do not know. The provenance header records where the
export came from, so a consumer can tell that the product folder changed since: the product, its
product folder, the repository commit of that folder, fingerprints of the product model XML, product
setup and cell file, the extraction depth (down to the disassembled states, or a full teardown) and the
code version. Steps keep `capability`, `cap_idx` and `raw_action` for
env replay only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import networkx as nx

from disassembly_pomdp.graph_extraction import _jsonify, check_extraction_depth
from disassembly_pomdp.travel_time_table import table_path

SCHEMA_VERSION = 1


class UnsupportedSchemaVersion(ValueError):
    """The export has a `schema_version` this code does not know (or none)."""


def fingerprint(path: str | Path) -> str | None:
    """SHA-256 of a text file with line endings normalised, or None if there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint_of_cell(cell_file: str | Path) -> str | None:
    """The fingerprint of a cell: of its cell file and its travel-time table together (ADR-0012), or None if
    there is no such cell file."""
    cell = fingerprint(cell_file)
    if cell is None:
        return None
    table = fingerprint(table_path(cell_file)) or ""
    return hashlib.sha256(f"{cell}\n{table}".encode()).hexdigest()


FILES_REPOSITORY = (
    "files_repository"  # a recorded path is from the root of the 3D files repository
)
EXPORT_FOLDER = "export_folder"  # a recorded path is relative to the folder the export is written to
UNRESOLVED = (
    "unresolved"  # only the name: the file is on another drive than the export folder
)
PRODUCT_ROOT = "product_root"  # a recorded path is relative to the product root (#257)
ABSOLUTE = (
    "absolute"  # a recorded path is the absolute path: it lies under no functional root
)


def make_provenance(
    product_name: str,
    product_folder: str | Path,
    product_model_xml: str,
    product_setup: str | None = None,
    cell_file: str | Path | None = None,
    *,
    export_folder: str | Path,
    cell_fingerprint: str | None = None,
    product_root: str | Path | None = None,
    extraction_depth: str | None = None,
) -> dict:
    """The provenance header. `product_model_xml` and `product_setup` are paths inside the product folder.
    `extraction_depth` is the depth the graph was extracted with (`graph.graph["extraction_depth"]`); None
    for a graph that does not say.

    Without a `product_root` no absolute path of this machine goes in. The product folder and the cell file are each recorded from the
    root of the 3D files repository when they lie in one, otherwise relative to `export_folder` (on another
    Windows drive: the bare name, base "unresolved"); `path_bases` says which. `cell_fingerprint` is the
    fingerprint of the cell (its cell file and travel-time table) as it was read (default: of the files now).

    With a `product_root` in effect (#257) the paths are recorded by `record_path` instead, whatever the
    export folder is; the root itself is not written into the header.
    """
    if extraction_depth is not None:
        check_extraction_depth(extraction_depth)
    folder = Path(product_folder)

    def portable(path: Path) -> tuple[str, str]:
        if product_root is None:
            return _portable(path, export_folder)
        return record_path(path, product_root)

    product_path, product_base = portable(folder)
    cell_path, cell_base = (
        (None, None) if cell_file is None else portable(Path(cell_file))
    )
    return {
        "product_name": product_name,
        "product_folder": product_path,
        "product_model_xml": product_model_xml,
        "product_setup": product_setup,
        "cell_file": cell_path,
        "path_bases": {"product_folder": product_base, "cell_file": cell_base},
        "fingerprints": {
            "product_model_xml": fingerprint(folder / product_model_xml),
            "product_setup": None
            if product_setup is None
            else fingerprint(folder / product_setup),
            "cell_file": None
            if cell_file is None
            else cell_fingerprint or fingerprint_of_cell(cell_file),
        },
        "extraction_depth": extraction_depth,
        "repo_commit": _git_commit(folder),
        "code_version": _git_commit(Path(__file__).resolve().parent),
    }


def export_graph(graph: nx.MultiDiGraph, path: str | Path, provenance: dict) -> None:
    """Write `graph` (the complete one, after reversal) as node-link JSON with the header."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        **_jsonify(nx.node_link_data(graph, edges="edges")),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_export(path: str | Path) -> tuple[nx.MultiDiGraph, dict]:
    """The graph and the provenance header of an export; an unknown `schema_version` is refused."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"{path} has schema_version {version!r}; this code reads schema_version {SCHEMA_VERSION}"
        )
    for node in data["nodes"]:
        node["id"] = tuple(node["id"])  # product states are tuples of part names
    for edge in data["edges"]:
        edge["source"], edge["target"] = tuple(edge["source"]), tuple(edge["target"])
    graph = nx.node_link_graph(
        {k: v for k, v in data.items() if k not in ("schema_version", "provenance")},
        directed=True,
        multigraph=True,
        edges="edges",
    )
    return graph, data["provenance"]


def _portable(path: Path, export_folder: str | Path) -> tuple[str, str]:
    """`path` as recorded in a provenance header: (text with forward slashes, base it is relative to)."""
    path = path.resolve()
    root = files_repository_root(path)
    if root is not None:
        return path.relative_to(root).as_posix(), FILES_REPOSITORY
    try:
        return Path(
            os.path.relpath(path, Path(export_folder).resolve())
        ).as_posix(), EXPORT_FOLDER
    except (
        ValueError
    ):  # another drive on Windows: no relative path exists, so keep the bare name
        return path.name, UNRESOLVED


def functional_roots(product_root: str | Path) -> list[Path]:
    """The product root, then its ancestor folders up to the root of the 3D files repository when the product
    root lies in that repository (nearest first); otherwise the product root alone."""
    root = Path(product_root).resolve()
    repo = files_repository_root(root)
    if repo is None:
        return [root]
    roots = [root]
    while roots[-1] != repo and roots[-1] != roots[-1].parent:
        roots.append(roots[-1].parent)
    return roots


def record_path(path: Path, product_root: str | Path) -> tuple[str, str]:
    """`path` as recorded with a product root in effect: (text with forward slashes, base).

    Under the product root: relative to it, base "product_root". Outside it but under a higher functional
    root: relative to the nearest such root, base "product_root+N" (N levels above the product root). Under
    none: the absolute path, base "absolute"."""
    path = path.resolve()
    for level, root in enumerate(functional_roots(product_root)):
        if path.is_relative_to(root):
            base = PRODUCT_ROOT if level == 0 else f"{PRODUCT_ROOT}+{level}"
            return path.relative_to(root).as_posix(), base
    return path.as_posix(), ABSOLUTE


def resolve_recorded_path(
    recorded: str, base: str | None, product_root: str | Path
) -> Path:
    """The full path of a recorded one, for a reader that knows the product root of its machine. Only the
    product-root bases are resolved; the older bases are relative to something this function does not know."""
    if base == ABSOLUTE:
        return Path(recorded)
    if base == PRODUCT_ROOT:
        return Path(product_root).resolve() / recorded
    if base is not None and base.startswith(PRODUCT_ROOT + "+"):
        root = Path(product_root).resolve()
        for _ in range(int(base[len(PRODUCT_ROOT) + 1 :])):
            root = root.parent
        return root / recorded
    raise ValueError(f"base {base!r} is not relative to the product root")


def files_repository_root(path: str | Path) -> Path | None:
    """The root of the 3D files repository (`crc_1574_3D_Files`) that `path` lies in, else None. The path may
    not exist yet: the nearest existing ancestor decides. Other git repositories (the techstack's test
    products, other projects) are not the 3D files repository."""
    folder = Path(path).resolve()
    while not folder.is_dir():
        if folder.parent == folder:
            return None
        folder = folder.parent
    root = _git_root(folder)
    return (
        root if root is not None and "crc_1574_3d_files" in root.name.lower() else None
    )


def _git_root(folder: Path) -> Path | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        Path(done.stdout.strip()).resolve()
        if done.returncode == 0 and done.stdout.strip()
        else None
    )


def _git_commit(folder: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None
