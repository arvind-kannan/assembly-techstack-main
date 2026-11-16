"""Mesh update of the Product explorer (#244): the meshes of the open product are matched by file name to a new
CAD export folder, changed parts get the new mesh and fresh contacts, missing parts go, new files may join.

Everything about meshes and contacts lives here, the session only runs it (see `mesh_import`). `plan_update`
reads and changes nothing; `apply_update` changes the product model copy it is given and copies the new meshes
into the product folder under a fresh `update-<n>` folder, because undo reloads the old meshes from their old
paths. The folder with the new export is never written.
"""

from __future__ import annotations

import hashlib
import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from diss_assembly_dataclasses.part import Part, load_parts_from_folder

from product_explorer.mesh_import import MESH_ENDINGS
from product_explorer.product_folder import ProductFolderError


@dataclass(frozen=True)
class UpdatePlan:
    folder: Path
    changed: dict[str, Path]  # part name -> its new mesh file
    unchanged: list[str]
    gone: list[str]  # parts whose mesh file is not in the new folder
    merged: list[str]  # merged subassemblies: skipped
    new_files: dict[str, Path]  # file name -> file, matching no part
    clashing: list[str]  # new files whose part name is taken already

    def as_dict(self) -> dict:
        return {
            "changed": list(self.changed),
            "unchanged": list(self.unchanged),
            "removed": list(self.gone),
            "merged": list(self.merged),
            "new_files": list(self.new_files),
            "clashing": list(self.clashing),
        }


def plan_update(assembly, merged, folder: str | Path, depth: int) -> UpdatePlan:
    """Match the parts of `assembly` to the meshes of `folder` by file name. Raises `ProductFolderError`."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ProductFolderError(f"{folder}: not a folder")
    if depth < 0:
        raise ProductFolderError("The subfolder depth cannot be negative")
    files: dict[str, Path] = {}
    twice = set()
    for found in load_parts_from_folder(str(folder), depth, MESH_ENDINGS):
        path = Path(found.mesh_path)
        if path.name in files:
            twice.add(path.name)
        files[path.name] = path
    if not files:
        raise ProductFolderError(
            f"{folder}: no meshes (STL or OBJ) found at subfolder depth {depth}"
        )
    if twice:
        raise ProductFolderError(
            f"Mesh file names must be unique, but several meshes are named {', '.join(sorted(twice))}"
        )
    changed, unchanged, gone, skipped = {}, [], [], []
    for name, part in assembly.parts.items():
        if name in merged:
            skipped.append(name)
            continue
        new = files.get(Path(part.mesh_path).name)
        if new is None:
            gone.append(name)
        elif _same_content(new, Path(part.mesh_path)):
            unchanged.append(name)
        else:
            changed[name] = new
    used = {Path(p.mesh_path).name for p in assembly.parts.values()}
    new_files = {f: p for f, p in files.items() if f not in used}
    # a gone part's name is free again: its mesh may come back under another ending (.stl -> .obj)
    clashing = [
        f
        for f in new_files
        if Path(f).stem in assembly.parts and Path(f).stem not in gone
    ]
    return UpdatePlan(folder, changed, unchanged, gone, skipped, new_files, clashing)


def check_add(plan: UpdatePlan, add: list[str]) -> None:
    unknown = [f for f in add if f not in plan.new_files]
    if unknown:
        raise ProductFolderError(
            f"Not a new mesh file of {plan.folder}: {', '.join(unknown)}"
        )
    clash = [f for f in add if f in plan.clashing]
    if clash:
        raise ProductFolderError(
            f"A part is named like {', '.join(clash)} already: rename it first"
        )


def apply_update(
    assembly,
    setup: dict,
    plan: UpdatePlan,
    add: list[str],
    mesh_dir: Path,
    progress: Callable[[int, int], None],
) -> dict:
    """Change `assembly` and `setup` (copies of the open product) as the plan says and return the report."""
    check_add(plan, add)
    update_dir = _fresh_folder(mesh_dir)
    update_dir.mkdir(parents=True)
    old_partners = {
        name: {
            n
            for c in assembly.connections
            if c.connection_type == "geometric" and name in (c.part_name1, c.part_name2)
            for n in (c.part_name1, c.part_name2)
            if n != name
        }
        for name in plan.changed
    }
    for name, source in plan.changed.items():
        old = assembly.parts[name]
        copy = update_dir / source.name
        shutil.copyfile(source, copy)
        new = Part(name, str(copy), density=old.density, material=old.material)
        new.degree_of_damage = old.degree_of_damage
        assembly.parts[name] = new  # same key: the order of the parts stays
    for name in plan.gone:
        assembly.remove_part(assembly.parts[name])
        for key, names in setup.items():
            setup[key] = [n for n in names if n != name]
    added = []
    for file in add:
        copy = update_dir / file
        shutil.copyfile(plan.new_files[file], copy)
        part = Part(Path(file).stem, str(copy))
        assembly.add_part(part)
        added.append(part.name)
    assembly._fcl_objects = {}  # rebuilt on use: a changed part kept its name
    pairs = _pairs(assembly, plan, add, old_partners)
    progress(0, len(pairs))
    for done, (first, second) in enumerate(pairs, 1):
        redetect_pair(assembly, first, second)
        progress(done, len(pairs))
    return {
        "folder": str(plan.folder),
        "changed": list(plan.changed),
        "unchanged": len(plan.unchanged),
        "removed": list(plan.gone),
        "added": added,
        "not_added": [f for f in plan.new_files if f not in add],
        "merged": list(plan.merged),
        "redetected": [list(p) for p in pairs],
    }


def redetect_pair(assembly, part1: str, part2: str) -> None:
    """Drop the geometric connections of the pair and detect them again; the parts keep their order of
    `Assembly.find_contacts`, so the connection is named like a whole-product detection would name it."""
    pair = {part1, part2}
    assembly.connections = [
        c
        for c in assembly.connections
        if not (
            c.connection_type == "geometric" and {c.part_name1, c.part_name2} == pair
        )
    ]
    first, second = (n for n in assembly.parts if n in pair)
    assembly.check_pair_for_contact_new(
        (assembly.parts[first], assembly.parts[second]),
        contact_tollerance=assembly.contact_tollerance,
        dot_product_threshold=assembly.dot_product_threshold,
    )


def report_line(report: dict) -> str:
    """The report in one line for the status log."""
    bits = [f"{len(report['changed'])} changed"]
    if report["changed"]:
        bits[-1] += f" ({', '.join(report['changed'])})"
    if report["removed"]:
        bits.append(f"removed {', '.join(report['removed'])}")
    if report["added"]:
        bits.append(f"added {', '.join(report['added'])}")
    if report["not_added"]:
        bits.append(f"{len(report['not_added'])} new mesh files not added")
    if report["merged"]:
        bits.append(f"re-merge by hand: {', '.join(report['merged'])}")
    return "Mesh update from {}: {}".format(report["folder"], "; ".join(bits))


def _pairs(
    assembly, plan: UpdatePlan, add: list[str], old_partners=None
) -> list[tuple]:
    """Part pairs to detect again: a changed part against its old partners (a contact may be gone) and against
    every part its box touches (one may be new); an added part against every part its box touches."""
    old_partners = old_partners or {}
    names = list(assembly.parts)
    fresh = [*plan.changed, *(Path(f).stem for f in add)]
    pairs = set()
    for name in fresh:
        if name not in assembly.parts:
            continue
        near = {
            other
            for other in names
            if other != name and _boxes_touch(assembly, name, other)
        }
        for other in (near | old_partners.get(name, set())) & set(names):
            pairs.add(frozenset((name, other)))
    order = {n: i for i, n in enumerate(names)}
    return sorted(
        (tuple(sorted(p, key=order.get)) for p in pairs),
        key=lambda p: (order[p[0]], order[p[1]]),
    )


def _boxes_touch(assembly, name1: str, name2: str) -> bool:
    pad = assembly.contact_tollerance
    lo1, hi1 = _box(assembly.parts[name1])
    lo2, hi2 = _box(assembly.parts[name2])
    return bool(np.all(lo1 - pad <= hi2) and np.all(lo2 - pad <= hi1))


def _box(part) -> tuple[np.ndarray, np.ndarray]:
    box = part.mesh.get_axis_aligned_bounding_box()
    return np.asarray(box.get_min_bound()), np.asarray(box.get_max_bound())


def _same_content(new: Path, current: Path) -> bool:
    try:
        return _digest(new) == _digest(current)
    except OSError:
        return False


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_folder(mesh_dir: Path) -> Path:
    for n in itertools.count(1):
        if not (mesh_dir / f"update-{n}").exists():
            return mesh_dir / f"update-{n}"
