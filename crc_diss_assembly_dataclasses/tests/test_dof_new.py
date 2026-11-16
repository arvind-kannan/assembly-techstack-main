# crc_diss_assembly_dataclasses/tests/test_dof_new.py


import os
from contextlib import contextmanager
from pathlib import Path

import inspect
import copy
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.connection import evenly_spaced_points_on_sphere, Screw_Connection


@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


HERE = Path(__file__).resolve()
# crc_diss_assembly_dataclasses/tests/...
# pomdp_disassembly_gym/tests/test_dof/test_dof.py
SUBMODULE_ROOT = HERE.parents[2]  # -> pomdp_disassembly_gym
XML_PATH = SUBMODULE_ROOT / "emo_sr_v2" / "EMO_SR_V2.xml"

@pytest.fixture
def emo_test() -> Assembly:
    if not XML_PATH.exists():
        pytest.skip(f"Missing test asset: {XML_PATH}")

    # Important: XML references relative .npy assets -> load with cwd at XML folder
    with chdir(XML_PATH.resolve().parent):
        return Assembly.load_from_xml(str(XML_PATH))
    
# ----------------------------
# Helpers
# ----------------------------

UPPER = "01_upper_lid"
STATOR = "02_stator"
ROTOR = "03_rotor_short"
SCREWS = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]


def _remove_part_by_name(asm: Assembly, name: str) -> None:
    p = asm.get_part_by_name(name)
    assert p is not None, f"Expected part '{name}' to exist before removal"
    asm.remove_part(p)
    assert asm.get_part_by_name(name) is None, f"Part '{name}' should be removed"


def _dof_count(asm: Assembly, part_name: str) -> int:
    p = asm.get_part_by_name(part_name)
    assert p is not None, f"Part '{part_name}' missing in assembly"
    dof = asm.get_part_dof(p)
    return 0 if not dof else len(dof)


def _assert_no_dof(asm: Assembly, part_name: str, *, ctx: str) -> None:
    n = _dof_count(asm, part_name)
    assert n == 0, f"[{ctx}] Expected NO DOF for {part_name}, but got {n}"


def _assert_has_dof(asm: Assembly, part_name: str, *, ctx: str) -> None:
    n = _dof_count(asm, part_name)
    assert n > 0, f"[{ctx}] Expected SOME DOF for {part_name}, but got {n}"


# ----------------------------
# Tests
# ----------------------------

@pytest.mark.integration
def test_G0_any_screw_present_non_screw_parts_have_no_dof(emo_test: Assembly):
    """
    Regression guard (your key bug):
    While any screw part exists, non-screw parts (upper lid / rotor / stator)
    must NOT become removable via any DOF direction.

    If this fails, the agent can "grip-remove" upper/rotor/stator too early.
    """
    asm = copy.deepcopy(emo_test)

    # Sanity: screws exist initially
    for s in SCREWS:
        assert asm.get_part_by_name(s) is not None

    # Strong invariant: no DOF for non-screw parts while screws exist
    _assert_no_dof(asm, UPPER, ctx="initial screws present")
    _assert_no_dof(asm, ROTOR, ctx="initial screws present")
    _assert_no_dof(asm, STATOR, ctx="initial screws present")


@pytest.mark.integration
def test_G1_after_all_screws_removed_upper_lid_has_dof(emo_test: Assembly):
    """
    After removing ALL screw parts, upper lid should regain at least one DOF.
    Otherwise disassembly will deadlock (cannot progress to lid removal).
    """
    asm = copy.deepcopy(emo_test)

    for s in SCREWS:
        _remove_part_by_name(asm, s)

    # Now upper lid should become removable (at least one DOF direction)
    _assert_has_dof(asm, UPPER, ctx="all screws removed")


@pytest.mark.integration
def test_G2_after_screws_removed_rotor_still_blocked_until_upper_lid_removed(emo_test: Assembly):
    """
    Another early-removal bug guard:
    Even if screws are removed, rotor should still NOT be removable while upper lid exists.
    (Typical intended order: screws -> upper lid -> rotor.)
    """
    asm = copy.deepcopy(emo_test)

    # Remove all screws first
    for s in SCREWS:
        _remove_part_by_name(asm, s)

    # Upper lid present -> rotor should still have no DOF
    assert asm.get_part_by_name(UPPER) is not None
    _assert_no_dof(asm, ROTOR, ctx="screws removed but upper lid present")

    # Once upper lid removed -> rotor should regain DOF
    _remove_part_by_name(asm, UPPER)
    _assert_has_dof(asm, ROTOR, ctx="screws removed + upper lid removed")


@pytest.mark.integration
def test_G3_stator_still_blocked_until_upper_lid_removed(emo_test: Assembly):
    """
    Optional guard similar to rotor:
    Stator should not become removable while upper lid exists.
    (Depending on your intended order, keep or drop this test.)
    """
    asm = copy.deepcopy(emo_test)

    # Remove all screws first
    for s in SCREWS:
        _remove_part_by_name(asm, s)

    # Upper lid still present -> stator should have no DOF
    assert asm.get_part_by_name(UPPER) is not None
    _assert_no_dof(asm, STATOR, ctx="screws removed but upper lid present")

    # After upper lid removed -> stator may or may not become removable depending on your model.
    # If your intended order is screws -> upper -> rotor -> stator, stator might still be blocked.
    # So here we only assert it does NOT become removable *before* upper lid removal.

@pytest.mark.integration
def test_G4_screws_have_dof_initially(emo_test):
    asm = copy.deepcopy(emo_test)

    # Sanity: all screws exist
    for s in SCREWS:
        assert asm.get_part_by_name(s) is not None, f"Missing screw part in assembly: {s}"

    # Print DOF vectors for each screw
    dof_map: dict[str, list[list[float]]] = {}
    for s in SCREWS:
        part = asm.get_part_by_name(s)
        dofs = asm.get_part_dof(part)

        # Pretty formatting: round + convert to plain Python lists
        pretty = [np.asarray(v, dtype=float).round(5).tolist() for v in dofs]
        dof_map[s] = pretty

    # Also print counts (you already had this)
    counts = {s: len(dof_map[s]) for s in SCREWS}

    print("[G4] screw DOF counts:", counts)
    for s in SCREWS:
        print(f"[G4] {s} DOF vectors ({counts[s]}): {dof_map[s]}")

    # Assertions
    for s, n in counts.items():
        assert n > 0, f"Expected screw {s} to have DOF > 0 initially, but got {n}"
    """
    [G4] screw DOF counts: {'04_screw_1': 4, '05_screw_2': 4, '06_screw_3': 4, '07_screw_4': 4}
    [G4] 04_screw_1 DOF vectors (4): [[-0.14404, -0.12628, 0.98148], [-0.14125, 0.12939, 0.98148], [0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383]]
    [G4] 05_screw_2 DOF vectors (4): [[-0.14404, -0.12628, 0.98148], [-0.14125, 0.12939, 0.98148], [0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383]]
    [G4] 06_screw_3 DOF vectors (4): [[-0.14404, -0.12628, 0.98148], [-0.14125, 0.12939, 0.98148], [0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383]]
    [G4] 07_screw_4 DOF vectors (4): [[-0.14404, -0.12628, 0.98148], [-0.14125, 0.12939, 0.98148], [0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383]]
    """


def _print_dof(asm, part_name: str, *, tag: str) -> None:
    """Pretty print DOF vectors for a part (rounded), sorted for stable output."""
    part = asm.get_part_by_name(part_name)
    if part is None:
        print(f"[{tag}] {part_name}: <missing>")
        return

    dof = asm.get_part_dof(part)
    # normalize defensively + round for readability
    vecs = []
    for v in dof:
        v = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v = v / n
        vecs.append(v)

    # stable sort: z desc, then x/y
    vecs = sorted(vecs, key=lambda x: (x[2], x[0], x[1]), reverse=True)

    rounded = [np.round(v, 5).tolist() for v in vecs]
    print(f"[{tag}] {part_name} DOF vectors ({len(rounded)}): {rounded}")


@pytest.mark.integration
def test_G5_upper_lid_dof_changes_before_after_screws_removed(emo_test):
    """
    G5:
    Print upper lid DOF count + vectors:
      - initially (screws present)
      - after all screws removed
    This helps verify that screws really 'unlock' upper lid.
    """
    asm = copy.deepcopy(emo_test)

    # --- before: screws present ---
    n0 = _dof_count(asm, UPPER)
    print(f"[G5] upper lid DOF count (screws present): {n0}")
    _print_dof(asm, UPPER, tag="G5/before")

    # --- remove all screws ---
    for s in SCREWS:
        _remove_part_by_name(asm, s)

    # --- after: screws removed ---
    n1 = _dof_count(asm, UPPER)
    print(f"[G5] upper lid DOF count (after screws removed): {n1}")
    _print_dof(asm, UPPER, tag="G5/after")

    # Assertions (lightweight, but meaningful):
    # 1) upper lid should not be missing
    assert asm.get_part_by_name(UPPER) is not None
    # 2) DOF should (typically) increase or at least not get worse
    #    If you want to be stricter: change to `assert n1 > 0`.
    assert n1 >= n0, f"[G5] Expected upper lid DOF to not decrease after screws removed, got {n0} -> {n1}"
    """
    [G5] upper lid DOF count (screws present): 0
    [G5/before] 01_upper_lid DOF vectors (0): []
    [G5] upper lid DOF count (after screws removed): 4
    [G5/after] 01_upper_lid DOF vectors (4): [[0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383], [-0.14125, 0.12939, 0.98148], [-0.14404, -0.12628, 0.98148]]
    """


@pytest.mark.integration
def test_G6_rotor_dof_changes_before_after_upper_lid_removed(emo_test):
    """
    G6:
    Print rotor DOF count + vectors:
      - with upper lid present (typically blocked)
      - after upper lid removed (should open up)
    """
    asm = copy.deepcopy(emo_test)

    # Make the comparison meaningful:
    # remove screws first (otherwise rotor DOF may be dominated by screw-lock logic)
    for s in SCREWS:
        _remove_part_by_name(asm, s)

    assert asm.get_part_by_name(UPPER) is not None
    assert asm.get_part_by_name(ROTOR) is not None

    # --- before: upper lid present ---
    n0 = _dof_count(asm, ROTOR)
    print(f"[G6] rotor DOF count (upper lid present): {n0}")
    _print_dof(asm, ROTOR, tag="G6/before")

    # --- remove upper lid ---
    _remove_part_by_name(asm, UPPER)

    assert asm.get_part_by_name(UPPER) is None
    assert asm.get_part_by_name(ROTOR) is not None

    # --- after: upper lid removed ---
    n1 = _dof_count(asm, ROTOR)
    print(f"[G6] rotor DOF count (after upper lid removed): {n1}")
    _print_dof(asm, ROTOR, tag="G6/after")

    # Assertions:
    # rotor should become "more removable" after upper lid is gone
    # (If your intended order differs, relax/change this.)
    assert n1 >= n0, f"[G6] Expected rotor DOF to not decrease after upper lid removed, got {n0} -> {n1}"

    # assert n1 > 0, f"[G6] Expected rotor DOF > 0 after upper lid removed, got {n1}"
    """
    [G6] rotor DOF count (upper lid present): 0
    [G6/before] 03_rotor_short DOF vectors (0): []
    [G6] rotor DOF count (after upper lid removed): 4
    [G6/after] 03_rotor_short DOF vectors (4): [[0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383], [-0.14125, 0.12939, 0.98148], [-0.14404, -0.12628, 0.98148]]
    """

from diss_assembly_dataclasses.connection import Fixed_Connection

def _print_all_parts_dof(asm, *, tag: str) -> None:
    """
    Print DOF count + vector list for every part currently present in asm.
    Output is sorted by part name for stable logs.
    """
    names = sorted([p.name for p in asm.parts.values()])
    counts = {}
    for name in names:
        part = asm.get_part_by_name(name)
        dof = asm.get_part_dof(part) if part is not None else []
        counts[name] = len(dof)

    print(f"[{tag}] DOF counts:", counts)

    # Print vectors (rounded) for each part
    for name in names:
        part = asm.get_part_by_name(name)
        if part is None:
            print(f"[{tag}] {name}: <missing>")
            continue
        dof = asm.get_part_dof(part)
        vecs = []
        for v in dof:
            v = np.asarray(v, dtype=float).reshape(3)
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                v = v / n
            vecs.append(v)

        # stable sort: z desc, then x, y
        vecs = sorted(vecs, key=lambda x: (x[2], x[0], x[1]), reverse=True)
        rounded = [np.round(v, 5).tolist() for v in vecs]
        print(f"[{tag}] {name} DOF vectors ({len(rounded)}): {rounded}")


@pytest.mark.integration
def test_G7_inject_fixed_screw_to_lowerlid_then_remove_other_screws(emo_test):
    """
    G7:
    1) Inject a Fixed_Connection between one screw and lower lid.
       Print DOF count + list for all parts.
    2) Remove the other 3 screws (keep the fixed screw).
       Print DOF count + list for remaining parts.
    """
    asm = copy.deepcopy(emo_test)

    # Choose "any" screw deterministically for reproducibility
    fixed_screw = "04_screw_1"
    lower = "08_lower_lid"
    other_screws = [s for s in SCREWS if s != fixed_screw]

    # Sanity
    assert asm.get_part_by_name(fixed_screw) is not None, f"Missing {fixed_screw}"
    assert asm.get_part_by_name(lower) is not None, f"Missing {lower}"

    # 1) Inject fixed connection
    asm.add_connection(Fixed_Connection(fixed_screw, lower))

    # Assert it really exists (avoid silent no-op)
    assert any(
        getattr(c, "connection_type", None) == "fixed"
        and (
            (getattr(c, "part_name1", None) == fixed_screw and getattr(c, "part_name2", None) == lower)
            or (getattr(c, "part_name1", None) == lower and getattr(c, "part_name2", None) == fixed_screw)
        )
        for c in asm.connections
    ), "[G7] Failed to inject fixed connection into assembly.connections"

    _print_all_parts_dof(asm, tag="G7/after_inject_fixed")

    # 2) Remove the other screws
    for s in other_screws:
        _remove_part_by_name(asm, s)
        assert asm.get_part_by_name(s) is None, f"[G7] Expected screw removed: {s}"

    # Keep fixed screw present
    assert asm.get_part_by_name(fixed_screw) is not None, f"[G7] Fixed screw unexpectedly missing: {fixed_screw}"
    assert asm.get_part_by_name(lower) is not None, f"[G7] Lower lid unexpectedly missing: {lower}"

    _print_all_parts_dof(asm, tag="G7/after_remove_other_screws")
"""
[G7/after_inject_fixed] DOF counts: {'01_upper_lid': 0, '02_stator': 0, '03_rotor_short': 0, '04_screw_1': 0, '05_screw_2': 4, '06_screw_3': 4, '07_screw_4': 4, '08_lower_lid': 0}
[G7/after_inject_fixed] 01_upper_lid DOF vectors (0): []
[G7/after_inject_fixed] 02_stator DOF vectors (0): []
[G7/after_inject_fixed] 03_rotor_short DOF vectors (0): []
[G7/after_inject_fixed] 04_screw_1 DOF vectors (0): []
[G7/after_inject_fixed] 05_screw_2 DOF vectors (4): [[0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383], [-0.14125, 0.12939, 0.98148], [-0.14404, -0.12628, 0.98148]]
[G7/after_inject_fixed] 06_screw_3 DOF vectors (4): [[0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383], [-0.14125, 0.12939, 0.98148], [-0.14404, -0.12628, 0.98148]]
[G7/after_inject_fixed] 07_screw_4 DOF vectors (4): [[0.11091, -0.00242, 0.99383], [0.11094, 0.0, 0.99383], [-0.14125, 0.12939, 0.98148], [-0.14404, -0.12628, 0.98148]]
[G7/after_inject_fixed] 08_lower_lid DOF vectors (0): []
[G7/after_remove_other_screws] DOF counts: {'01_upper_lid': 0, '02_stator': 0, '03_rotor_short': 0, '04_screw_1': 0, '08_lower_lid': 0}
[G7/after_remove_other_screws] 01_upper_lid DOF vectors (0): []
[G7/after_remove_other_screws] 02_stator DOF vectors (0): []
[G7/after_remove_other_screws] 03_rotor_short DOF vectors (0): []
[G7/after_remove_other_screws] 04_screw_1 DOF vectors (0): []
[G7/after_remove_other_screws] 08_lower_lid DOF vectors (0): []
"""