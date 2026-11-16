# crc_diss_assembly_dataclasses/tests/test_copy_deepcopy.py

import os
from contextlib import contextmanager

import copy
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from diss_assembly_dataclasses.assembly import Assembly

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
SUBMODULE_ROOT = HERE.parents[1]  # -> crc_diss_assembly_dataclasses
XML_PATH = SUBMODULE_ROOT / "tests" / "emo_sr_v2" / "EMO_SR_V2.xml"


def emo_test() -> Assembly:
    if not XML_PATH.exists():
        pytest.skip(f"Missing test asset: {XML_PATH}")

    with chdir(XML_PATH.resolve().parent):
        return Assembly.load_from_xml(str(XML_PATH))

def _pick_two_part_names(asm: Assembly) -> tuple[str, str]:
    names = list(asm.parts.keys())
    assert len(names) >= 2
    return names[0], names[1]


def _assert_connection_endpoints_exist(asm: Assembly) -> None:
    names = set(asm.parts.keys())
    for c in asm.connections:
        assert c.part_name1 in names
        assert c.part_name2 in names


def test_copy_returns_new_assembly_object():
    asm = emo_test()
    asm2 = copy.copy(asm)

    assert asm2 is not asm
    assert isinstance(asm2, Assembly)
    assert asm2.name == asm.name
    assert asm2.contact_tollerance == asm.contact_tollerance
    assert asm2.dot_product_threshold == asm.dot_product_threshold


def test_copy_preserves_basic_invariants():
    asm = emo_test()
    asm2 = copy.copy(asm)

    assert set(asm2.parts.keys()) == set(asm.parts.keys())
    assert len(asm2.connections) == len(asm.connections)

    _assert_connection_endpoints_exist(asm2)


def test_copy_has_independent_containers_parts_and_connections():
    asm = emo_test()
    asm2 = copy.copy(asm)

    assert asm2.parts is not asm.parts
    assert asm2.connections is not asm.connections

    pA, _ = _pick_two_part_names(asm)

    asm2.parts.pop(pA)
    assert pA in asm.parts

    asm2.connections.clear()
    assert len(asm.connections) > 0


def test_copy_connections_container_isolated_elements_may_be_shared():
    """
    This documents a practical choice:
    - copy() creates a new connections list container
    - but may share the Connection objects for performance
    deepcopy() is responsible for isolating Connection objects.
    """
    asm = emo_test()
    asm2 = copy.copy(asm)

    if len(asm.connections) == 0:
        pytest.skip("No connections in test asset.")

    assert asm2.connections is not asm.connections
    assert asm2.connections[0] is asm.connections[0]


def test_deepcopy_is_structurally_independent():
    asm = emo_test()
    asm2 = copy.deepcopy(asm)

    assert asm2 is not asm
    assert asm2.parts is not asm.parts
    assert asm2.connections is not asm.connections

    pA, pB = _pick_two_part_names(asm)

    assert asm2.parts[pA] is not asm.parts[pA]
    assert asm2.parts[pB] is not asm.parts[pB]

    if len(asm.connections) > 0:
        assert asm2.connections[0] is not asm.connections[0]

    _assert_connection_endpoints_exist(asm2)


def test_deepcopy_mutations_do_not_affect_original():
    asm = emo_test()
    asm2 = copy.deepcopy(asm)

    pA, _ = _pick_two_part_names(asm)

    old_density = asm.parts[pA].density
    asm2.parts[pA].density = old_density + 123.0
    assert asm.parts[pA].density == old_density

    if len(asm.connections) > 0:
        old = asm.connections[0].part_name1
        asm2.connections[0].part_name1 = "__MUTATED__"
        assert asm.connections[0].part_name1 == old


def test_copy_state_hash_and_string_stable():
    """
    Copy should preserve the logical state representation.
    If this fails, copy is altering semantics (bad for planning/RL).
    """
    asm = emo_test()
    asm2 = copy.copy(asm)

    assert asm2.generate_state_string() == asm.generate_state_string()
    assert asm2.generate_hashID() == asm.generate_hashID()


def test_copy_fcl_cache_container_is_independent_and_may_be_empty():
    asm = emo_test()
    asm2 = copy.copy(asm)

    assert hasattr(asm2, "_fcl_objects")
    assert asm2._fcl_objects is not asm._fcl_objects

    # Cache may be dropped intentionally; accept empty dict.
    assert isinstance(asm2._fcl_objects, dict)

def test_copy_shares_part_objects_mutating_part_reflects_in_original():
    asm = emo_test()
    asm2 = copy.copy(asm)
    pA, _ = _pick_two_part_names(asm)

    # Shallow copy semantics: same Part object
    assert asm2.parts[pA] is asm.parts[pA]

    old_density = asm.parts[pA].density
    asm2.parts[pA].density = old_density + 1.0
    assert asm.parts[pA].density == old_density + 1.0

def test_copy_shares_connection_objects_mutating_connection_reflects_in_original():
    asm = emo_test()
    if len(asm.connections) == 0:
        pytest.skip("No connections in test asset.")
    asm2 = copy.copy(asm)

    assert asm2.connections[0] is asm.connections[0]

    old = asm.connections[0].part_name1
    asm2.connections[0].part_name1 = "__MUTATED__"
    assert asm.connections[0].part_name1 == "__MUTATED__"

    # cleanup to not poison other tests if asset is reused in-process
    asm.connections[0].part_name1 = old

def test_deepcopy_does_not_force_mesh_load():
    asm = emo_test()

    # Ensure we start from lazy state: delete meshes if any were loaded.
    asm.delete_part_meshes()

    asm2 = copy.deepcopy(asm)

    # After deepcopy, parts should still have no loaded mesh (depends on Part.__deepcopy__)
    for p in asm2.parts.values():
        assert getattr(p, "_mesh", None) is None

@st.composite
def part_key(draw, asm: Assembly):
    keys = list(asm.parts.keys())
    return draw(st.sampled_from(keys))

@given(st.integers(min_value=0, max_value=3))
@settings(max_examples=20, deadline=None)
def test_copy_container_isolation_under_random_deletions(k):
    asm = emo_test()
    asm2 = copy.copy(asm)

    keys = list(asm2.parts.keys())
    if not keys:
        pytest.skip("No parts in asset.")

    # Delete up to k keys from asm2.parts
    for name in keys[:k]:
        asm2.parts.pop(name, None)

    # Original must keep all parts
    assert set(asm.parts.keys()) == set(keys)


def test_deepcopy_of_a_shallow_copy_works():
    asm = emo_test()

    asm3 = copy.deepcopy(copy.copy(asm))

    assert set(asm3.parts) == set(asm.parts)
