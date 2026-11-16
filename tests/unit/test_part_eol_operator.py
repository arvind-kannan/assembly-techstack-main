from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from diss_assembly_dataclasses.assembly import Assembly

from tests.conftest import FakePart

# ---------------------------------------------------------------------
# Robust import helper
# ---------------------------------------------------------------------

MODULE_CANDIDATES = [
    "disassembly_pomdp.part_eol_operator",
    "disassembly_pomdp.part_eol_operator_v2",
    "disassembly_pomdp.eol_operator",
    "disassembly_pomdp.eol_operator_v2",
]


def _load_eol_module():
    last_error = None
    for module_name in MODULE_CANDIDATES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            last_error = exc

    raise ModuleNotFoundError(
        "Could not import the EOL operator module. "
        "Please add the actual module path to MODULE_CANDIDATES in this test file."
    ) from last_error


_eol_module = _load_eol_module()
PartEOLOperatorV2 = _eol_module.PartEOLOperatorV2
EOLPartEvent = _eol_module.EOLPartEvent
EOLUnifiedSampler = _eol_module.EOLUnifiedSampler
from disassembly_pomdp.stuck_part_rules import StuckPartRule, StuckPartRules

RULES = StuckPartRules(
    (
        StuckPartRule(part_prefixes=("04_screw_",), prior="screw"),
        StuckPartRule(parts=("01_upper_lid",), prior="upper"),
    )
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def simple_eol_assembly():
    asm = Assembly("simple_eol_assembly")
    asm.parts = {
        "part_1": FakePart("part_1"),
        "part_2": FakePart("part_2"),
        "04_screw_a": FakePart("04_screw_a"),
        "01_upper_lid": FakePart("01_upper_lid"),
    }
    asm.connections = []
    asm._fcl_objects = {}
    return asm


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------


def test_constructor_accepts_valid_parameters():
    op = PartEOLOperatorV2(
        p_free_success=0.99,
        p_stuck_success=0.05,
        default_p_stuck=0.2,
        clip_eps=1e-6,
        min_update_p_exec=1e-3,
    )

    assert op.p_free_success == 0.99
    assert op.p_stuck_success == 0.05
    assert op.default_p_stuck == 0.2
    assert op.freeze_updates is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_free_success": -0.1},
        {"p_free_success": 1.1},
        {"p_stuck_success": -0.1},
        {"p_stuck_success": 1.1},
        {"default_p_stuck": -0.1},
        {"default_p_stuck": 1.1},
        {"clip_eps": 0.0},
        {"clip_eps": 0.6},
        {"min_update_p_exec": -0.1},
        {"p_free_success": 0.2, "p_stuck_success": 0.5},
    ],
)
def test_constructor_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        PartEOLOperatorV2(**kwargs)


# ---------------------------------------------------------------------
# Reset, priors, snapshot, restore
# ---------------------------------------------------------------------


def test_reset_for_parts_sets_default_priors(simple_eol_assembly, rng):
    op = PartEOLOperatorV2(default_p_stuck=0.25)

    op.reset_for_parts(
        ["part_1", "part_2", "part_1"],
        rng=rng,
        assembly=simple_eol_assembly,
    )

    snap = op.snapshot()

    assert snap["p_stuck"] == {
        "part_1": 0.25,
        "part_2": 0.25,
    }
    assert snap["p_missing"] == {
        "part_1": 0.0,
        "part_2": 0.0,
    }


def test_reset_for_parts_uses_inspection_prior_sampler(simple_eol_assembly, rng):
    op = PartEOLOperatorV2(default_p_stuck=0.2)

    def sampler(part_name, assembly, rng):
        if part_name == "part_1":
            return 0.8
        return 0.1

    op.reset_for_parts(
        ["part_1", "part_2"],
        rng=rng,
        assembly=simple_eol_assembly,
        inspection_prior_sampler=sampler,
    )

    assert op.get_p_stuck("part_1") == pytest.approx(0.8)
    assert op.get_p_stuck("part_2") == pytest.approx(0.1)
    assert op.get_p_missing("part_1") == pytest.approx(0.0)


def test_reset_for_parts_clips_bad_sampler_values(simple_eol_assembly, rng):
    op = PartEOLOperatorV2(default_p_stuck=0.2)

    def sampler(part_name, assembly, rng):
        if part_name == "too_high":
            return 10.0
        if part_name == "nan_part":
            return float("nan")
        raise RuntimeError("sampler failed")

    op.reset_for_parts(
        ["too_high", "nan_part", "exception_part"],
        rng=rng,
        assembly=simple_eol_assembly,
        inspection_prior_sampler=sampler,
    )

    assert op.get_p_stuck("too_high") == pytest.approx(1.0)
    assert op.get_p_stuck("nan_part") == pytest.approx(0.2)
    assert op.get_p_stuck("exception_part") == pytest.approx(0.2)


def test_prior_from_assembly_fixed_detects_fixed_connection(simple_eol_assembly, rng):
    fixed_conn = SimpleNamespace(
        connection_type="fixed",
        part_name1="part_1",
        part_name2="part_2",
    )
    simple_eol_assembly.connections = [fixed_conn]

    p_fixed = PartEOLOperatorV2.prior_from_assembly_fixed(
        "part_1",
        simple_eol_assembly,
        rng,
        p_fixed=0.95,
        p_free=0.15,
    )
    p_free = PartEOLOperatorV2.prior_from_assembly_fixed(
        "unconnected_part",
        simple_eol_assembly,
        rng,
        p_fixed=0.95,
        p_free=0.15,
    )

    assert p_fixed == pytest.approx(0.95)
    assert p_free == pytest.approx(0.15)


def test_snapshot_and_restore_clip_values_and_default_missing():
    op = PartEOLOperatorV2(default_p_stuck=0.2)

    op.restore(
        {
            "p_stuck": {
                "part_1": -1.0,
                "part_2": 2.0,
                "part_3": 0.4,
            },
            "p_missing": {
                "part_1": -0.5,
                "part_2": 1.5,
                # part_3 intentionally missing
            },
        }
    )

    snap = op.snapshot()

    assert snap["p_stuck"]["part_1"] == pytest.approx(0.0)
    assert snap["p_stuck"]["part_2"] == pytest.approx(1.0)
    assert snap["p_stuck"]["part_3"] == pytest.approx(0.4)

    assert snap["p_missing"]["part_1"] == pytest.approx(0.0)
    assert snap["p_missing"]["part_2"] == pytest.approx(1.0)
    assert snap["p_missing"]["part_3"] == pytest.approx(0.0)


def test_getters_return_defaults_for_unknown_part():
    op = PartEOLOperatorV2(default_p_stuck=0.33)

    assert op.get_p_stuck("unknown") == pytest.approx(0.33)
    assert op.get_p_missing("unknown") == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Expected success probability
# ---------------------------------------------------------------------


def test_expected_success_prob_combines_missing_stuck_and_exec():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.25,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.25},
            "p_missing": {"part_1": 0.2},
        }
    )

    p = op.expected_success_prob("part_1", p_exec=0.5)

    # p_exist * ((1-ps)*p_exec*p_free_success + ps*p_exec*p_stuck_success)
    expected = 0.8 * ((1.0 - 0.25) * 0.5 * 0.9 + 0.25 * 0.5 * 0.1)

    assert p == pytest.approx(expected)


def test_expected_success_prob_is_zero_when_missing_probability_is_one():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.2,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.2},
            "p_missing": {"part_1": 1.0},
        }
    )

    assert op.expected_success_prob("part_1", p_exec=1.0) == pytest.approx(0.0)


def test_success_given_free_and_stuck_are_scaled_by_p_exec():
    op = PartEOLOperatorV2(
        p_free_success=0.8,
        p_stuck_success=0.2,
    )

    assert op._p_success_given_free(0.5) == pytest.approx(0.4)
    assert op._p_success_given_stuck(0.5) == pytest.approx(0.1)


# ---------------------------------------------------------------------
# Missing observation update
# ---------------------------------------------------------------------


def test_on_missing_observed_sets_high_missing_probability_and_caps_stuck():
    op = PartEOLOperatorV2(default_p_stuck=0.9, clip_eps=1e-4)
    op.restore(
        {
            "p_stuck": {"part_1": 0.9},
            "p_missing": {"part_1": 0.0},
        }
    )

    op.on_missing_observed("part_1")

    assert op.get_p_missing("part_1") == pytest.approx(1.0 - 1e-4)
    assert op.get_p_stuck("part_1") == pytest.approx(1e-4)
    assert op.get_p_stuck("part_1") <= 1.0 - op.get_p_missing("part_1") + op.clip_eps


def test_update_on_missing_part_is_gated():
    op = PartEOLOperatorV2(default_p_stuck=0.4)
    op.restore(
        {
            "p_stuck": {"part_1": 0.4},
            "p_missing": {"part_1": 0.9},
        }
    )

    ps_after = op.update_on_outcome("part_1", observed_success=False, p_exec=1.0)

    assert ps_after == pytest.approx(0.4)
    assert op.get_p_stuck("part_1") == pytest.approx(0.4)


# ---------------------------------------------------------------------
# Bayesian stuck-belief update
# ---------------------------------------------------------------------


def test_update_on_success_decreases_p_stuck_when_free_success_is_higher():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.5,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.5},
            "p_missing": {"part_1": 0.0},
        }
    )

    ps_after = op.update_on_outcome("part_1", observed_success=True, p_exec=1.0)

    assert ps_after < 0.5


def test_update_on_failure_increases_p_stuck_when_free_success_is_higher():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.2,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.2},
            "p_missing": {"part_1": 0.0},
        }
    )

    ps_after = op.update_on_outcome("part_1", observed_success=False, p_exec=1.0)

    assert ps_after > 0.2


def test_update_is_gated_when_p_exec_is_too_low():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.3,
        min_update_p_exec=0.5,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.3},
            "p_missing": {"part_1": 0.0},
        }
    )

    ps_after = op.update_on_outcome("part_1", observed_success=False, p_exec=0.1)

    assert ps_after == pytest.approx(0.3)
    assert op.get_p_stuck("part_1") == pytest.approx(0.3)


def test_update_clips_p_exec_to_valid_range():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.3,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.3},
            "p_missing": {"part_1": 0.0},
        }
    )

    ps_after = op.update_on_outcome("part_1", observed_success=True, p_exec=10.0)

    assert op.clip_eps <= ps_after <= 1.0 - op.clip_eps


def test_freeze_updates_prevents_belief_update():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.4,
        freeze_updates=True,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.4},
            "p_missing": {"part_1": 0.0},
        }
    )

    event = op.on_attempt("part_1", success=False, p_exec=1.0)

    assert event.p_stuck_prior == pytest.approx(0.4)
    assert event.p_stuck_post == pytest.approx(0.4)
    assert op.get_p_stuck("part_1") == pytest.approx(0.4)


# ---------------------------------------------------------------------
# on_attempt event record
# ---------------------------------------------------------------------


def test_on_attempt_returns_event_with_prior_and_post_values():
    op = PartEOLOperatorV2(
        p_free_success=0.9,
        p_stuck_success=0.1,
        default_p_stuck=0.5,
    )
    op.restore(
        {
            "p_stuck": {"part_1": 0.5},
            "p_missing": {"part_1": 0.0},
        }
    )

    event = op.on_attempt("part_1", success=True, p_exec=1.0)

    assert isinstance(event, EOLPartEvent)
    assert event.part == "part_1"
    assert event.success is True
    assert event.p_exec == pytest.approx(1.0)
    assert event.p_stuck_prior == pytest.approx(0.5)
    assert event.p_stuck_post < event.p_stuck_prior
    assert event.p_success_prior >= 0.0
    assert event.p_success_post >= 0.0


def test_on_attempt_success_marks_part_as_existing_not_missing():
    op = PartEOLOperatorV2(clip_eps=1e-5)
    op.restore(
        {
            "p_stuck": {"part_1": 0.2},
            "p_missing": {"part_1": 0.8},
        }
    )

    # Since p_missing > 0.5, update itself is gated,
    # but success still means the part existed at attempt time.
    op.on_attempt("part_1", success=True, p_exec=1.0)

    assert op.get_p_missing("part_1") == pytest.approx(1e-5)


def test_on_attempt_clips_p_exec_in_event():
    op = PartEOLOperatorV2(default_p_stuck=0.2)

    event_low = op.on_attempt("part_1", success=False, p_exec=-1.0)
    event_high = op.on_attempt("part_1", success=False, p_exec=2.0)

    assert event_low.p_exec == pytest.approx(0.0)
    assert event_high.p_exec == pytest.approx(1.0)


# ---------------------------------------------------------------------
# EOLUnifiedSampler
# ---------------------------------------------------------------------


def test_unified_sampler_returns_expected_stuck_priors(simple_eol_assembly, rng):
    sampler = EOLUnifiedSampler(
        p_stuck_screw=0.05,
        p_stuck_upper=0.25,
        p_missing_screw=0.02,
        rules=RULES,
    )

    assert sampler.sample_stuck(
        "04_screw_a", simple_eol_assembly, rng
    ) == pytest.approx(0.05)
    assert sampler.sample_stuck(
        "01_upper_lid", simple_eol_assembly, rng
    ) == pytest.approx(0.25)
    assert sampler.sample_stuck("part_1", simple_eol_assembly, rng) == pytest.approx(
        0.0
    )


def test_unified_sampler_returns_expected_missing_priors(simple_eol_assembly, rng):
    sampler = EOLUnifiedSampler(
        p_stuck_screw=0.05,
        p_stuck_upper=0.25,
        p_missing_screw=0.02,
        rules=RULES,
    )

    assert sampler.sample_missing(
        "04_screw_a", simple_eol_assembly, rng
    ) == pytest.approx(0.02)
    assert sampler.sample_missing(
        "01_upper_lid", simple_eol_assembly, rng
    ) == pytest.approx(0.0)
    assert sampler.sample_missing("part_1", simple_eol_assembly, rng) == pytest.approx(
        0.0
    )
