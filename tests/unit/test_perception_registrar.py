"""The registrar and the #45 gate on clouds rendered from the angle grinder's gear housing."""

import numpy as np
import open3d as o3d
import pytest

from perception.cloud import pose_difference
from perception.quality_gate import ExtraGates, IcpGate, PlausibilityGate
from perception.registrar import REJECTED, UNIQUE_PASS, Registrar, RegistrationProfile
from tests.perception_scenes import (
    FAST_PROFILE,
    bevel_gear,
    housing,
    housing_scene,
    pose,
)

NO_RANSAC = RegistrationProfile(random_seeds=())


@pytest.fixture(autouse=True)
def single_threaded_open3d():
    """Open3D's multi-threaded RANSAC ignores its seed; one thread makes a run repeatable."""
    previous = o3d.utility.get_max_threads()
    o3d.utility.set_max_threads(1)
    yield
    o3d.utility.set_max_threads(previous)


def test_gate_defaults_are_the_legacy_check_icp_match():
    gate, plausibility = IcpGate(), PlausibilityGate()
    assert (
        gate.min_fitness,
        gate.max_rmse_m,
        gate.min_fitness_per_mm,
        gate.correspondence_m,
    ) == (
        0.7,
        0.001,
        1.5,
        0.001,
    )
    assert (plausibility.max_translation_m, plausibility.max_rotation_deg) == (
        0.005,
        3.0,
    )


@pytest.mark.parametrize(
    "offset_mm, fraction, passes",
    [
        (0.0, 1.0, True),
        (0.3, 1.0, True),  # fitness 1, RMSE 0.3 mm: ratio 3.3
        (0.8, 1.0, False),  # RMSE below 1 mm, but fitness / RMSE = 1.25
        (0.0, 0.65, False),  # fitness 0.65
    ],
)
def test_icp_gate_judges_fitness_rmse_and_their_ratio(offset_mm, fraction, passes):
    rng = np.random.default_rng(0)
    target = rng.uniform(0.0, 0.1, size=(4000, 3))
    source = target + [offset_mm / 1000.0, 0.0, 0.0]
    source[int(fraction * len(source)) :] += 0.05  # these find no target within 1 mm
    assert IcpGate().evaluate(source, target)["pass"] is passes


def test_plausibility_gate_rejects_more_than_5_mm_or_3_deg():
    nominal = pose([0, 0, 0.5], [0, 0, 0])
    gate = PlausibilityGate()
    assert gate.evaluate(pose([0.004, 0, 0.5], [0, 0, 2.5]), nominal)["pass"]
    assert not gate.evaluate(pose([0.006, 0, 0.5], [0, 0, 0]), nominal)["pass"]
    assert not gate.evaluate(pose([0, 0, 0.5], [3.5, 0, 0]), nominal)["pass"]


@pytest.mark.slow
def test_registrar_finds_the_housing_from_fresh_ransac_seeds():
    cloud, mask, truth = housing_scene()
    result = Registrar(FAST_PROFILE).register(housing(), cloud, mask=mask)
    assert result.status == UNIQUE_PASS
    translation, rotation = pose_difference(result.pose, truth)
    assert translation < 0.001 and rotation < 0.5


@pytest.mark.slow
def test_registrar_refines_a_nominal_seed_within_the_plausibility_gate():
    cloud, mask, truth = housing_scene()
    nominal = truth @ pose([0.003, -0.002, 0.0], [0, 0, 1.5])
    result = Registrar(NO_RANSAC).register(housing(), cloud, mask=mask, nominal=nominal)
    assert result.status == UNIQUE_PASS
    assert pose_difference(result.pose, truth)[0] < 0.001
    assert result.best.gate["pass"] and result.best.plausibility["pass"]


@pytest.mark.slow
def test_registrar_rejects_a_pose_far_from_nominal():
    cloud, mask, truth = housing_scene()
    nominal = truth @ pose(
        [0.0, 0.0, 0.008], [0, 0, 0]
    )  # the part sits 8 mm off its nominal place
    result = Registrar(FAST_PROFILE).register(
        housing(), cloud, mask=mask, nominal=nominal
    )
    assert result.status == REJECTED and result.pose is None
    assert result.best.gate["pass"] and not result.best.plausibility["pass"]


@pytest.mark.slow
def test_registrar_rejects_the_wrong_model_at_the_icp_gate():
    cloud, mask, truth = housing_scene()
    result = Registrar(FAST_PROFILE).register(
        bevel_gear(), cloud, mask=mask, nominal=truth
    )
    assert result.status == REJECTED and result.pose is None
    assert not any(candidate.gate["pass"] for candidate in result.candidates)


@pytest.mark.slow
def test_extra_gates_run_in_addition():
    cloud, mask, truth = housing_scene()
    result = Registrar(NO_RANSAC, extra=ExtraGates()).register(
        housing(), cloud, mask=mask, nominal=truth
    )
    assert result.status == UNIQUE_PASS
    assert result.best.extra["pass"] and result.best.extra["cad_visible_coverage"] > 0.9


@pytest.mark.slow
def test_registrar_rejects_an_empty_observation():
    cloud, mask, _ = housing_scene()
    assert (
        Registrar(FAST_PROFILE)
        .register(housing(), cloud, mask=np.zeros_like(mask))
        .status
        == REJECTED
    )
