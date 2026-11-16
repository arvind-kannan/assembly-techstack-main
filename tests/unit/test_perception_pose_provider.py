"""The pose provider on a simulated cell: product, desk and clamp poses in the world frame."""

import pytest

from perception.cloud import pose_difference
from perception.pose_provider import PoseProvider
from perception.registrar import REJECTED, UNIQUE_PASS, Registrar, RegistrationProfile
from tests.perception_scenes import (
    T_DESK_CLAMP,
    T_DESK_HOUSING,
    T_WORLD_CAMERA,
    T_WORLD_DESK,
    cell_scene,
    clamp,
    housing,
    pose,
)

pytestmark = pytest.mark.slow  # renders the simulation meshes (Open3D raycasting)

# Nominal seeds only: the RANSAC path is covered in test_perception_registrar.
PROVIDER = PoseProvider(Registrar(RegistrationProfile(random_seeds=())))
TRUE_HOUSING = T_WORLD_DESK @ T_DESK_HOUSING
TRUE_CLAMP = T_WORLD_DESK @ T_DESK_CLAMP


def _locate(housing_offset):
    return PROVIDER.locate(
        cell_scene(True),
        T_WORLD_CAMERA,
        product=housing(),
        nominal_world_product=TRUE_HOUSING @ housing_offset,
        nominal_world_desk=T_WORLD_DESK @ pose([0.01, -0.01, 0.0], [0, 0, 2]),
        clamps={
            "small": (clamp(), T_DESK_CLAMP @ pose([0.003, 0.002, 0.0], [0, 0, 1]))
        },
    )


def test_pose_provider_returns_product_desk_and_clamp_in_the_world_frame():
    scene = _locate(pose([0.002, -0.002, 0.001], [0, 0, 1]))
    assert scene.product.status == UNIQUE_PASS
    assert pose_difference(scene.T_world_product, TRUE_HOUSING)[0] < 0.001
    # One clamp is asymmetric under the desk turns, so the clamps decide.
    assert scene.desk.resolved_by == "clamps"
    translation, rotation = pose_difference(scene.T_world_desk, T_WORLD_DESK)
    assert translation < 0.002 and rotation < 0.2
    assert scene.clamps["small"].status == UNIQUE_PASS
    assert pose_difference(scene.T_world_clamps["small"], TRUE_CLAMP)[0] < 0.002


def test_pose_provider_rejects_a_product_off_its_nominal_place():
    scene = _locate(pose([0.0, 0.009, 0.0], [0, 0, 0]))
    assert scene.product.status == REJECTED and scene.T_world_product is None
    assert scene.desk.pose is not None and scene.clamps["small"].status == UNIQUE_PASS


def test_pose_provider_without_a_desk_rejects_the_clamps():
    from perception.cloud import raycasting_scene, render
    from tests.perception_scenes import PINHOLE, plane_mesh

    flat = render(raycasting_scene([plane_mesh(1.0)]), PINHOLE)
    scene = PROVIDER.locate(
        flat,
        T_WORLD_CAMERA,
        product=housing(),
        clamps={"small": (clamp(), T_DESK_CLAMP)},
    )
    assert scene.desk.pose is None and scene.desk.error
    assert scene.product.status == REJECTED
    assert scene.clamps["small"].status == REJECTED


def test_pose_provider_rejects_a_desk_fit_at_the_desk_gate():
    from perception.system28 import DeskGate

    strict = PoseProvider(
        PROVIDER.registrar, desk_gate=DeskGate(max_residual_rmse_m=1e-4)
    )
    scene = strict.locate(
        cell_scene(True),
        T_WORLD_CAMERA,
        product=housing(),
        nominal_world_product=TRUE_HOUSING,
        nominal_world_desk=T_WORLD_DESK,
        clamps={"small": (clamp(), T_DESK_CLAMP)},
    )
    assert scene.desk.pose is None and "gate" in scene.desk.error
    assert scene.clamps["small"].status == REJECTED
    assert (
        scene.product.status == UNIQUE_PASS
    )  # the product does not depend on the desk
