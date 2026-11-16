"""The clamps on the desk single out one desk hypothesis when no marker is in view.

The layouts are invented from the small-clamp mesh: the repository records no clamp setup yet.
"""

import cv2
import numpy as np
import pytest

from perception.cloud import (
    OrganizedCloud,
    invert,
    pose_difference,
    raycasting_scene,
    render,
)
from perception.pose_provider import PoseProvider
from perception.registrar import Registrar, RegistrationProfile
from perception.system28 import SUPPORT_CENTER_M, TOP_Z_M
from tests.perception_scenes import (
    PINHOLE,
    T_DESK_CLAMP,
    T_DESK_HOUSING,
    T_WORLD_CAMERA,
    T_WORLD_DESK,
    clamp,
    desk,
    housing,
    pose,
)

PROVIDER = PoseProvider(Registrar(RegistrationProfile(random_seeds=())))
CENTRE = np.array(SUPPORT_CENTER_M[:2])


def _turn_about_centre(degrees) -> np.ndarray:
    """A desk-frame turn about the desk centre, which maps the square desk onto itself."""
    turn = pose([0, 0, 0], [0, 0, degrees])
    turn[:3, 3] = [*CENTRE, 0.0] - turn[:3, :3] @ [*CENTRE, 0.0]
    return turn


def _scene(true_desk_world, desk_clamps):
    """The desk and the housing at ``true_desk_world`` with each clamp on it, seen by the camera."""
    camera_world = invert(T_WORLD_CAMERA)
    meshes = [(desk().vertices_m, desk().triangles, camera_world @ true_desk_world)]
    for model, placement in [(housing(), T_DESK_HOUSING)] + [
        (clamp(), T_desk_clamp) for T_desk_clamp in desk_clamps
    ]:
        world = camera_world @ true_desk_world @ placement
        meshes.append((model.vertices_m, model.triangles, world))
    return render(raycasting_scene(meshes), PINHOLE)


def _locate(true_desk_world, nominal_clamps, *, true_clamps=None, provider=PROVIDER):
    """``nominal_clamps`` is what the pose provider is told; ``true_clamps`` what stands on the desk."""
    cloud = _scene(
        true_desk_world, nominal_clamps if true_clamps is None else true_clamps
    )
    return provider.locate(
        cloud,
        T_WORLD_CAMERA,
        product=housing(),
        nominal_world_desk=T_WORLD_DESK,
        clamps={
            f"clamp_{i}": (clamp(), placement)
            for i, placement in enumerate(nominal_clamps)
        },
    ).desk


ASYMMETRIC = [T_DESK_CLAMP]
SYMMETRIC = [
    _turn_about_centre(degrees) @ T_DESK_CLAMP for degrees in (0, 90, 180, 270)
]


@pytest.mark.parametrize(
    "turn_deg",
    [
        90,
        pytest.param(180, marks=pytest.mark.slow),
        pytest.param(270, marks=pytest.mark.slow),
    ],
)
def test_an_asymmetric_clamp_layout_resolves_the_turned_desk(turn_deg):
    truth = T_WORLD_DESK @ _turn_about_centre(turn_deg)
    desk_result = _locate(truth, ASYMMETRIC)
    assert desk_result.resolved_by == "clamps"
    translation, rotation = pose_difference(desk_result.pose, truth)
    assert translation < 0.005 and rotation < 0.5


@pytest.mark.slow
def test_a_symmetric_clamp_layout_falls_through_to_the_nominal_step():
    desk_result = _locate(T_WORLD_DESK @ _turn_about_centre(180), SYMMETRIC)
    assert desk_result.resolved_by == "nominal"


@pytest.mark.slow
def test_without_clamps_the_nominal_step_decides():
    cloud = _scene(T_WORLD_DESK, [])
    desk_result = PROVIDER.locate(
        cloud, T_WORLD_CAMERA, product=housing(), nominal_world_desk=T_WORLD_DESK
    ).desk
    assert desk_result.resolved_by == "nominal"


@pytest.mark.slow
def test_a_near_tie_falls_through_to_the_nominal_step():
    # Four clamps that look alike under every turn, and a fifth that the true turn alone supports.
    layout = SYMMETRIC + [pose([0.2, 0.1, 0.0], [0, 0, 0]) @ T_DESK_CLAMP]
    truth = T_WORLD_DESK @ _turn_about_centre(180)
    assert _locate(truth, layout).resolved_by == "nominal"
    # The same scene is decided by the clamps once the margin is small enough: no tie by accident.
    narrow = PoseProvider(PROVIDER.registrar, clamp_margin=0.1)
    assert _locate(truth, layout, provider=narrow).resolved_by == "clamps"


def test_a_clamp_off_its_nominal_pose_falls_through_to_the_nominal_step():
    moved = [pose([0.12, 0.10, 0.0], [0, 0, 0]) @ T_DESK_CLAMP]
    desk_result = _locate(
        T_WORLD_DESK @ _turn_about_centre(90), ASYMMETRIC, true_clamps=moved
    )
    assert desk_result.resolved_by == "nominal"


@pytest.mark.slow
def test_markers_keep_their_priority_over_the_clamps():
    truth = T_WORLD_DESK @ _turn_about_centre(90)
    places = {
        3: (0.205, 0.455),
        17: (0.805, 0.555),
    }  # between holes, clear of the parts
    camera_truth = invert(T_WORLD_CAMERA) @ truth
    rendered = _scene(truth, ASYMMETRIC)
    rgb = np.full(rendered.xyz_m.shape, 90, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, xy in places.items():
        point = camera_truth[:3, :3] @ [*xy, TOP_Z_M] + camera_truth[:3, 3]
        u = int(round(PINHOLE.fx * point[0] / point[2] + PINHOLE.cx))
        v = int(round(PINHOLE.fy * point[1] / point[2] + PINHOLE.cy))
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 48)
        marker = np.pad(marker, 8, constant_values=255)
        half = marker.shape[0] // 2
        rgb[
            v - half : v - half + marker.shape[0], u - half : u - half + marker.shape[1]
        ] = marker[:, :, None]
    provider = PoseProvider(PROVIDER.registrar, aruco_markers=places)
    desk_result = provider.locate(
        OrganizedCloud(rendered.xyz_m, rgb),
        T_WORLD_CAMERA,
        product=housing(),
        nominal_world_desk=T_WORLD_DESK,
        clamps={"small": (clamp(), T_DESK_CLAMP)},
    ).desk
    assert desk_result.resolved_by == "aruco"
    assert pose_difference(desk_result.pose, truth)[0] < 0.005
