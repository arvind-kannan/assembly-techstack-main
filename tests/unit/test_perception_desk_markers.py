"""Desk markers (ArUco and Hao's white shapes) single out one desk hypothesis."""

import cv2
import numpy as np
import pytest

from perception.cloud import OrganizedCloud, fit_table_plane, invert, pose_difference
from perception.desk_markers import DeclaredShape, choose_by_aruco, choose_by_shapes
from perception.system28 import TOP_Z_M, localize_desk
from tests.perception_scenes import PINHOLE, T_WORLD_CAMERA, T_WORLD_DESK, cell_scene

pytestmark = pytest.mark.slow  # renders the simulation meshes (Open3D raycasting)

TRUTH = invert(T_WORLD_CAMERA) @ T_WORLD_DESK
# Between holes, clear of the housing and the clamp, in view.
PLACES = {"a": (0.205, 0.455), "b": (0.805, 0.555)}


def _pixel(desk_xy):
    point = TRUTH[:3, :3] @ [*desk_xy, TOP_Z_M] + TRUTH[:3, 3]
    return (
        int(round(PINHOLE.fx * point[0] / point[2] + PINHOLE.cx)),
        int(round(PINHOLE.fy * point[1] / point[2] + PINHOLE.cy)),
    )


def _hypotheses(cloud):
    normal, offset = fit_table_plane(cloud.points())
    return localize_desk(cloud, normal, offset)


def _with_image(paint):
    rendered = cell_scene(True)
    rgb = np.full(rendered.xyz_m.shape, 90, dtype=np.uint8)
    for name, place in PLACES.items():
        paint(rgb, name, _pixel(place))
    return OrganizedCloud(rendered.xyz_m, rgb)


def test_aruco_markers_pick_the_true_desk_pose():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ids = {"a": 3, "b": 17}

    def paint(rgb, name, centre):
        marker = cv2.aruco.generateImageMarker(dictionary, ids[name], 48)
        marker = np.pad(marker, 8, constant_values=255)
        half = marker.shape[0] // 2
        u, v = centre
        rgb[
            v - half : v - half + marker.shape[0], u - half : u - half + marker.shape[1]
        ] = marker[:, :, None]

    cloud = _with_image(paint)
    hypotheses = _hypotheses(cloud)
    assert len(hypotheses.poses) > 1  # the hole grid alone is ambiguous here
    choice = choose_by_aruco(
        cloud, hypotheses, {ids[n]: xy for n, xy in PLACES.items()}
    )
    assert choice.index is not None and len(choice.seen) == 2
    assert pose_difference(hypotheses.poses[choice.index], TRUTH)[0] < 0.002


def test_white_shapes_pick_the_true_desk_pose():
    def paint(rgb, name, centre):
        if name == "a":
            cv2.circle(rgb, centre, 14, (255, 255, 255), -1)
        else:
            u, v = centre
            rgb[v - 11 : v + 12, u - 11 : u + 12] = 255

    cloud = _with_image(paint)
    hypotheses = _hypotheses(cloud)
    pixel_m = 1.0 / PINHOLE.fx  # about 1 mm per pixel at the 1 m desk distance
    shapes = {
        "a": DeclaredShape(PLACES["a"], np.pi * (14 * pixel_m) ** 2, 0.88),
        "b": DeclaredShape(PLACES["b"], (23 * pixel_m) ** 2, 0.785),
    }
    choice = choose_by_shapes(cloud, hypotheses, shapes)
    assert choice.index is not None and set(choice.seen) == {"a", "b"}
    assert pose_difference(hypotheses.poses[choice.index], TRUTH)[0] < 0.002


def test_no_markers_choose_nothing():
    cloud = _with_image(lambda rgb, name, centre: None)
    hypotheses = _hypotheses(cloud)
    assert choose_by_aruco(cloud, hypotheses, {3: PLACES["a"]}).index is None
    assert choose_by_aruco(cell_scene(True), hypotheses, {3: PLACES["a"]}).index is None
