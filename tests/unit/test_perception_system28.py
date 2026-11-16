"""System 28 desk: the generated hole catalogue against the sim desk mesh, and desk localisation."""

import numpy as np
import open3d as o3d
import pytest

from perception.cloud import (
    OrganizedCloud,
    fit_table_plane,
    invert,
    pose_difference,
    raycasting_scene,
)
from perception.system28 import (
    TOP_Z_M,
    DeskLocalizationError,
    hole_catalogue,
    localize_desk,
    select_nearest_to_nominal,
)
from tests.perception_scenes import (
    PINHOLE,
    T_WORLD_CAMERA,
    T_WORLD_DESK,
    cell_scene,
    desk,
    plane_mesh,
    pose,
)


def _drop(scene, starts):
    """Height in the desk frame where a vertical ray from 0.2 m above each start hits."""
    rays = np.hstack([starts + [0, 0, 0.2], np.tile([0, 0, -1.0], (len(starts), 1))])
    t_hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    return starts[:, 2] + 0.2 - t_hit


def test_hole_catalogue_has_the_265_system28_holes():
    centres, ids, patterns = hole_catalogue()
    assert centres.shape == (265, 3) and len(set(ids)) == 265
    assert patterns.count("base_grid") == 144 and patterns.count("diagonal") == 121
    assert np.allclose(centres[:, 2], TOP_Z_M)


@pytest.mark.slow
def test_hole_catalogue_matches_the_desk_mesh():
    centres, _, patterns = hole_catalogue()
    scene = raycasting_scene([(desk().vertices_m, desk().triangles, np.eye(4))])
    # Every catalogued hole is open: the ray misses the desk or ends well below the top.
    assert np.all(_drop(scene, centres) < TOP_Z_M - 0.05)
    # Half a pitch beside a base hole (between two diagonal holes) is solid top face.
    base = centres[np.array(patterns) == "base_grid"]
    inner = base[base[:, 0] < 1.0] + [0.05, 0.0, 0.0]
    assert np.allclose(_drop(scene, inner), TOP_Z_M, atol=1e-4)


@pytest.mark.slow
def test_table_plane_normal_points_to_the_camera():
    points = cell_scene(False).points()
    normal, offset = fit_table_plane(points)
    assert offset > 0
    up_in_world = T_WORLD_CAMERA[:3, :3] @ normal
    assert up_in_world[2] > 0.999


@pytest.mark.slow
def test_localize_desk_finds_the_desk_from_its_holes():
    cloud = cell_scene(True)
    normal, offset = fit_table_plane(cloud.points())
    hypotheses = localize_desk(cloud, normal, offset)
    truth = invert(T_WORLD_CAMERA) @ T_WORLD_DESK
    assert hypotheses.matched_holes > 50
    assert min(pose_difference(p, truth)[0] for p in hypotheses.poses) < 0.002

    nominal = truth @ pose([0.02, -0.015, 0.0], [0, 0, 4])
    _, chosen = select_nearest_to_nominal(hypotheses, nominal)
    translation, rotation = pose_difference(chosen, truth)
    assert translation < 0.002 and rotation < 0.2
    assert (T_WORLD_CAMERA @ chosen)[2, 2] > 0.999  # desk z up in the world


@pytest.mark.slow
def test_localize_desk_refuses_a_plain_table():
    cloud = OrganizedCloud(
        _render_plane()  # a flat table without holes
    )
    normal, offset = fit_table_plane(cloud.points())
    with pytest.raises(DeskLocalizationError):
        localize_desk(cloud, normal, offset)


def _render_plane():
    from perception.cloud import render

    return render(raycasting_scene([plane_mesh(1.0)]), PINHOLE).xyz_m


@pytest.mark.slow
def test_above_plane_keeps_what_stands_on_the_table():
    from perception.cloud import above_plane

    cloud = cell_scene(True)
    normal, offset = fit_table_plane(cloud.points())
    above = above_plane(cloud, normal, offset)
    heights = (T_WORLD_CAMERA[:3, :3] @ cloud.xyz_m[above].T).T[:, 2] + T_WORLD_CAMERA[
        2, 3
    ]
    assert above.sum() > 1000 and heights.min() > TOP_Z_M + 0.003
