"""Simulated scenes for the perception tests: sim meshes raycast through a pinhole camera.

Neither Hao's .zdf captures nor pybullet are available in CI, so the clouds are
rendered with Open3D from the same meshes the cell simulation uses. Loaders are
cached, so test modules can wrap them in their own fixtures.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from perception.cloud import Pinhole, invert, raycasting_scene, render
from perception.registrar import MeshModel, RegistrationProfile
from perception.system28 import TOP_Z_M, desk_model

ROOT = Path(__file__).resolve().parents[1]
GRINDER = ROOT / "cases" / "angle_grinder" / "model" / "angle_grinder"
FIXTURES = ROOT / "crc_disassembly_sim" / "fixtures"

# A Zivid-like camera, downscaled: about 0.7 mm per pixel at 0.5 m.
PINHOLE = Pinhole(fx=700.0, fy=700.0, cx=320.0, cy=256.0, width=640, height=512)

# Two of Hao's seeds that find the housing; the full eight-seed profile is the default.
FAST_PROFILE = RegistrationProfile(random_seeds=(920501, 920503))


def pose(xyz, euler_deg) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def plane_mesh(z: float, half: float = 1.0):
    vertices = np.array(
        [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]]
    )
    return vertices, np.array([[0, 1, 2], [0, 2, 3]]), np.eye(4)


@lru_cache
def housing() -> MeshModel:
    return MeshModel.from_file(GRINDER / "Getriebegehaeuse_400_V1_02p.stl")


@lru_cache
def bevel_gear() -> MeshModel:
    return MeshModel.from_file(GRINDER / "Kegelrad_V1_01p.stl")


@lru_cache
def housing_scene():
    """The housing 0.5 m in front of the camera, on a plane at 0.55 m.

    Returns (cloud, mask of the housing pixels, T_camera_housing).
    """
    model = housing()
    truth = pose([0.01, -0.02, 0.5], [160, 10, 30])
    scene = raycasting_scene(
        [(model.vertices_m, model.triangles, truth), plane_mesh(0.55)]
    )
    cloud = render(scene, PINHOLE)
    mask = np.nan_to_num(cloud.xyz_m[:, :, 2], nan=1.0) < 0.545
    return cloud, mask, truth


@lru_cache
def desk() -> MeshModel:
    return desk_model()


@lru_cache
def clamp() -> MeshModel:
    return MeshModel.from_file(
        FIXTURES / "Small_system" / "body.stl", name="small_clamp"
    )


# The cell: world frame, desk turned by 20 deg, camera 1 m above the desk top, tilted.
T_WORLD_DESK = pose([0.2, -0.1, 0.0], [0, 0, 20])
T_WORLD_CAMERA = pose([0.65, 0.35, 1.3], [192, 5, 0])
T_DESK_HOUSING = pose([0.45, 0.55, TOP_Z_M + 0.045], [0, 0, 70])
T_DESK_CLAMP = pose([0.75, 0.35, TOP_Z_M + 0.0746], [0, 0, -15])


@lru_cache
def cell_scene(with_parts: bool = True):
    """The desk, and with ``with_parts`` the housing and a small clamp on it, as the
    camera sees them. Returns the organized cloud in the camera frame."""
    camera_world = invert(T_WORLD_CAMERA)
    meshes = [(desk().vertices_m, desk().triangles, camera_world @ T_WORLD_DESK)]
    if with_parts:
        for model, placement in ((housing(), T_DESK_HOUSING), (clamp(), T_DESK_CLAMP)):
            world = camera_world @ T_WORLD_DESK @ placement
            meshes.append((model.vertices_m, model.triangles, world))
    return render(raycasting_scene(meshes), PINHOLE)
