import numpy as np
import open3d as o3d

from diss_assembly_dataclasses.part import compute_mesh_volume_and_com, Part


def make_tetrahedron():
    # vertices of a tetrahedron: mean should be (0.25,0.25,0.25)
    V = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    T = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=int)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(T)
    mesh.compute_vertex_normals()
    return mesh


def make_single_triangle():
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    T = np.array([[0, 1, 2]], dtype=int)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(T)
    mesh.compute_vertex_normals()
    return mesh


def test_compute_mesh_com_tetrahedron():
    mesh = make_tetrahedron()
    vol, com = compute_mesh_volume_and_com(mesh)

    # expected centroid is mean of vertices
    expected = np.mean(np.asarray(mesh.vertices), axis=0)
    assert np.allclose(com, expected, atol=1e-8)
    # volume for a tetrahedron should be > 0
    assert vol > 0


def test_part_constructor_sets_center_of_mass():
    mesh = make_tetrahedron()
    p = Part(name="tetra", mesh_path="/no/path", mesh=mesh)
    expected = np.mean(np.asarray(mesh.vertices), axis=0)
    assert p.center_of_mass is not None
    assert np.allclose(p.center_of_mass, expected, atol=1e-8)


def test_non_watertight_triangle_returns_aabb_center():
    mesh = make_single_triangle()
    vol, com = compute_mesh_volume_and_com(mesh)
    assert vol == 0.0
    # AABB center should be (0.5, 0.5, 0.0)
    expected = np.array([0.5, 0.5, 0.0])
    assert np.allclose(com, expected, atol=1e-8)
