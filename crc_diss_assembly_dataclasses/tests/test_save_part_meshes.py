import os

import numpy as np
import open3d as o3d
import pytest

from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part


def make_tetrahedron():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int))
    return mesh


def make_assembly():
    part = Part(name="tetra", mesh_path="/no/path", mesh=make_tetrahedron())
    return Assembly("a", parts=[part])


def test_saved_stl_is_not_empty_and_reloads(tmp_path):
    asm = make_assembly()
    asm.save_part_meshes(str(tmp_path))
    path = tmp_path / "tetra.stl"
    assert path.stat().st_size > 0
    assert len(o3d.io.read_triangle_mesh(str(path)).triangles) == 4
    assert asm.parts["tetra"].mesh_path == str(path)


def test_save_does_not_mutate_part_mesh(tmp_path):
    asm = make_assembly()
    asm.save_part_meshes(str(tmp_path))
    assert not asm.parts["tetra"].mesh.has_triangle_normals()


def test_write_failure_raises_ioerror(tmp_path):
    asm = make_assembly()
    asm.parts["tetra"].mesh = o3d.geometry.TriangleMesh()  # empty mesh: o3d refuses to write
    with pytest.raises(IOError):
        asm.save_part_meshes(str(tmp_path))
