import numpy as np
import open3d as o3d
import fcl
import xml.etree.ElementTree as ET
import fnmatch
import os
import copy

from .xml_paths import resolve_xml_path, relativize_xml_path




class FCLTriMesh:
    """
    Small helper to build and cache an FCL BVHModel from triangle mesh data.
    You generally want to build this ONCE per part (unless geometry changes).
    """
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        V = np.asarray(vertices, dtype=np.float64)
        T = np.asarray(triangles, dtype=np.int32)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("vertices must be (N,3)")
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError("triangles must be (M,3)")

        model = fcl.BVHModel()
        model.beginModel(len(V), len(T))
        model.addSubModel(V, T)
        model.endModel()

        self.model = model

    def make_object(self, R=None, t=None) -> fcl.CollisionObject:
        if R is None:
            R = np.eye(3, dtype=np.float64)
        if t is None:
            t = np.zeros(3, dtype=np.float64)

        tf = fcl.Transform(np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))
        obj = fcl.CollisionObject(self.model, tf)
        return obj

def compute_mesh_volume_and_com(mesh: o3d.geometry.TriangleMesh) -> tuple[float, np.ndarray]:
    """
    Returns (volume, center_of_mass).
    For non-watertight meshes, Open3D volume may be invalid; we fall back to AABB center and 0 volume.
    """
    # Open3D has is_watertight(); if not, volume is not reliable.
    if hasattr(mesh, "is_watertight") and mesh.is_watertight():
        # get_volume exists on TriangleMesh; COM isn't a direct method in older versions,
        # but you can compute COM from tetrahedralization or use vertices mean as approximation.
        vol = float(mesh.get_volume())
        # For watertight meshes, a decent COM approximation is the centroid of vertices
        # unless you implement full polyhedral COM. If you need accurate COM, tell me and
        # I’ll drop in a robust polyhedral COM routine.
        com = np.asarray(mesh.vertices).mean(axis=0)
        return vol, com
    else:
        aabb = mesh.get_axis_aligned_bounding_box()
        com = np.asarray(aabb.get_center(), dtype=np.float64)
        return 0.0, com

def o3d_load_triangle_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {path}")
    # Ensure triangles exist (some formats can load as points/lines)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    # Normals are often useful for viz; cheap to compute
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh



class Part:
    def __init__(self,
                 name: str,
                 mesh_path: str,
                 density: float = 8000.0,
                 mesh: o3d.geometry.TriangleMesh | None = None,
                 material: str = "Generic"):
        self.name = name
        self.mesh_path = mesh_path
        self.density = density
        self.material = material

        self.center_of_mass: np.ndarray | None

        self._mesh: o3d.geometry.TriangleMesh | None = mesh
        self._fcl_trimesh: FCLTriMesh | None = None  # cached BVHModel wrapper

        # Physical properties
        self.degree_of_damage = 0.0

        # Initialize properties if mesh is already provided
        if self._mesh is not None:
            vol, com = compute_mesh_volume_and_com(self._mesh)
            self.center_of_mass = com
            self.weight = vol * density
        else:
            # lazy, computed once on first mesh load
            self.center_of_mass = None
            self.weight = None

    def __repr__(self):
        return f"Part({self.name})"

    def __str__(self):
        com = self.center_of_mass if self.center_of_mass is not None else "?"
        w = self.weight if self.weight is not None else float("nan")
        return f"Part '{self.name}' with center of mass {com} and weight {w:.2f} kg"

    def __eq__(self, other):
        return isinstance(other, Part) and (self.name == other.name)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        # Copy pure-Python state
        new.name = copy.deepcopy(self.name, memo)
        new.mesh_path = copy.deepcopy(self.mesh_path, memo)
        new.density = copy.deepcopy(self.density, memo)
        new.material = copy.deepcopy(self.material, memo)
        new.degree_of_damage = copy.deepcopy(self.degree_of_damage, memo)

        # Keep derived scalars if already computed (do not force recompute)
        new.center_of_mass = copy.deepcopy(self.center_of_mass, memo)
        new.weight = copy.deepcopy(self.weight, memo)

        # Skip C++ backed objects
        new._mesh = None
        new._fcl_trimesh = None

        return new

    @property
    def mesh(self) -> o3d.geometry.TriangleMesh:
        if self._mesh is None:
            self._mesh = o3d_load_triangle_mesh(self.mesh_path)
            vol, com = compute_mesh_volume_and_com(self._mesh)
            self.center_of_mass = com
            self.weight = vol * self.density
        return self._mesh

    @mesh.setter
    def mesh(self, value: o3d.geometry.TriangleMesh | None):
        self._mesh = value
        # geometry changed -> invalidate cached FCL model
        self._fcl_trimesh = None
        if value is not None:
            vol, com = compute_mesh_volume_and_com(value)
            self.center_of_mass = com
            self.weight = vol * self.density
        else:
            self.center_of_mass = None
            self.weight = None

    def delete_mesh(self):
        self._mesh = None
        self._fcl_trimesh = None
        self.center_of_mass = None
        self.weight = None

    def soft_compare(self, other):
        return isinstance(other, Part) and fnmatch.fnmatch(other.name, self.name + "*")

    def to_xml_element(self, base_dir=None):
        element = ET.Element('part')
        element.set('name', self.name)
        mesh_path = self.mesh_path if base_dir is None else relativize_xml_path(self.mesh_path, base_dir)
        element.set('mesh_path', mesh_path)
        element.set('density', str(self.density))
        element.set('material', self.material)
        return element

    @classmethod
    def from_xml_element(cls, part_element, base_dir=None):
        part_name = part_element.attrib['name']
        mesh_path = resolve_xml_path(part_element.attrib['mesh_path'], base_dir)
        density = float(part_element.attrib['density'])
        material = part_element.attrib.get('material', "Generic")
        return cls(part_name, mesh_path, density, material=material)

    # ---- FCL integration ----

    def _ensure_fcl_mesh(self):
        if self._fcl_trimesh is None:
            m = self.mesh  # triggers lazy load
            V = np.asarray(m.vertices, dtype=np.float64)
            T = np.asarray(m.triangles, dtype=np.int32)
            self._fcl_trimesh = FCLTriMesh(V, T)

    def make_fcl_object(self, R: np.ndarray | None = None, t: np.ndarray | None = None) -> fcl.CollisionObject:
        """
        Create a CollisionObject for this part at a given pose. This is needed since collision and rendering 
        now use different entities. It is used by the assembly class to build collision objects for all parts.
        """
        self._ensure_fcl_mesh()
        return self._fcl_trimesh.make_object(R=R, t=t)



def load_parts_from_folder(folder, depth, file_endings=('.stl', '.obj')):
    parts = []
    # Check if depth limit is reached
    if depth == 0:
        # Add all files in folder that end with '_vis.stl' to the list
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)) and file.endswith(file_endings):
                file_name = os.path.splitext(os.path.basename(file))[0]
                part = Part(file_name, os.path.join(folder, file))
                parts.append(part)
        return parts

    # Check if folder has no subfolders
    if len([f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]) == 0:
        # Add all files in folder that end with '_vis.stl' to the list
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)) and file.endswith(file_endings):
                file_name = os.path.splitext(os.path.basename(file))[0]
                part = Part(file_name, os.path.join(folder, file))
                parts.append(part)
        return parts

    # Recursively iterate through subfolders
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            parts += load_parts_from_folder(subfolder_path, depth - 1, file_endings)
    return parts