import numpy
from dataclasses import dataclass

from diss_assembly_dataclasses.part import Part, load_parts_from_folder
from diss_assembly_dataclasses.connection import Connection, Geometric_Connection, Screw_Connection, evenly_spaced_points_on_sphere, \
    Fixed_Connection, Safety_Ring_Connection

import xml.etree.ElementTree as ET
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import hashlib
import math
import itertools
import copy
import time
import warnings

import fcl
import open3d as o3d




def _warn_hold_api(name: str) -> None:
    warnings.warn(
        f"Assembly.{name} is deprecated: robot holds live in CellState "
        "(diss_assembly_dataclasses.cell_state) and the hold API is removed after the capabilities migrate.",
        DeprecationWarning,
        stacklevel=3,
    )


kit_deep_blue = "#144466"
kit_blue_green = "#00838f"
kit_yellow = "#EEB70D"
kit_red = "#B2372C"

def hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return [
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    ]

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    [kit_deep_blue, kit_blue_green, kit_yellow]
)


def o3d_visualize_contact_points(points: np.ndarray,
                                 normals: np.ndarray | None = None,
                                 meshes: list[o3d.geometry.TriangleMesh] | None = None,
                                 normal_len: float = 0.01,
                                 mesh_colors: list | None = None,
                                 show_edges: bool = True,
                                 mesh_alpha: float = 1.0,
                                 show_skybox: bool = False):
    """Visualize contact points with optional normal sticks and colored meshes.
    
    Args:
        points: Contact point coordinates (N, 3)
        normals: Normal vectors at contact points (N, 3), optional
        meshes: List of meshes to display as context
        normal_len: Length of normal indicator lines
        mesh_colors: List of RGB colors (0-1) for each mesh. If None, uses colormap.
        show_edges: If True, show wireframe edges for each mesh
        mesh_alpha: Transparency for meshes (0=transparent, 1=opaque).
        show_skybox: If True, show the environment skybox background.
    """
    geoms: list = []

    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)

    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float64)
        # Normalize normals safely (avoid div-by-zero)
        norms = np.linalg.norm(nrm, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        nrm_unit = nrm / norms

        # Map orientation [-1,1] -> color [0,1]
        colors = (nrm_unit * 0.5) + 0.5
        colors = np.clip(colors, 0.0, 1.0)

        # Set per-point colors (so points get same coloring as their normals)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    else:
        nrm_unit = None
        colors = np.tile([1.0, 0.0, 0.0], (len(pts), 1))  # Red points if no normals
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms.append({"name": "contact_points", "geometry": pcd})

    if normals is not None and len(pts) > 0:
        ends = pts + nrm_unit * float(normal_len)
        line_pts = np.vstack([pts, ends])
        lines = np.array([[i, i + len(pts)] for i in range(len(pts))], dtype=np.int32)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(line_pts.astype(np.float64))
        ls.lines = o3d.utility.Vector2iVector(lines)

        # Line colors: one color per line, same as the normal color for that point
        line_colors = colors.astype(np.float64)
        ls.colors = o3d.utility.Vector3dVector(line_colors)

        geoms.append({"name": "normals", "geometry": ls})

    if meshes:
        n_meshes = len(meshes)
        for i, mesh in enumerate(meshes):
            m = o3d.geometry.TriangleMesh(mesh)  # copy
            m.compute_vertex_normals()
            
            # Assign color from provided list or colormap
            if mesh_colors is not None and i < len(mesh_colors):
                color = mesh_colors[i]
            else:
                t = i / max(1, n_meshes - 1) if n_meshes > 1 else 0.5
                color = custom_cmap(t)[:3]
            
            m.paint_uniform_color(color)
            
            # Create material with transparency if supported
            try:
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLitTransparency"
                mat.base_color = [color[0], color[1], color[2], mesh_alpha]
                geoms.append({"name": f"mesh_{i}", "geometry": m, "material": mat})
            except AttributeError:
                # Fallback for older Open3D versions without rendering module
                geoms.append({"name": f"mesh_{i}", "geometry": m})
            
            # Add wireframe edges for better part distinction
            if show_edges:
                edges = o3d.geometry.LineSet.create_from_triangle_mesh(m)
                # Darken the mesh color for edges
                edge_color = [c * 0.3 for c in color]
                edges.paint_uniform_color(edge_color)
                geoms.append({"name": f"edges_{i}", "geometry": edges})

    o3d.visualization.draw(geoms, show_skybox=show_skybox)


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


def o3d_merge_triangle_meshes(meshes: list[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    """Concatenate triangle meshes (like trimesh.util.concatenate) by index-offsetting triangles."""
    Vs, Ts = [], []
    v_offset = 0

    for m in meshes:
        V = np.asarray(m.vertices)
        T = np.asarray(m.triangles)

        Vs.append(V)
        Ts.append(T + v_offset)
        v_offset += V.shape[0]

    V_merged = np.vstack(Vs) if Vs else np.zeros((0, 3), dtype=np.float64)
    T_merged = np.vstack(Ts) if Ts else np.zeros((0, 3), dtype=np.int32)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(V_merged.astype(np.float64))
    out.triangles = o3d.utility.Vector3iVector(T_merged.astype(np.int32))
    out.compute_vertex_normals()
    return out


def o3d_save_triangle_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    ok = o3d.io.write_triangle_mesh(path, mesh)
    if not ok:
        raise IOError(f"Open3D failed to write mesh: {path}")


def o3d_offset_mesh(mesh: o3d.geometry.TriangleMesh, offset_distance: float) -> o3d.geometry.TriangleMesh:
    """
    Offset the mesh vertices along their normals by a given distance.
    
    Args:
        mesh: Open3D TriangleMesh
        offset_distance: Distance to offset (positive = outward, negative = inward)
    
    Returns:
        A new TriangleMesh with offset vertices
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    
    # Move each vertex along its normal by the offset distance
    offset_vertices = vertices + offset_distance * normals
    
    # Create a new mesh with the offset vertices
    offset_mesh = o3d.geometry.TriangleMesh()
    offset_mesh.vertices = o3d.utility.Vector3dVector(offset_vertices)
    offset_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    offset_mesh.compute_vertex_normals()
    
    return offset_mesh


def o3d_get_triangle_centers(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute the centroid of each triangle face."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Get vertices for each triangle
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # Compute centroids
    centers = (v0 + v1 + v2) / 3.0
    return centers


def o3d_get_face_normals(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute per-face normals (normalized)."""
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return np.asarray(mesh.triangle_normals)


def o3d_nearby_faces(mesh: o3d.geometry.TriangleMesh, points: np.ndarray, radius: float = None) -> np.ndarray:
    """
    Find faces near a set of points using a KDTree on triangle centers.
    
    Args:
        mesh: Open3D TriangleMesh
        points: (N, 3) array of query points
        radius: Search radius. If None, returns the nearest face for each point.
    
    Returns:
        Array of unique face indices near the query points
    """
    centers = o3d_get_triangle_centers(mesh)
    
    # Build a KDTree from triangle centers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    face_indices = set()
    for pt in points:
        if radius is not None:
            # Search within radius
            [k, idx, _] = kdtree.search_radius_vector_3d(pt, radius)
            for i in idx:
                face_indices.add(i)
        else:
            # Find nearest face
            [k, idx, _] = kdtree.search_knn_vector_3d(pt, 1)
            if k > 0:
                face_indices.add(idx[0])
    
    return np.array(list(face_indices), dtype=np.int32)


def fcl_check_collision(mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> tuple[bool, list]:
    """
    Check collision between two Open3D meshes using FCL.
    
    Returns:
        Tuple of (collision_detected, list of contact points)
    """
    # Create FCL BVH models
    V1 = np.asarray(mesh1.vertices, dtype=np.float64)
    T1 = np.asarray(mesh1.triangles, dtype=np.int32)
    V2 = np.asarray(mesh2.vertices, dtype=np.float64)
    T2 = np.asarray(mesh2.triangles, dtype=np.int32)
    
    model1 = fcl.BVHModel()
    model1.beginModel(len(V1), len(T1))
    model1.addSubModel(V1, T1)
    model1.endModel()
    
    model2 = fcl.BVHModel()
    model2.beginModel(len(V2), len(T2))
    model2.addSubModel(V2, T2)
    model2.endModel()
    
    # Create collision objects at identity transform
    tf = fcl.Transform()
    obj1 = fcl.CollisionObject(model1, tf)
    obj2 = fcl.CollisionObject(model2, tf)
    
    # Set up collision request for contact points
    request = fcl.CollisionRequest(num_max_contacts=10000, enable_contact=True)
    result = fcl.CollisionResult()
    
    # Perform collision check
    ret = fcl.collide(obj1, obj2, request, result)
    
    contact_points = []
    if ret > 0:
        for contact in result.contacts:
            contact_points.append(contact.pos)
    
    return ret > 0, contact_points


def o3d_ray_mesh_intersection(mesh: o3d.geometry.TriangleMesh, 
                               ray_origins: np.ndarray, 
                               ray_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform ray-mesh intersection using Open3D's RaycastingScene.
    
    Args:
        mesh: Open3D TriangleMesh
        ray_origins: (N, 3) array of ray origins
        ray_directions: (N, 3) array of ray directions (will be normalized)
    
    Returns:
        hit_points: (N, 3) array of hit points (NaN if no hit)
        hit_distances: (N,) array of distances (inf if no hit)
        hit_triangle_ids: (N,) array of triangle indices (-1 if no hit)
    """
    # Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # Normalize ray directions
    ray_dirs = np.asarray(ray_directions, dtype=np.float32)
    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    ray_dirs = ray_dirs / norms
    
    ray_origins_f = np.asarray(ray_origins, dtype=np.float32)
    
    # Create rays tensor (N, 6): [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
    rays = np.concatenate([ray_origins_f, ray_dirs], axis=1)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    # Cast rays
    result = scene.cast_rays(rays_tensor)
    
    t_hit = result['t_hit'].numpy()
    triangle_ids = result['primitive_ids'].numpy().astype(np.int32)
    
    # Compute hit points
    #hit_points = ray_origins_f + ray_dirs * t_hit[:, np.newaxis]
    """
    Motivation:
    tests/test_milling/test_mo_static_apply_contract.py::test_mo_remove_target_geometric_connection_by_topology_not_identity
    tests/test_milling/test_mo_static_apply_contract.py::test_mo_mesh_assignment_is_deepcopied_no_aliasing
    tests/test_milling/test_mo_static_apply_contract.py::test_mo_property_cost_passthrough_when_feasible
      /home/hans/Uni/Bachelorarbeit/pomdp_disassembly_gym/src/diss_assembly_dataclasses/assembly.py:353: RuntimeWarning: invalid value encountered in multiply
        hit_points = ray_origins_f + ray_dirs * t_hit[:, np.newaxis]
    
    -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
    """
    t = t_hit
    valid = np.isfinite(t)  # exclude inf / nan
    
    hit_points = np.empty((t.shape[0], 3), dtype=ray_origins_f.dtype)
    hit_points[:] = np.nan
    
    hit_points[valid] = ray_origins_f[valid] + ray_dirs[valid] * t[valid, np.newaxis]
    
    # Mark misses
    miss_mask = np.isinf(t_hit)
    hit_points[miss_mask] = np.nan
    triangle_ids[miss_mask] = -1
    
    return hit_points, t_hit, triangle_ids

@dataclass
class ActiveGrasp:
    agent_name: str
    part_name: str
    grasp_id: int | str
    matrix: np.ndarray = None
    opening_width: float = 0.05
    gripper_type: str = "gripper"
    tcp_offset_mm: float = 294.0


class Assembly:
    def __init__(self, name, parts: list[Part] = None, connections: list[Connection] = None, contact_tollerance= 1.0, dot_product_threshold = -0.99):
        self.name = name
        self.xml_path = None
        
        self.parts: dict[str, Part] = {}
        self.connections: list[Connection] = []

        self.contact_tollerance = contact_tollerance
        self.dot_product_threshold = dot_product_threshold

        self.agents = {
            "robot_1": None,
            "robot_2": None,
            #"clamping_station": None
        }

        # collision cache: part name -> fcl.CollisionObject
        self._fcl_objects: dict[str, fcl.CollisionObject] = {}
        
        if parts:
            for part in parts:
                self.add_part(part)
        if connections:
            for connection in connections:
                self.add_connection(connection)

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)

        new.name = self.name
        new.xml_path = self.xml_path
        
        new.parts = dict(self.parts)
        new.connections = list(self.connections)

        new.contact_tollerance = self.contact_tollerance
        new.dot_product_threshold = self.dot_product_threshold

        new.agents = dict(self.agents)

        new.last_regrasp = dict(getattr(self, 'last_regrasp', {}))

        new._fcl_objects = {}

        return new
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        new.name = copy.deepcopy(self.name, memo)
        new.xml_path = copy.deepcopy(self.xml_path, memo)

        new.parts = copy.deepcopy(self.parts, memo)
        new.connections = copy.deepcopy(self.connections, memo)

        new.contact_tollerance = copy.deepcopy(self.contact_tollerance, memo)
        new.dot_product_threshold = copy.deepcopy(self.dot_product_threshold, memo)

        # ÄNDERUNG: Kopieren von agents
        new.agents = {}
        for agent_name, agent_obj in self.agents.items():
            if agent_obj is None:
                new.agents[agent_name] = None
            else:
                new.agents[agent_name] = copy.deepcopy(agent_obj, memo)
        new.last_regrasp = dict(getattr(self, 'last_regrasp', {}))

        new._fcl_objects = {}

        return new
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.parts) != len(other.parts) or len(self.connections) != len(other.connections):
                return False
            for part in self.parts.values():
                found = False
                for part2 in other.parts.values():
                    if part == part2:
                        found=True
                        break
                if not found:
                    return False
            for con in self.connections:
                found = False
                for con2 in other.connections:
                    if con == con2:
                        found=True
                        break
                if not found:
                    return False
            return True
            #return self.parts == other.parts and \
            #       self.connections == other.connections
        return False

    def __str__(self):
        num_parts = len(self.parts)
        num_connections = len(self.connections)
        return f"{self.name} assembly with {num_parts} parts and {num_connections} connections."

    def __hash__(self):
        return hash(self.generate_state_string())

    def add_part(self, part):
        if not isinstance(part, Part):
            raise TypeError("Only Part objects can be added to the assembly.")
        if part.name in self.parts:
            raise ValueError(f"A part with name '{part.name}' already exists in the assembly.")

        self.parts[part.name] = part

        # Build & cache FCL object now.
        self._fcl_objects[part.name] = part.make_fcl_object()

    def remove_part(self, part: Part):
        if part.name not in self.parts:
            raise ValueError(f"Part '{part.name}' does not exist in the assembly.")

        # Remove connections referencing this part
        for conn in list(self.get_connections_by_part_name(part.name)):
            self.connections.remove(conn)

        # Remove cache
        self._fcl_objects.pop(part.name, None)

        # Remove from dict
        self.parts.pop(part.name, None)

    def add_connection(self, connection):
        if not isinstance(connection, Connection):
            raise TypeError("Only Connection objects can be added to the assembly.")
        self.connections.append(connection)

    def remove_connection(self, connection):
        if connection not in self.connections:
            raise ValueError(f"Connection '{connection}' does not exist in the assembly.")
        self.connections.remove(connection)

    def get_part_by_name(self, name):
        return self.parts.get(name, None)

    def get_connections_by_part_name(self, part_name):
        return [c for c in self.connections if c.part_name1 == part_name or c.part_name2 == part_name]

    def find_contacts(self):
        self.connections = []
        n = 0
        parts_list = list(self.parts.values())
        if len(parts_list) > 2:
            num_combinations = math.factorial(len(parts_list)) // (
                        math.factorial(2) * math.factorial(len(parts_list) - 2))
        else:
            num_combinations = 1
        for pair in itertools.combinations(parts_list, 2):
            print(f"searching for contacts between {pair[0].name} and {pair[1].name} {n + 1}/{num_combinations}")
            #self.check_pair_for_contact_new(pair, contact_tollerance = 1.0, dot_product_threshold = -0.99) #regular
            #self.check_pair_for_contact_new(pair, contact_tollerance = 5.0, dot_product_threshold = -0.99) #cable
            self.check_pair_for_contact_new(pair,
                                            contact_tollerance = self.contact_tollerance,
                                            dot_product_threshold = self.dot_product_threshold)
            n += 1

    def check_pair_for_contact_new(self, pair, contact_tollerance = 1.0, dot_product_threshold = -0.99):
        """
        Check for contacts between two parts using Open3D and FCL.
        
        Args:
            pair: Tuple of two Part objects
            contact_tollerance: Distance tolerance for contact detection
            dot_product_threshold: Threshold for opposing normals (e.g., -0.99)
        """
        start_time = time.time()
        intersect_points = []
        intersect_normals = []
        intersect_distances = []

        # Get meshes from parts
        mesh0 = pair[0].mesh
        mesh1 = pair[1].mesh

        # Create offset meshes for collision detection
        offset_mesh_out = o3d_offset_mesh(mesh0, contact_tollerance)
        offset_mesh_in = o3d_offset_mesh(mesh0, -contact_tollerance)

        # Check collision between expanded mesh0 and mesh1
        collision1, contacts1 = fcl_check_collision(offset_mesh_out, mesh1)
        print(f"collision check1: {time.time()-start_time}")

        # Check collision between original mesh0 and mesh1
        collision2, contacts2 = fcl_check_collision(mesh0, mesh1)
        print(f"collision check2: {time.time()-start_time}")

        # Combine contacts
        contacts = contacts1 + contacts2

        if collision1 or collision2:
            if len(contacts) == 0:
                # FCL detected collision but returned no contact points
                # Use face centers as potential contact regions
                print("Warning: Collision detected but no contact points returned")
                return

            points = np.unique(np.array(contacts), axis=0)
            print(f"points: {time.time()-start_time}")

            # Find nearby faces for each mesh
            search_radius = contact_tollerance * 3.0
            faces0 = o3d_nearby_faces(mesh0, points, radius=search_radius)
            print(f"faces0: {time.time()-start_time}")
            faces1 = o3d_nearby_faces(mesh1, points, radius=search_radius)
            print(f"faces1: {time.time()-start_time}")

            if len(faces0) == 0 or len(faces1) == 0:
                print("No nearby faces found")
                return

            # Get mesh data
            vertices0 = np.asarray(mesh0.vertices)
            triangles0 = np.asarray(mesh0.triangles)
            vertices1 = np.asarray(mesh1.vertices)
            triangles1 = np.asarray(mesh1.triangles)

            # Get offset mesh vertices
            offset_vertices_in = np.asarray(offset_mesh_in.vertices)

            # Extract faces to keep
            faces_to_keep0 = triangles0[faces0]
            faces_to_keep1 = triangles1[faces1]

            # Find unique vertices and remap faces
            unique_vertices0, inverse_indices0 = np.unique(faces_to_keep0, return_inverse=True)
            unique_vertices1, inverse_indices1 = np.unique(faces_to_keep1, return_inverse=True)

            # Create new vertex arrays (use offset_mesh_in for mesh0)
            new_vertices0 = offset_vertices_in[unique_vertices0]
            new_vertices1 = vertices1[unique_vertices1]

            # Reshape inverse_indices to form new faces array
            new_faces0 = np.reshape(inverse_indices0, faces_to_keep0.shape).astype(np.int32)
            new_faces1 = np.reshape(inverse_indices1, faces_to_keep1.shape).astype(np.int32)

            # Create new Open3D meshes for the contact region
            new_mesh0 = o3d.geometry.TriangleMesh()
            new_mesh0.vertices = o3d.utility.Vector3dVector(new_vertices0)
            new_mesh0.triangles = o3d.utility.Vector3iVector(new_faces0)
            new_mesh0.compute_vertex_normals()
            new_mesh0.compute_triangle_normals()

            new_mesh1 = o3d.geometry.TriangleMesh()
            new_mesh1.vertices = o3d.utility.Vector3dVector(new_vertices1)
            new_mesh1.triangles = o3d.utility.Vector3iVector(new_faces1)
            new_mesh1.compute_vertex_normals()
            new_mesh1.compute_triangle_normals()

            # Get face normals and centers for contact region
            face_normals0 = o3d_get_face_normals(new_mesh0)
            face_centers0 = o3d_get_triangle_centers(new_mesh0)
            face_normals1 = o3d_get_face_normals(new_mesh1)
            face_centers1 = o3d_get_triangle_centers(new_mesh1)

            # Get full mesh face normals for dot product checks
            mesh0.compute_triangle_normals()
            mesh1.compute_triangle_normals()
            full_face_normals0 = o3d_get_face_normals(mesh0)
            full_face_normals1 = o3d_get_face_normals(mesh1)

            print(f"mesh creation: {time.time()-start_time}")

            # Ray cast from mesh0 faces to FULL mesh1 (not truncated new_mesh1!)
            # This ensures we hit even if the ray direction is not directly toward the contact region
            if len(face_centers0) > 0 and len(face_normals0) > 0:
                hit_pts0, hit_dists0, hit_tris0 = o3d_ray_mesh_intersection(
                    mesh1, face_centers0, face_normals0  # Cast against FULL mesh1
                )

                for i in range(len(face_centers0)):
                    if hit_tris0[i] >= 0:
                        distance = hit_dists0[i]
                        if distance < 2.1 * contact_tollerance:
                            # Check opposing normals using FULL mesh normals
                            if np.dot(face_normals0[i], full_face_normals1[hit_tris0[i]]) < dot_product_threshold:
                                intersect_points.append(face_centers0[i])
                                intersect_normals.append(-face_normals0[i])
                                intersect_distances.append(distance)

                # Batch vertex raycasting: collect all vertices and their face normals
                all_vertices0 = []
                all_vertex_normals0 = []
                vertex_face_map0 = []  # Maps each vertex back to its face index
                
                for i in range(len(new_faces0)):
                    face_vertex_indices = new_faces0[i]
                    face_vertices = new_vertices0[face_vertex_indices]
                    for v in face_vertices:
                        all_vertices0.append(v)
                        all_vertex_normals0.append(face_normals0[i])
                        vertex_face_map0.append(i)
                
                if len(all_vertices0) > 0:
                    all_vertices0 = np.array(all_vertices0)
                    all_vertex_normals0 = np.array(all_vertex_normals0)
                    
                    v_hit_pts, v_hit_dists, v_hit_tris = o3d_ray_mesh_intersection(
                        mesh1, all_vertices0, all_vertex_normals0  # Cast against FULL mesh1
                    )
                    
                    for j in range(len(all_vertices0)):
                        if v_hit_tris[j] >= 0:
                            v_distance = v_hit_dists[j]
                            if v_distance < 2.1 * contact_tollerance:
                                face_idx = vertex_face_map0[j]
                                # Use full mesh normals for dot product check
                                if np.dot(all_vertex_normals0[j], full_face_normals1[v_hit_tris[j]]) < dot_product_threshold:
                                    intersect_points.append(all_vertices0[j])
                                    intersect_normals.append(-all_vertex_normals0[j])
                                    intersect_distances.append(v_distance)

            print(f"tracing0: {time.time()-start_time}")

            # Ray cast from mesh1 faces to FULL mesh0
            if len(face_centers1) > 0 and len(face_normals1) > 0:
                hit_pts1, hit_dists1, hit_tris1 = o3d_ray_mesh_intersection(
                    mesh0, face_centers1, face_normals1  # Cast against FULL mesh0 (use offset_mesh_in for distance)
                )

                for i in range(len(face_centers1)):
                    if hit_tris1[i] >= 0:
                        distance = hit_dists1[i]
                        if distance < 2.1 * contact_tollerance:
                            # Use full mesh normals for dot product check
                            if np.dot(face_normals1[i], full_face_normals0[hit_tris1[i]]) < dot_product_threshold:
                                intersect_points.append(face_centers1[i])
                                intersect_normals.append(face_normals1[i])
                                intersect_distances.append(0)

                # Batch vertex raycasting for mesh1
                all_vertices1 = []
                all_vertex_normals1 = []
                vertex_face_map1 = []
                
                for i in range(len(new_faces1)):
                    face_vertex_indices = new_faces1[i]
                    face_vertices = new_vertices1[face_vertex_indices]
                    for v in face_vertices:
                        all_vertices1.append(v)
                        all_vertex_normals1.append(face_normals1[i])
                        vertex_face_map1.append(i)
                
                if len(all_vertices1) > 0:
                    all_vertices1 = np.array(all_vertices1)
                    all_vertex_normals1 = np.array(all_vertex_normals1)
                    
                    v_hit_pts, v_hit_dists, v_hit_tris = o3d_ray_mesh_intersection(
                        mesh0, all_vertices1, all_vertex_normals1  # Cast against FULL mesh0
                    )
                    
                    for j in range(len(all_vertices1)):
                        if v_hit_tris[j] >= 0:
                            v_distance = v_hit_dists[j]
                            if v_distance < 2.1 * contact_tollerance:
                                face_idx = vertex_face_map1[j]
                                # Use full mesh normals for dot product check
                                if np.dot(all_vertex_normals1[j], full_face_normals0[v_hit_tris[j]]) < dot_product_threshold:
                                    intersect_points.append(all_vertices1[j])
                                    intersect_normals.append(all_vertex_normals1[j])
                                    intersect_distances.append(v_distance)

            print(f"tracing1: {time.time()-start_time}")

        if len(intersect_points) > 0:
            self.add_connection(Geometric_Connection(pair[0].name, pair[1].name, intersect_points, intersect_normals,
                                                     intersect_distances))


    def generate_state_string(self):
        ret = ""
        for part in self.parts.values():
            ret += f" {part.name} {part.degree_of_damage}"
        for conn in self.connections:
            ret += f" {conn.part_name1} {conn.connection_type} {conn.part_name2}"
        return ret

    def generate_short_string(self):
        ret = ""
        for part in self.parts.values():
            ret += " " + part.name[0:2]
        return ret

    def generate_name_string(self):
        ret = ""
        for part in self.parts.values():
            ret += " " + part.name
        return ret

    def generate_hashID(self):
        hash = hashlib.md5(self.generate_state_string().encode())
        return int(hash.hexdigest(), 16)

    def rename_parts(self, rules_file):
        """
        Replaces whole names in an Assembly based on a list of replacement rules.

        Parameters:
            ass (list): Assembly with the parts to be renamed.
            rules_file (str): The path to the text file containing the replacement
                rules, where each line contains two strings separated by a space.
                The first string is the string to be replaced, and the second string
                is the replacement string.

        """
        with open(rules_file, 'r') as f:
            rules = [line.strip().split() for line in f]
        for part in self.parts.values():
            for rule in rules:
                if part.name == rule[0]:
                    part.name = rule[1]
        for connection in self.connections:
            for rule in rules:
                if connection.part_name1 == rule[0]:
                    connection.part_name1 = rule[1]
                if connection.part_name2 == rule[0]:
                    connection.part_name2 = rule[1]

    def rename_single_part(self, part: Part, new_name: str):
        old_name = part.name
        if old_name not in self.parts:
            raise KeyError(old_name)
        if new_name in self.parts:
            raise ValueError(new_name)
    
        # update connections
        for c in self.connections:
            if c.part_name1 == old_name:
                c.part_name1 = new_name
            if c.part_name2 == old_name:
                c.part_name2 = new_name
    
        # move dict entry
        self.parts[new_name] = self.parts.pop(old_name)
        part.name = new_name
    
        # move fcl cache entry if exists
        if hasattr(self, "_fcl_objects") and self._fcl_objects is not None:
            if old_name in self._fcl_objects:
                self._fcl_objects[new_name] = self._fcl_objects.pop(old_name)

    #def rename_single_part(self, part: Part, new_name: str):
    #    for connection in self.connections:
    #        if connection.part_name1 == part.name:
    #            connection.part_name1 = new_name
    #        if connection.part_name2 == part.name:
    #            connection.part_name2 = new_name
    #    part.name = new_name

    def get_part_list(self):
        list = []
        for part in self.parts.values():
            list.append(part.name)
        return list

    def generate_graph(self):
        G = nx.Graph()
        labels = {}
        G.add_nodes_from([part.name for part in self.parts.values()])
        for part in self.parts.values():
            labels[part.name] = part.name.replace("_", "\n")  # [:2]
        for conn in self.connections:
            G.add_edge(conn.part_name1, conn.part_name2, connection_type=conn.connection_type)
        return G, labels

    def show(self, highlight_red: list[str] | None = None, use_names: bool = False, show_edges: bool = False):
        """Visualize the assembly with colored parts.
        
        Args:
            highlight_red: List of part names to highlight in red
            use_names: If True, use the new draw() API with named geometries
            show_edges: If True, show wireframe edges for better part distinction
        """
        highlight_red = set(highlight_red or [])
        geoms = []

        parts = list(self.parts.values())
        n = max(1, len(parts))
        for i, part in enumerate(parts):
            m = o3d.geometry.TriangleMesh(part.mesh)  # copy
            m.compute_vertex_normals()

            if part.name in highlight_red:
                color = hex_to_rgb01(kit_red)
            else:
                t = i / max(1, n - 1) if n > 1 else 0.5
                color = custom_cmap(t)[:3]

            m.paint_uniform_color(color)

            if use_names:
                geoms.append({"name": part.name, "geometry": m})
            else:
                geoms.append(m)
            
            # Add wireframe edges for better part distinction
            if show_edges:
                edges = o3d.geometry.LineSet.create_from_triangle_mesh(m)
                # Darken the mesh color for edges
                edge_color = [c * 0.3 for c in color]
                edges.paint_uniform_color(edge_color)
                if use_names:
                    geoms.append({"name": f"{part.name}_edges", "geometry": edges})
                else:
                    geoms.append(edges)

        if use_names:
            o3d.visualization.draw(geoms, show_skybox=False)
        else:
            o3d.visualization.draw_geometries(geoms)

    def show_contact_points(self, connection: Geometric_Connection, normal_len: float = 0.01, show_edges: bool = False, mesh_alpha: float = 0.5):
        """Visualize contact points for a specific connection.
        
        Args:
            connection: The geometric connection to visualize
            normal_len: Length of normal indicator lines
            show_edges: If True, show wireframe edges for meshes
            mesh_alpha: Transparency for meshes (0=transparent, 1=opaque)
        """
        p1 = self.get_part_by_name(connection.part_name1)
        p2 = self.get_part_by_name(connection.part_name2)
        meshes = []
        mesh_colors = []
        
        if p1 is not None:
            meshes.append(p1.mesh)
            mesh_colors.append(custom_cmap(0.0)[:3])  # First color from colormap
        if p2 is not None:
            meshes.append(p2.mesh)
            mesh_colors.append(custom_cmap(1.0)[:3])  # Last color from colormap

        o3d_visualize_contact_points(
            points=np.asarray(connection.points),
            normals=np.asarray(connection.normals) if connection.normals is not None else None,
            meshes=meshes,
            normal_len=normal_len,
            mesh_colors=mesh_colors,
            show_edges=show_edges,
            mesh_alpha=mesh_alpha,
        )

    def show_all_contact_points(self, normal_len: float = 0.01, show_edges: bool = True, mesh_alpha: float = 0.3):
        """Visualize all contact points in the assembly.
        
        Args:
            normal_len: Length of normal indicator lines
            show_edges: If True, show wireframe edges for meshes
            mesh_alpha: Transparency for meshes (0=transparent, 1=opaque)
        """
        all_pts = []
        all_nrm = []

        for conn in self.connections:
            if isinstance(conn, Geometric_Connection) and conn.points is not None and len(conn.points) > 0:
                all_pts.append(np.asarray(conn.points))
                if conn.normals is not None and len(conn.normals) == len(conn.points):
                    all_nrm.append(np.asarray(conn.normals))
                else:
                    all_nrm.append(np.zeros_like(np.asarray(conn.points)))

        if len(all_pts) == 0:
            print("No contact points to display.")
            self.show()
            return

        pts = np.vstack(all_pts)
        nrm = np.vstack(all_nrm)
        
        # Get meshes and generate consistent colors
        parts = list(self.parts.values())
        meshes = [p.mesh for p in parts]
        n = max(1, len(parts))
        mesh_colors = [custom_cmap(i / max(1, n - 1) if n > 1 else 0.5)[:3] for i in range(n)]
        
        o3d_visualize_contact_points(pts, nrm, meshes=meshes, normal_len=normal_len, 
                                     mesh_colors=mesh_colors, show_edges=show_edges,
                                     mesh_alpha=mesh_alpha)

    def debug_contact_detection(self, part_name1: str, part_name2: str, 
                                 contact_tolerance: float = None,
                                 dot_product_threshold: float = None):
        """Visualize contact detection parameters between two parts.
        
        This function helps tune contact detection parameters by showing:
        - Green: Accepted contact points
        - Red: Rejected by dot product (normals not opposing enough)
        - Blue: Rejected by distance (too far apart)
        - Yellow: No ray hit (ray from face didn't hit other mesh)
        
        Args:
            part_name1: Name of the first part
            part_name2: Name of the second part
            contact_tolerance: Override the assembly's contact tolerance
            dot_product_threshold: Override the assembly's dot product threshold
        """
        p1 = self.get_part_by_name(part_name1)
        p2 = self.get_part_by_name(part_name2)
        
        if p1 is None or p2 is None:
            print(f"Could not find parts: {part_name1}, {part_name2}")
            return
        
        contact_tol = contact_tolerance if contact_tolerance is not None else self.contact_tollerance
        dot_thresh = dot_product_threshold if dot_product_threshold is not None else self.dot_product_threshold
        
        print(f"\n{'='*60}")
        print(f"Contact Detection Parameters Test")
        print(f"Parts: {part_name1} <-> {part_name2}")
        print(f"Contact tolerance: {contact_tol}")
        print(f"Dot product threshold: {dot_thresh}")
        print(f"{'='*60}\n")
        
        mesh0 = p1.mesh
        mesh1 = p2.mesh
        
        # Create offset meshes
        offset_mesh_out = o3d_offset_mesh(mesh0, contact_tol)
        offset_mesh_in = o3d_offset_mesh(mesh0, -contact_tol)
        
        # Check collisions
        collision1, contacts1 = fcl_check_collision(offset_mesh_out, mesh1)
        collision2, contacts2 = fcl_check_collision(mesh0, mesh1)
        
        print(f"FCL collision (expanded mesh): {collision1}, {len(contacts1)} contact points")
        print(f"FCL collision (original mesh): {collision2}, {len(contacts2)} contact points")
        
        contacts = contacts1 + contacts2
        
        if not (collision1 or collision2):
            print("No collision detected!")
            return
        
        if len(contacts) == 0:
            print("Collision detected but no contact points returned by FCL")
            return
        
        points = np.unique(np.array(contacts), axis=0)
        print(f"Unique FCL contact points: {len(points)}")
        
        # Find nearby faces
        search_radius = contact_tol * 3.0
        faces0 = o3d_nearby_faces(mesh0, points, radius=search_radius)
        faces1 = o3d_nearby_faces(mesh1, points, radius=search_radius)
        
        print(f"Nearby faces on mesh0: {len(faces0)}")
        print(f"Nearby faces on mesh1: {len(faces1)}")
        
        if len(faces0) == 0 or len(faces1) == 0:
            print("No nearby faces found!")
            return
        
        # Get mesh data
        vertices0 = np.asarray(mesh0.vertices)
        triangles0 = np.asarray(mesh0.triangles)
        vertices1 = np.asarray(mesh1.vertices)
        triangles1 = np.asarray(mesh1.triangles)
        offset_vertices_in = np.asarray(offset_mesh_in.vertices)
        
        # Extract contact region faces
        faces_to_keep0 = triangles0[faces0]
        faces_to_keep1 = triangles1[faces1]
        
        unique_vertices0, inverse_indices0 = np.unique(faces_to_keep0, return_inverse=True)
        unique_vertices1, inverse_indices1 = np.unique(faces_to_keep1, return_inverse=True)
        
        new_vertices0 = offset_vertices_in[unique_vertices0]
        new_vertices1 = vertices1[unique_vertices1]
        
        new_faces0 = np.reshape(inverse_indices0, faces_to_keep0.shape).astype(np.int32)
        new_faces1 = np.reshape(inverse_indices1, faces_to_keep1.shape).astype(np.int32)
        
        # Create contact region meshes
        new_mesh0 = o3d.geometry.TriangleMesh()
        new_mesh0.vertices = o3d.utility.Vector3dVector(new_vertices0)
        new_mesh0.triangles = o3d.utility.Vector3iVector(new_faces0)
        new_mesh0.compute_vertex_normals()
        new_mesh0.compute_triangle_normals()
        
        new_mesh1 = o3d.geometry.TriangleMesh()
        new_mesh1.vertices = o3d.utility.Vector3dVector(new_vertices1)
        new_mesh1.triangles = o3d.utility.Vector3iVector(new_faces1)
        new_mesh1.compute_vertex_normals()
        new_mesh1.compute_triangle_normals()
        
        face_normals0 = o3d_get_face_normals(new_mesh0)
        face_centers0 = o3d_get_triangle_centers(new_mesh0)
        face_normals1 = o3d_get_face_normals(new_mesh1)
        face_centers1 = o3d_get_triangle_centers(new_mesh1)
        
        # Get full mesh face normals for dot product checks
        mesh0.compute_triangle_normals()
        mesh1.compute_triangle_normals()
        full_face_normals0 = o3d_get_face_normals(mesh0)
        full_face_normals1 = o3d_get_face_normals(mesh1)
        
        print(f"\nContact region mesh0: {len(face_centers0)} faces")
        print(f"Contact region mesh1: {len(face_centers1)} faces")
        
        # Analyze ray casting
        accepted_points = []
        accepted_normals = []
        rejected_distance = []
        rejected_dot_product = []
        rejected_no_hit = []
        
        # Cast rays from mesh0 to mesh1
        if len(face_centers0) > 0:
            hit_pts0, hit_dists0, hit_tris0 = o3d_ray_mesh_intersection(
                mesh1, face_centers0, face_normals0
            )
            
            for i in range(len(face_centers0)):
                if hit_tris0[i] >= 0:
                    distance = hit_dists0[i]
                    dot_val = np.dot(face_normals0[i], full_face_normals1[hit_tris0[i]])
                    
                    if distance >= 2.1 * contact_tol:
                        rejected_distance.append((face_centers0[i], face_normals0[i], distance))
                    elif dot_val >= dot_thresh:
                        rejected_dot_product.append((face_centers0[i], face_normals0[i], dot_val))
                    else:
                        accepted_points.append(face_centers0[i])
                        accepted_normals.append(-face_normals0[i])
                else:
                    rejected_no_hit.append((face_centers0[i], face_normals0[i]))
        
        # Cast rays from mesh1 to mesh0
        if len(face_centers1) > 0:
            hit_pts1, hit_dists1, hit_tris1 = o3d_ray_mesh_intersection(
                mesh0, face_centers1, face_normals1
            )
            
            for i in range(len(face_centers1)):
                if hit_tris1[i] >= 0:
                    distance = hit_dists1[i]
                    dot_val = np.dot(face_normals1[i], full_face_normals0[hit_tris1[i]])
                    
                    if distance >= 2.1 * contact_tol:
                        rejected_distance.append((face_centers1[i], face_normals1[i], distance))
                    elif dot_val >= dot_thresh:
                        rejected_dot_product.append((face_centers1[i], face_normals1[i], dot_val))
                    else:
                        accepted_points.append(face_centers1[i])
                        accepted_normals.append(face_normals1[i])
                else:
                    rejected_no_hit.append((face_centers1[i], face_normals1[i]))
        
        # Print summary
        print(f"\n--- Results ---")
        print(f"Accepted contact points: {len(accepted_points)}")
        print(f"Rejected (distance >= {2.1 * contact_tol:.4f}): {len(rejected_distance)}")
        print(f"Rejected (dot product >= {dot_thresh}): {len(rejected_dot_product)}")
        print(f"Rejected (no ray hit): {len(rejected_no_hit)}")
        
        if len(rejected_dot_product) > 0:
            dot_vals = [r[2] for r in rejected_dot_product]
            print(f"\n  Dot product values of rejected:")
            print(f"    Min: {min(dot_vals):.4f}, Max: {max(dot_vals):.4f}, Mean: {np.mean(dot_vals):.4f}")
            print(f"    Suggestion: Increase dot_product_threshold to {max(dot_vals) + 0.01:.2f} to accept these")
        
        if len(rejected_distance) > 0:
            dist_vals = [r[2] for r in rejected_distance]
            print(f"\n  Distance values of rejected:")
            print(f"    Min: {min(dist_vals):.4f}, Max: {max(dist_vals):.4f}")
            print(f"    Suggestion: Increase contact_tolerance to {max(dist_vals) / 2.1 + 0.1:.2f} to accept these")
        
        # Visualize
        print(f"\n--- Visualization ---")
        print(f"  Green: Accepted ({len(accepted_points)})")
        print(f"  Red: Rejected by dot product ({len(rejected_dot_product)})")
        print(f"  Blue: Rejected by distance ({len(rejected_distance)})")
        print(f"  Yellow: No ray hit ({len(rejected_no_hit)})")
        
        geoms = []
        
        # Add meshes (transparent)
        m0 = o3d.geometry.TriangleMesh(mesh0)
        m0.compute_vertex_normals()
        m0.paint_uniform_color(custom_cmap(0.0)[:3])
        m1 = o3d.geometry.TriangleMesh(mesh1)
        m1.compute_vertex_normals()
        m1.paint_uniform_color(custom_cmap(1.0)[:3])
        
        try:
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [custom_cmap(0.0)[0], custom_cmap(0.0)[1], custom_cmap(0.0)[2], 0.3]
            geoms.append({"name": part_name1, "geometry": m0, "material": mat})
            
            mat2 = o3d.visualization.rendering.MaterialRecord()
            mat2.shader = "defaultLitTransparency"
            mat2.base_color = [custom_cmap(1.0)[0], custom_cmap(1.0)[1], custom_cmap(1.0)[2], 0.3]
            geoms.append({"name": part_name2, "geometry": m1, "material": mat2})
        except AttributeError:
            geoms.append({"name": part_name1, "geometry": m0})
            geoms.append({"name": part_name2, "geometry": m1})
        
        # Add accepted points (green)
        if len(accepted_points) > 0:
            pcd_accept = o3d.geometry.PointCloud()
            pcd_accept.points = o3d.utility.Vector3dVector(np.array(accepted_points))
            pcd_accept.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append({"name": "accepted", "geometry": pcd_accept})
        
        # Add rejected by dot product (red)
        if len(rejected_dot_product) > 0:
            pcd_dot = o3d.geometry.PointCloud()
            pcd_dot.points = o3d.utility.Vector3dVector(np.array([r[0] for r in rejected_dot_product]))
            pcd_dot.paint_uniform_color([1.0, 0.0, 0.0])
            geoms.append({"name": "rejected_dot", "geometry": pcd_dot})
        
        # Add rejected by distance (blue)
        if len(rejected_distance) > 0:
            pcd_dist = o3d.geometry.PointCloud()
            pcd_dist.points = o3d.utility.Vector3dVector(np.array([r[0] for r in rejected_distance]))
            pcd_dist.paint_uniform_color([0.0, 0.0, 1.0])
            geoms.append({"name": "rejected_dist", "geometry": pcd_dist})
        
        # Add no hit (yellow)
        if len(rejected_no_hit) > 0:
            pcd_nohit = o3d.geometry.PointCloud()
            pcd_nohit.points = o3d.utility.Vector3dVector(np.array([r[0] for r in rejected_no_hit]))
            pcd_nohit.paint_uniform_color([1.0, 1.0, 0.0])
            geoms.append({"name": "no_hit", "geometry": pcd_nohit})
        
        o3d.visualization.draw(geoms, show_skybox=False)

    def show_connection_dof(self, connection: Connection, 
                            move_part: str | None = None,
                            show_edges: bool = True):
        """Visualize a connection and allow interactive mouse-based movement along DOF directions.
        
        Shows the two connected parts. The user can drag the moving part with the mouse,
        and movement is automatically constrained to the allowed DOF directions.
        
        Args:
            connection: The connection to visualize
            move_part: Name of the part to move (default: part_name2). The other part stays fixed.
            show_edges: If True, show wireframe edges for meshes
            
        Mouse controls:
            - Click and drag on the moving part to translate it along allowed DOF
            - Mouse movement is projected onto the DOF subspace
            - Scroll wheel: Zoom
            - Right-click drag: Rotate view
            - Middle-click drag: Pan view
            
        Keyboard controls:
            - R: Reset to original position
            - +/-: Increase/decrease movement sensitivity
        """
        p1 = self.get_part_by_name(connection.part_name1)
        p2 = self.get_part_by_name(connection.part_name2)
        
        if p1 is None or p2 is None:
            print(f"Cannot find parts for connection: {connection}")
            return
        
        # Determine which part moves
        if move_part is None:
            move_part = connection.part_name2
        
        if move_part == connection.part_name1:
            fixed_part, moving_part = p2, p1
            # DOF directions are relative to part1->part2, so negate for part1 movement
            dof_directions = [-np.array(d) for d in connection.DOF] if connection.DOF else []
        else:
            fixed_part, moving_part = p1, p2
            dof_directions = [np.array(d) for d in connection.DOF] if connection.DOF else []
        
        if len(dof_directions) == 0:
            print(f"Connection has no degrees of freedom (fixed connection).")
            print("Showing parts without movement capability.")
        
        # Create visualization
        self._show_dof_interactive_mouse(
            fixed_part=fixed_part,
            moving_part=moving_part,
            dof_directions=dof_directions,
            show_edges=show_edges,
            connection=connection,
        )
    
    def _show_dof_interactive_mouse(self, fixed_part: Part, moving_part: Part,
                                     dof_directions: list[np.ndarray],
                                     show_edges: bool,
                                     connection: Connection):
        """Internal method for interactive DOF visualization with mouse dragging."""
        
        # Build DOF basis matrix for projection (each column is a DOF direction)
        if len(dof_directions) > 0:
            dof_matrix = np.column_stack([d / np.linalg.norm(d) for d in dof_directions])
        else:
            dof_matrix = None
        
        # State
        current_offset = np.array([0.0, 0.0, 0.0])
        sensitivity = [1.0]  # Use list to allow modification in nested function
        is_dragging = [False]
        last_mouse_pos = [None]
        
        # Calculate scale based on bounding box
        combined_mesh = o3d_merge_triangle_meshes([fixed_part.mesh, moving_part.mesh])
        bbox = combined_mesh.get_axis_aligned_bounding_box()
        bbox_extent = np.linalg.norm(bbox.get_extent())
        base_speed = bbox_extent * 0.002
        
        # Colors
        fixed_color = custom_cmap(0.0)[:3]
        moving_color = custom_cmap(1.0)[:3]
        
        # Create geometries
        fixed_mesh = o3d.geometry.TriangleMesh(fixed_part.mesh)
        fixed_mesh.compute_vertex_normals()
        fixed_mesh.paint_uniform_color(fixed_color)
        
        moving_mesh = o3d.geometry.TriangleMesh(moving_part.mesh)
        moving_mesh.compute_vertex_normals()
        moving_mesh.paint_uniform_color(moving_color)
        
        # Store original vertices for reset
        original_vertices = np.asarray(moving_part.mesh.vertices).copy()
        
        geoms = []
        geoms.append({"name": fixed_part.name, "geometry": fixed_mesh})
        geoms.append({"name": moving_part.name, "geometry": moving_mesh})
        
        if show_edges:
            fixed_edges = o3d.geometry.LineSet.create_from_triangle_mesh(fixed_mesh)
            fixed_edges.paint_uniform_color([c * 0.3 for c in fixed_color])
            geoms.append({"name": f"{fixed_part.name}_edges", "geometry": fixed_edges})
            
            moving_edges = o3d.geometry.LineSet.create_from_triangle_mesh(moving_mesh)
            moving_edges.paint_uniform_color([c * 0.3 for c in moving_color])
            geoms.append({"name": f"{moving_part.name}_edges", "geometry": moving_edges})
        else:
            moving_edges = None
        
        # Add contact points if geometric connection
        if isinstance(connection, Geometric_Connection) and connection.points is not None and len(connection.points) > 0:
            contact_pcd = o3d.geometry.PointCloud()
            contact_pcd.points = o3d.utility.Vector3dVector(np.asarray(connection.points))
            contact_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
            geoms.append({"name": "contact_points", "geometry": contact_pcd})
        
        def project_to_dof_subspace(delta_3d):
            """Project a 3D movement vector onto the DOF subspace."""
            if dof_matrix is None or dof_matrix.shape[1] == 0:
                return np.zeros(3)
            # Project: result = D @ (D^T @ delta) where D is orthonormal DOF basis
            # This projects delta onto the column space of dof_matrix
            coeffs = dof_matrix.T @ delta_3d
            return dof_matrix @ coeffs
        
        def update_mesh_position():
            """Update the moving mesh vertices."""
            moving_mesh.vertices = o3d.utility.Vector3dVector(original_vertices + current_offset)
            moving_mesh.compute_vertex_normals()
        
        # Print instructions
        print("\n=== Interactive DOF Movement ===")
        print(f"Moving part: {moving_part.name} (yellow)")
        print(f"Fixed part: {fixed_part.name} (blue)")
        print(f"Number of DOF: {len(dof_directions)}")
        print("\nControls:")
        print("  Left-click + drag: Move part along DOF directions")
        print("  Ctrl + Left-click + drag: Move along screen X/Y")
        print("  Shift + Left-click + drag: Move along screen Z (depth)")
        print("  R: Reset to original position")
        print("  +/-: Increase/decrease movement sensitivity")
        print("  Right-click drag: Rotate view")
        print("  Scroll: Zoom")
        print("  Close window to exit")
        print("================================\n")
        
        # Use the simpler keyboard-based approach with VisualizerWithKeyCallback
        # since Open3D's mouse callbacks are limited
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"DOF: {fixed_part.name} <-> {moving_part.name}")
        
        for g in geoms:
            vis.add_geometry(g["geometry"])
        
        def move_along_dof(dof_idx, positive=True):
            """Move along a specific DOF direction."""
            nonlocal current_offset
            if dof_idx < len(dof_directions):
                direction = dof_directions[dof_idx]
                direction = direction / np.linalg.norm(direction)
                delta = direction * base_speed * sensitivity[0] * (1 if positive else -1)
                current_offset = current_offset + delta
                update_mesh_position()
                vis.update_geometry(moving_mesh)
                if moving_edges is not None:
                    new_edges = o3d.geometry.LineSet.create_from_triangle_mesh(moving_mesh)
                    moving_edges.points = new_edges.points
                    moving_edges.lines = new_edges.lines
                    vis.update_geometry(moving_edges)
                vis.poll_events()
                vis.update_renderer()
        
        def reset_callback(vis_cb):
            nonlocal current_offset
            current_offset = np.array([0.0, 0.0, 0.0])
            update_mesh_position()
            vis.update_geometry(moving_mesh)
            if moving_edges is not None:
                new_edges = o3d.geometry.LineSet.create_from_triangle_mesh(moving_mesh)
                moving_edges.points = new_edges.points
                moving_edges.lines = new_edges.lines
                vis.update_geometry(moving_edges)
            print(f"Reset to original position")
            return False
        
        def increase_sensitivity(vis_cb):
            sensitivity[0] *= 1.5
            print(f"Sensitivity: {sensitivity[0]:.2f}x")
            return False
        
        def decrease_sensitivity(vis_cb):
            sensitivity[0] /= 1.5
            print(f"Sensitivity: {sensitivity[0]:.2f}x")
            return False
        
        # Movement callbacks
        def make_dof_callback(dof_idx, positive):
            def callback(vis_cb):
                move_along_dof(dof_idx, positive)
                return False
            return callback
        
        # Register key callbacks for movement
        # Use number keys 1-9 for different DOF directions (+ and -)
        # Or use intuitive keys based on number of DOFs
        
        if len(dof_directions) >= 1:
            # DOF 0: W/S
            vis.register_key_callback(ord('W'), make_dof_callback(0, True))
            vis.register_key_callback(ord('S'), make_dof_callback(0, False))
        
        if len(dof_directions) >= 2:
            # DOF 1: A/D
            vis.register_key_callback(ord('A'), make_dof_callback(1, False))
            vis.register_key_callback(ord('D'), make_dof_callback(1, True))
        
        if len(dof_directions) >= 3:
            # DOF 2: Q/E
            vis.register_key_callback(ord('Q'), make_dof_callback(2, False))
            vis.register_key_callback(ord('E'), make_dof_callback(2, True))
        
        if len(dof_directions) >= 4:
            # DOF 3: Z/X
            vis.register_key_callback(ord('Z'), make_dof_callback(3, False))
            vis.register_key_callback(ord('X'), make_dof_callback(3, True))
        
        if len(dof_directions) >= 5:
            # DOF 4: C/V
            vis.register_key_callback(ord('C'), make_dof_callback(4, False))
            vis.register_key_callback(ord('V'), make_dof_callback(4, True))
        
        # Reset and sensitivity
        vis.register_key_callback(ord('R'), reset_callback)
        vis.register_key_callback(ord('='), increase_sensitivity)
        vis.register_key_callback(ord('-'), decrease_sensitivity)
        
        # Also print DOF-specific controls
        if len(dof_directions) > 0:
            print("DOF movement keys:")
            dof_keys = [("W/S", 0), ("A/D", 1), ("Q/E", 2), ("Z/X", 3), ("C/V", 4)]
            for keys, idx in dof_keys:
                if idx < len(dof_directions):
                    dir_str = np.array2string(dof_directions[idx], precision=2, suppress_small=True)
                    print(f"  {keys}: DOF {idx} direction {dir_str}")
        
        vis.run()
        vis.destroy_window()

    def merge_parts(self, part_names: list[str], merged_part_name: str):
        parts_to_merge: list[Part] = []
        for pn in part_names:
            p = self.get_part_by_name(pn)
            if p is None:
                raise ValueError(f"Part '{pn}' not found in assembly")
            parts_to_merge.append(p)

        merged_mesh = o3d_merge_triangle_meshes([p.mesh for p in parts_to_merge])

        _, ext = os.path.splitext(parts_to_merge[0].mesh_path)
        base_path = os.path.dirname(parts_to_merge[0].mesh_path)
        merged_mesh_path = os.path.join(base_path, f"{merged_part_name}{ext}")

        o3d_save_triangle_mesh(merged_mesh, merged_mesh_path)

        merged_part = Part(merged_part_name, mesh_path=merged_mesh_path, mesh=merged_mesh)

        # Update connections to reference merged part
        for connection in self.connections:
            if connection.part_name1 in part_names and connection.part_name2 not in part_names:
                connection.part_name1 = merged_part_name
            elif connection.part_name2 in part_names and connection.part_name1 not in part_names:
                connection.part_name2 = merged_part_name

        for p in parts_to_merge:
            self.remove_part(p)

        self.add_part(merged_part)

    def get_part_dof(self, part: Part, ignored_connections=None):
        connections = self.get_connections_by_part_name(part.name)
        if ignored_connections is not None:
            connections = [conn for conn in connections if conn not in ignored_connections]
        DOF_part = evenly_spaced_points_on_sphere()
        for conn in connections:
            ctype = getattr(conn, "connection_type", None)
            if ctype in ("screw",):
                continue
    
            if part.name == conn.part_name1:
                restrictions_temp = [r * -1 for r in conn.restrictions]
            else:
                restrictions_temp = conn.restrictions
    
            for rest in restrictions_temp:
                i = 0
                while i < len(DOF_part):
                    if np.array_equal(DOF_part[i], rest):
                        del DOF_part[i]
                    else:
                        i += 1
        return DOF_part
    
        #for conn in connections:
        #    if part.name == conn.part_name1:
        #        restrictions_temp = []
        #        DOF_temp = []
        #        for i in range(len(conn.DOF)):
        #            DOF_temp.append(conn.DOF[i] * -1)
        #        for i in range(len(conn.restrictions)):
        #            restrictions_temp.append(conn.restrictions[i] * -1)
        #    else:
        #        restrictions_temp = conn.restrictions
        #        DOF_temp = conn.DOF
        #    """if conn.connection_type == "geometric":
        #        mean_normals = np.mean(conn.normals, axis=0)
        #        # Calculate the length of the mean vector
        #        length_mean_vector = np.linalg.norm(mean_normals)
        #        if length_mean_vector >= 0.9 and len(conn.DOF) > 0:
        #            distances = np.linalg.norm(conn.DOF + mean_normals, axis=1)
        #            min_index = np.argmin(distances)
        #            restrictions_temp = restrictions_temp + DOF_temp[:min_index]
        #            DOF_temp = [DOF_temp[min_index]]"""
#
        #    for rest in restrictions_temp:
        #        i = 0
        #        while i < len(DOF_part):
        #            if np.array_equal(DOF_part[i], rest):
        #                del DOF_part[i]
        #            else:
        #                i += 1


#    def get_part_dof(self, part: Part, ignored_connections=None, eps: float = 1e-3):
#        """
#        Compute feasible DOF directions for a part by removing restricted directions.
#    
#        Patch:
#        - Replace exact-match deletion (np.array_equal) with nearest-neighbor deletion:
#            d* = argmin ||DOF_part - rest||
#            if min_dist < eps: remove d*
#        - This is robust to sampling sets that do NOT contain antipodal pairs or exact rounded matches.
#    
#        Parameters
#        ----------
#        part : Part
#            Target part whose DOF is requested.
#        ignored_connections : optional
#            List of connections to ignore (e.g., during what-if checks).
#        eps : float
#            Distance threshold to consider a direction "matching" a restriction.
#            For unit vectors, typical values: 1e-3 ~ 1e-2.
#        """
#        connections = self.get_connections_by_part_name(part.name)
#        if ignored_connections is not None:
#            connections = [conn for conn in connections if conn not in ignored_connections]
#    
#        # Start with full discretized sphere
#        DOF_part = evenly_spaced_points_on_sphere()
#    
#        if not DOF_part:
#            return []
#    
#        # Work in numpy for vectorized distance computations
#        DOF_arr = np.asarray(DOF_part, dtype=float).reshape((-1, 3))
#    
#        # Normalize DOF directions to be safe (rounding should still keep near-unit length)
#        DOF_norm = np.linalg.norm(DOF_arr, axis=1)
#        valid = DOF_norm > 1e-12
#        if not np.all(valid):
#            DOF_arr = DOF_arr[valid]
#            DOF_norm = DOF_norm[valid]
#        DOF_arr = DOF_arr / DOF_norm[:, None]
#    
#        # Boolean mask: True => still feasible (not deleted)
#        alive = np.ones((DOF_arr.shape[0],), dtype=bool)
#    
#        def _normalize_dir(v: np.ndarray) -> np.ndarray | None:
#            v = np.asarray(v, dtype=float).reshape((3,))
#            n = np.linalg.norm(v)
#            if n <= 1e-12:
#                return None
#            return v / n
#    
#        for conn in connections:
#            # Build restriction list in the part's local "sign convention"
#            if part.name == conn.part_name1:
#                # Flip directions for part_name1 side
#                restrictions_temp = [np.asarray(r, dtype=float) * -1.0 for r in (conn.restrictions or [])]
#            else:
#                restrictions_temp = [np.asarray(r, dtype=float) for r in (conn.restrictions or [])]
#    
#            # Remove each restricted direction by nearest-neighbor match
#            for rest in restrictions_temp:
#                rest_n = _normalize_dir(rest)
#                if rest_n is None:
#                    continue
#    
#                idx_alive = np.flatnonzero(alive)
#                if idx_alive.size == 0:
#                    # Fully blocked
#                    return []
#    
#                candidates = DOF_arr[idx_alive]  # (K,3)
#                # Euclidean distance on unit sphere chord length
#                dists = np.linalg.norm(candidates - rest_n[None, :], axis=1)
#                j = int(np.argmin(dists))
#                min_dist = float(dists[j])
#    
#                if min_dist < eps:
#                    # Remove that nearest direction
#                    alive[idx_alive[j]] = False
#    
#        # Return remaining feasible DOF directions as list of numpy arrays (same style as before)
#        remaining = DOF_arr[alive]
#        return [remaining[i, :] for i in range(remaining.shape[0])]

    def _ensure_fcl_cache(self):
        # Ensure dict exists
        if not hasattr(self, "_fcl_objects") or self._fcl_objects is None:
            self._fcl_objects = {}

        # Rebuild if missing entries (deepcopy/copy clears it)
        if set(self._fcl_objects.keys()) != set(self.parts.keys()):
            self._fcl_objects = {name: part.make_fcl_object()
                                 for name, part in self.parts.items()}

    def is_translation_feasible(
        self,
        moving_part_name: str,
        translation: np.ndarray,
        *,
        ignore_self: bool = True,
        return_contacts: bool = False,
        depth_threshold: float | None = None,
    ):
        self._ensure_fcl_cache()
    
        if moving_part_name not in self._fcl_objects:
            raise KeyError(f"Unknown part in FCL cache: {moving_part_name}")
    
        moving_obj = self._fcl_objects[moving_part_name]
    
        req = fcl.CollisionRequest(
            enable_contact=return_contacts,
            num_max_contacts=100,
        )
    
        # --- Temporarily move cached object ---
        orig_tf = moving_obj.getTransform()
        try:
            orig_R = orig_tf.getRotation()
            orig_t = orig_tf.getTranslation()
    
            new_t = orig_t + np.asarray(translation, dtype=float)
            tmp_tf = fcl.Transform(orig_R, new_t)
    
            # Most python-fcl builds expose setTransform; computeAABB is often required after moving
            if hasattr(moving_obj, "setTransform"):
                moving_obj.setTransform(tmp_tf)
            else:
                # extremely defensive fallback; rare
                raise AttributeError("FCL CollisionObject has no setTransform() in this binding.")
    
            if hasattr(moving_obj, "computeAABB"):
                moving_obj.computeAABB()
    
            violating_contacts = []
    
            for name, other in self._fcl_objects.items():
                if ignore_self and name == moving_part_name:
                    continue

                res = fcl.CollisionResult()
                ret = fcl.collide(moving_obj, other, req, res)
    
                if ret <= 0:
                    continue
    
                if depth_threshold is None:
                    return (False, res.contacts) if return_contacts else False
    
                for c in res.contacts:
                    depth = getattr(c, "penetration_depth", getattr(c, "depth", None))
                    if depth is None or depth > depth_threshold:
                        violating_contacts.append(c)
                        break
    
                if violating_contacts:
                    return (False, violating_contacts) if return_contacts else False
    
            return (True, []) if return_contacts else True
    
        finally:
            # Restore original pose
            if hasattr(moving_obj, "setTransform"):
                moving_obj.setTransform(orig_tf)
            if hasattr(moving_obj, "computeAABB"):
                moving_obj.computeAABB()

    def check_visibility(self, part_to_check, viewing_directions, debug_visualization: bool = False):
        """
        Check if a part is visible from any of the given viewing directions.
        Uses Open3D raycasting to check for occlusions.
        
        Args:
            part_to_check: The Part object to check visibility for
            viewing_directions: List of direction vectors to check
            debug_visualization: If True, show a debug visualization with:
                - Occluding objects in kit_deep_blue (semi-transparent)
                - The relevant part in kit_yellowyellow
                - Each ray with its intersection in kit_deep_red
            
        Returns:
            bool: True if part is visible from at least one direction
        """
        # Merge all other meshes
        other_meshes = [p.mesh for p in self.parts.values() if p.name != part_to_check.name]
        if len(other_meshes) == 0:
            return True  # No other parts to occlude
            
        merged_mesh = o3d_merge_triangle_meshes(other_meshes)

        visible = False  # Flag to track occlusion status
        part_vertices = np.asarray(part_to_check.mesh.vertices)
        num_vertices = len(part_vertices)
        
        # For debug visualization: collect all rays and intersections
        debug_ray_origins = []
        debug_ray_endpoints = []
        debug_hit_points = []
        
        # Create raycasting scene
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh)
        scene.add_triangles(mesh_t)
        
        for direction in viewing_directions:
            direction = np.asarray(direction, dtype=np.float32)
            direction = direction / np.linalg.norm(direction)  # Normalize
            
            # Batch all vertices for this direction - create rays for ALL vertices at once
            ray_origins = part_vertices.astype(np.float32)
            ray_directions = np.tile(direction, (num_vertices, 1))
            
            # Create rays tensor (N, 6): [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
            rays = np.concatenate([ray_origins, ray_directions], axis=1)
            rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            
            # Cast all rays at once
            result = scene.cast_rays(rays_tensor)
            t_hit = result['t_hit'].numpy()
            
            # Check for misses (inf means no hit = visible)
            miss_mask = np.isinf(t_hit)
            
            # Collect debug info if needed
            if debug_visualization:
                ray_length = 100.0  # Default ray length for visualization
                for i in range(num_vertices):
                    debug_ray_origins.append(part_vertices[i])
                    if miss_mask[i]:  # No hit
                        debug_ray_endpoints.append(part_vertices[i] + direction * ray_length)
                    else:  # Hit
                        hit_point = part_vertices[i] + direction * t_hit[i]
                        debug_hit_points.append(hit_point)
                        debug_ray_endpoints.append(hit_point)
            
            # If any vertex is visible (no hit), the part is visible from this direction
            if np.any(miss_mask):
                visible = True
                if not debug_visualization:
                    print("Target mesh is visible from direction:", direction)
                    break
        
        # Debug visualization
        if debug_visualization:
            geoms = []
            
            # Get colors
            occluding_color = hex_to_rgb01(kit_deep_blue)
            target_color = hex_to_rgb01(kit_yellow)
            ray_color = hex_to_rgb01(kit_red)
            
            # Add occluding meshes (semi-transparent)
            occluding_mesh = o3d.geometry.TriangleMesh(merged_mesh)
            occluding_mesh.compute_vertex_normals()
            occluding_mesh.paint_uniform_color(occluding_color)
            
            try:
                mat_occluding = o3d.visualization.rendering.MaterialRecord()
                mat_occluding.shader = "defaultLitTransparency"
                mat_occluding.base_color = [occluding_color[0], occluding_color[1], occluding_color[2], 0.7]
                geoms.append({"name": "occluding_parts", "geometry": occluding_mesh, "material": mat_occluding})
            except AttributeError:
                # Fallback if transparency not supported
                geoms.append({"name": "occluding_parts", "geometry": occluding_mesh})
            
            # Add target part (kit_yellowyellow, opaque)
            target_mesh = o3d.geometry.TriangleMesh(part_to_check.mesh)
            target_mesh.compute_vertex_normals()
            target_mesh.paint_uniform_color(target_color)
            geoms.append({"name": "target_part", "geometry": target_mesh})
            
            # Add rays as lines (kit_deep_red)
            if len(debug_ray_origins) > 0:
                ray_origins_arr = np.array(debug_ray_origins)
                ray_endpoints_arr = np.array(debug_ray_endpoints)
                
                # Create line set for rays
                num_rays = len(ray_origins_arr)
                line_pts = np.vstack([ray_origins_arr, ray_endpoints_arr])
                lines = np.array([[i, i + num_rays] for i in range(num_rays)], dtype=np.int32)
                
                ray_lines = o3d.geometry.LineSet()
                ray_lines.points = o3d.utility.Vector3dVector(line_pts.astype(np.float64))
                ray_lines.lines = o3d.utility.Vector2iVector(lines)
                ray_lines.paint_uniform_color(ray_color)
                geoms.append({"name": "rays", "geometry": ray_lines})
            
            # Add hit points as spheres (kit_deep_red)
            if len(debug_hit_points) > 0:
                hit_pcd = o3d.geometry.PointCloud()
                hit_pcd.points = o3d.utility.Vector3dVector(np.array(debug_hit_points))
                hit_pcd.paint_uniform_color(ray_color)
                geoms.append({"name": "hit_points", "geometry": hit_pcd})
            
            # Show visualization
            print(f"\n=== Visibility Debug ===")
            print(f"Target part: {part_to_check.name}")
            print(f"Visible: {visible}")
            print(f"Total rays cast: {len(debug_ray_origins)}")
            print(f"Rays blocked (hits): {len(debug_hit_points)}")
            print(f"========================\n")
            
            o3d.visualization.draw(geoms, show_skybox=False)
                
        return visible

    def check_geometric_connections(self) -> int:
        """
        Open3D + python-fcl version of the old trimesh-based logic:

        For each part:
          - if it has exactly 1 connection
          - and that connection is geometric
          - and the part has no DOF (get_part_dof(part) returns empty)
        then:
          - merge this part's mesh into the other part (Open3D mesh merge)
          - remove this part from the assembly
          - rebuild the other part's FCL collision object

        Returns:
            int: number of parts merged/removed
        """
        merged_count = 0

        # snapshot because we may remove parts while iterating
        for part in list(self.parts.values()):
            conns = self.get_connections_by_part_name(part.name)

            if len(conns) != 1:
                continue

            conn = conns[0]
            if getattr(conn, "connection_type", None) != "geometric":
                continue

            # keep old behavior: "not self.get_part_dof(part)" means fixed
            if self.get_part_dof(part):
                continue

            # determine the other part
            if part.name == conn.part_name1:
                other_part = self.get_part_by_name(conn.part_name2)
            else:
                other_part = self.get_part_by_name(conn.part_name1)

            if other_part is None:
                continue

            print(f"INFO: {part.name} is fixed in place to {other_part.name}")

            # Make copies so we don't accidentally mutate shared references elsewhere.
            m_other = o3d.geometry.TriangleMesh(other_part.mesh)
            m_part = o3d.geometry.TriangleMesh(part.mesh)

            # Ensure triangle normals can be recomputed later if desired
            # (not strictly required for collision, but useful for visualization)
            m_other += m_part
            
            # Optional
            m_other.remove_duplicated_vertices()
            m_other.remove_degenerate_triangles()
            m_other.remove_duplicated_triangles()
            m_other.compute_vertex_normals()

            other_part.mesh = m_other

            # Remove the fixed part (also removes its connections + cache entries)
            self.remove_part(part)
            merged_count += 1

            # Geometry of other_part changed -> rebuild FCL object cache
            if hasattr(self, "_fcl_objects"):
                self._fcl_objects[other_part.name] = other_part.make_fcl_object()

        return merged_count

    def generate_figures(self, rows=3, cols=3, only_show_blocked=False):
        figures = []
        fig = plt.figure(figsize=(16, 9))
        i = 0
        for connection in self.connections:
            if not connection.DOF or not only_show_blocked:
                i += 1
                if connection.connection_type == "geometric":
                    pnts = np.array(connection.normals).transpose()
                    #pnts = np.array(connection.restrictions).transpose()
                    ax = fig.add_subplot(rows, cols, i, projection="3d")
                    # Set axis limits
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])
                    ax.set_aspect('equal')
                    ax.grid(True)
                    ax.scatter(pnts[0], pnts[1], pnts[2])
                    zero = np.zeros_like(pnts[0])
                    ax.quiver(zero, zero, zero, pnts[0], pnts[1], pnts[2], color='r', linewidths=1.0, length=1.0, arrow_length_ratio=0.15)
                if connection.DOF:
                    pnts = np.array(connection.DOF).transpose()
                    #ax.scatter(pnts[0], pnts[1], pnts[2], color='g')
                    zero = np.zeros_like(pnts[0])
                    ax.quiver(zero, zero, zero, pnts[0], pnts[1], pnts[2], color='g', linewidths=1.0, length=1.0, arrow_length_ratio=0.2)
                ax.set_title(connection.__str__(), fontsize=12)
            if i == rows * cols:
                # Adjust the spacing between subplots and show the plot
                plt.tight_layout()
                figures.append(fig)
                fig = plt.figure(figsize=(16, 9))
                i = 0
        if i != 0:
            figures.append(fig)
        # else:
        #    figures.pop()
        #    fig.clf()
        return figures
    
    def generate_surface_plot(self, angle_resolution = np.pi/8):
        num_parts = len(self.parts)
        num_long = int(np.round(2 * np.pi / angle_resolution))  # Number of longitudinal lines
        num_lat = int(np.round(np.pi / angle_resolution)) + 1  # Number of latitudinal lines

        # Define the colors: 0 is red, 1 is green
        colors = np.array([[178, 55, 44],[0, 150, 130]])

        # Define the row and column labels
        labels = []
        for part in self.parts.values():
            labels.append(part.name)

        # Create a figure with 3x3 subplots
        fig, axs = plt.subplots(num_parts, num_parts, figsize=(6, 6))

        matrix = []
        for i in range(num_parts):
            for j in range(num_parts):
                matrix.append(np.ones((num_lat, num_long), dtype=numpy.int8))

        for con in self.connections:
            if con.connection_type == "geometric":
                for i in range(num_long):
                    for j in range(num_lat):
                        theta = i * angle_resolution  # Longitudinal angle
                        phi = j * angle_resolution  # Latitudinal angle
                        x = round( np.sin(phi) * np.cos(theta), 5)
                        y = round(np.sin(phi) * np.sin(theta), 5)
                        z = round(np.cos(phi), 5)
                        if any(np.array_equal(np.array([x,y,z]), point) for point in con.restrictions):
                            matrix[labels.index(con.part_name1) * num_parts + labels.index(con.part_name2)][j,i] = 0
                        if any(np.array_equal(-np.array([x,y,z]), point) for point in con.restrictions):
                            matrix[labels.index(con.part_name2) * num_parts + labels.index(con.part_name1)][j,i] = 0
        # Plot each matrix
        for i in range(num_parts):
            for j in range(num_parts):
                # Convert the matrix to RGB colors
                image = colors[matrix[i*num_parts+j]]

                # Plot the matrix
                axs[i, j].imshow(image, aspect='auto')
                axs[i, j].axis('off')

        padding_l = 0.125
        padding_r = 0.1
        # Add row labels
        for i, label in enumerate(reversed(labels), start=0):
            fig.text(0.1, padding_l + (1-padding_l-padding_r)/num_parts*i, label, ha='right', va='center', rotation=0, fontsize = 5)

        # Add column labels
        for i, label in enumerate(labels):
            fig.text(padding_l + (1-padding_l-padding_r)/num_parts*i, 0.9, label, ha='center', va='bottom', rotation=90, fontsize = 5)
        return fig

    def save_as_xml(self, file_path):
        root = ET.Element('assembly')
        root.set('name', self.name)
        base_dir = os.path.dirname(os.path.abspath(file_path))

        # Add part elements
        for part in self.parts.values():
            part_element = part.to_xml_element(base_dir)
            root.append(part_element)

        # Add connection elements
        for connection in self.connections:
            #print(f"{os.path.splitext(file_path)[0]}_data/")
            if isinstance(connection, Geometric_Connection):
                if not os.path.exists(f"{os.path.splitext(file_path)[0]}_data/"):
                    os.makedirs(f"{os.path.splitext(file_path)[0]}_data/")
                connection_element = connection.to_xml_element(
                    f"{os.path.splitext(file_path)[0]}_data/", base_dir)
            else:
                connection_element = connection.to_xml_element()

            root.append(connection_element)

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write(file_path, encoding="utf-8")
        self.xml_path = file_path

    def save_part_meshes(self, directory, overwrite = False):
        #print(directory)
        os.makedirs(directory, exist_ok=True)
        for part in self.parts.values():
            filename = os.path.join(directory, part.name + '.stl')
            if not os.path.exists(filename) or overwrite:
                mesh = copy.deepcopy(part.mesh)
                mesh.compute_triangle_normals()  # STL needs normals, otherwise the file is empty
                if not o3d.io.write_triangle_mesh(filename, mesh):
                    raise IOError(f"Failed to write mesh for part '{part.name}' to {filename}")
            part.mesh_path = filename
    def export_full(self, file_path):
        self.save_part_meshes(f"{os.path.dirname(file_path)}/stl", overwrite=True)
        self.save_as_xml(file_path)

    def load_part_meshes(self):
        for part in self.parts.values():
            part.load_mesh()
            #part.mesh = trimesh.load(part.mesh_path, force="mesh")

    def delete_part_meshes(self):
        # Replace the meshes of the parts with None
        for part in self.parts.values():
            part.delete_mesh()
            #part.mesh = None

    def generate_urdf_files(self, directory, mesh_dir = None):
        if mesh_dir is None:
            mesh_dir_urdf = "mesh"
            mesh_dir_save = os.path.join(directory, mesh_dir_urdf)
        else:
            mesh_dir_urdf = os.path.relpath(mesh_dir, directory)
            mesh_dir_save = mesh_dir
        os.makedirs(directory, exist_ok=True)
        self.save_part_meshes(mesh_dir_save)
        for part in self.parts.values():
            # create root element
            robot = ET.Element('robot', {'name': part.name})

            # create link elements
            baselink = ET.Element('link', {'name': "baselink"})
            link = ET.Element('link', {'name': "geometry"})

            # create joint element
            joint = ET.Element('joint', {'name': "dummy", 'type': "fixed"})
            parent = ET.Element('parent', {'link': "baselink"})
            child = ET.Element('child', {'link': "geometry"})
            joint.append(parent)
            joint.append(child)

            # create contact element
            contact = ET.Element('contact')
            contact_params = {'lateral_friction': 1.0,
                              'spinning_friction': 0.2,
                              'rolling_friction': 0.001,
                              'contact_cfm': 0.0,
                              'contact_erp': 0.8,
                              'damping': 1000,
                              'stiffness': 30000}
            for key, value in contact_params.items():
                param = ET.Element(key, {'value': str(value)})
                contact.append(param)

            # create inertial element
            inertial = ET.Element('inertial')
            origin_point = part.center_of_mass
            origin = ET.Element('origin', {'rpy': '0 0 0', 'xyz': f'{part.center_of_mass[0]} {part.center_of_mass[1]} {part.center_of_mass[2]}'})
            inertial.append(origin)
            mass = ET.Element('mass', {'value': f'{part.weight*0.000000001}'})
            inertial.append(mass)
            inertia = ET.Element('inertia', {'ixx': '1',
                                             'ixy': '0',
                                             'ixz': '0',
                                             'iyy': '1',
                                             'iyz': '0',
                                             'izz': '1'})
            inertial.append(inertia)

            # create visual element
            visual = ET.Element('visual')
            origin = ET.Element('origin', {'rpy': '0 0 0', 'xyz': '0 0 0'})
            visual.append(origin)
            geometry = ET.Element('geometry')
            mesh = ET.Element('mesh', {'filename': mesh_dir_urdf + "/" + part.name + '.stl'})
            geometry.append(mesh)
            visual.append(geometry)
            material = ET.Element('material', {'name': 'base'})
            color = ET.Element('color', {'rgba': '0.7 0.7 0.7 1'})
            material.append(color)
            visual.append(material)

            # create collision element
            collision = ET.Element('collision')
            origin = ET.Element('origin', {'rpy': '0 0 0', 'xyz': '0 0 0'})
            collision.append(origin)
            geometry = ET.Element('geometry')
            mesh = ET.Element('mesh', {'filename': mesh_dir_urdf + "/" + part.name + '.stl'})
            geometry.append(mesh)
            collision.append(geometry)

            # append all elements to the root element
            link.append(contact)
            link.append(inertial)
            link.append(visual)
            link.append(collision)
            robot.append(baselink)
            robot.append(joint)
            robot.append(link)

            # Write to file
            tree = ET.ElementTree(robot)
            ET.indent(tree, space="\t", level=0)
            tree.write(directory + '/' + part.name + '.urdf', encoding="utf-8")

    def copy_without_meshs(self):
        # Create a deep copy of the assembly object
        new_assembly = copy.deepcopy(self)

        # Replace the meshes of the parts with None
        new_assembly.delete_part_meshes()

        return new_assembly

    def assign_agent(self, agent_name: str, part_name: str, grasp_id, matrix=None, opening_width=50, gripper_type="gripper", tcp_offset_mm=294.0):
        """
        Weist einem Agenten ein Bauteil zu und speichert die Griff-Pose.

        Deprecated: use ``CellState.holds``.
        """
        _warn_hold_api("assign_agent")
        # 1. Existiert der Agent?
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        # 2. Ist der Agent frei?
        if self.agents[agent_name] is not None:
            current_hold = self.agents[agent_name]
            raise ValueError(f"Agent {agent_name} is busy holding {current_hold.part_name}")

        # 3. Wird das Teil schon von jemand anderem gehalten?
        holder = next((n for n, g in self.agents.items() if g is not None and g.part_name == part_name), None)
        if holder is not None:
            raise ValueError(f"Part {part_name} is already held by {holder}")

        # 4. Zuweisen
        self.agents[agent_name] = ActiveGrasp(
            agent_name=agent_name,
            part_name=part_name,
            grasp_id=str(grasp_id),
            matrix=matrix,
            opening_width=opening_width,
            gripper_type=gripper_type,
            tcp_offset_mm=tcp_offset_mm,
        )

    def release_agent(self, agent_name):
        """Deprecated: use ``CellState.holds``."""
        _warn_hold_api("release_agent")
        if agent_name in self.agents:
            self.agents[agent_name] = None

            if hasattr(self, "_gripper_fcl_cache"):
                self._gripper_fcl_cache = {}

    def get_part_holder(self, part_name: str):
        """Gibt den Namen des Agenten zurück, der dieses Teil hält (oder None).

        Deprecated: use ``CellState.holds``.
        """
        _warn_hold_api("get_part_holder")
        for agent_name, active_grasp in self.agents.items():
            if active_grasp is not None and active_grasp.part_name == part_name:
                return agent_name
        return None
    
    def is_agent_busy(self, agent_name):
        """Deprecated: use ``CellState.holds``."""
        _warn_hold_api("is_agent_busy")
        agent = self.agents.get(agent_name)
        if agent is None: 
            return False 
        return agent.part_name is not None

    @classmethod
    def load_from_xml(cls, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        base_dir = os.path.dirname(os.path.abspath(filename))

        assembly_name = root.attrib['name']
        assembly = cls(assembly_name)

        # Load parts
        for part_element in root.findall('part'):
            assembly.add_part(Part.from_xml_element(part_element, base_dir))

        # Load connections
        for conn_element in root.findall('connection'):
            conn_type = conn_element.attrib['type']
            if conn_type == 'generic':
                connection = Connection.from_xml_element(conn_element)
            elif conn_type == 'geometric':
                connection = Geometric_Connection.from_xml_element(conn_element, base_dir)
            elif conn_type == 'screw':
                connection = Screw_Connection.from_xml_element(conn_element)
            elif conn_type == 'safety_ring':
                connection = Safety_Ring_Connection.from_xml_element(conn_element)
            elif conn_type == 'fixed':
                connection = Fixed_Connection.from_xml_element(conn_element)
            assembly.add_connection(connection)
        assembly.xml_path = filename
        return assembly


if __name__ == '__main__':
    #parts = load_parts_from_folder(os.path.join("tests","electric_motor"),0)
    #asm = Assembly("test", parts)
    #asm.find_contacts()
    #asm.save_as_xml(os.path.join("tests","electric_motor","assembly_output.xml"))
    asm = Assembly.load_from_xml(os.path.join("tests","internal_cases","visibility_ring_case","xml","visibility_ring_case_r3.00.xml"))
    asm.show_all_contact_points(normal_len=10, show_edges=False)

    asm.check_visibility(
        asm.get_part_by_name("ring"),
        viewing_directions=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        debug_visualization=True,
    )
    
    connection = asm.connections[0]
    asm.debug_contact_detection(connection.part_name1, connection.part_name2)
    
    asm.show_connection_dof(connection, show_edges=False)

    asm.show()
    
    #asm.show_connection_dof(connection)
    fig = asm.generate_surface_plot()
    plt.show()