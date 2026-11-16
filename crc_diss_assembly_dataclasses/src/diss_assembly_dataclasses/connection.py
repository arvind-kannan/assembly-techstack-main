from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from .xml_paths import resolve_xml_path, relativize_xml_path
#from scipy.optimize import minimize

# Global cache for discretized sphere samples
_SPHERE_CACHE: dict[tuple, np.ndarray] = {}

def count_points_on_side_of_plane(normal, points, tollerance = 0.01):
    """
    Counts the number of points on the side of the plane defined by the given normal vector.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Calculate the signed distance from each point to the plane
    #distances = np.dot(points - plane_point, plane_normal)
    distances = np.dot(points, normal)

    # Count the number of points on the side of the plane defined by the normal
    count = np.sum(distances >= tollerance)
    """count = 0
    for distance in distances:
        if abs(distance)<0.2:
            count += 2
        elif distance > 0:
            count +=1"""

    return count

def evenly_spaced_points_on_sphere(
    radius: float = 1.0,
    *,
    angle_resolution: float | None = None,
    n_points: int | None = None,
    decimals: int = 5,
    dtype=np.float64,
    antipodal: bool = True,
) -> list[np.ndarray]:
    """
    Generate approximately evenly spaced points on a sphere using the Fibonacci lattice.

    Invariant:
      - Every call with the same (radius, angle_resolution, n_points, decimals, dtype) must return
        numerically identical points (after normalize+round+unique), enabling exact deletion checks.
      - Deterministic output for same params (after normalize+round+unique).
      - If antipodal=True: for every vector v in universe, -v is also in universe.

    Notes:
      - If angle_resolution is provided, n_points is derived deterministically.
      - All points are normalized to unit length (then scaled by radius), then rounded to `decimals`.
      - Returned value is a list of np.ndarray shape (3,), consistent with old code expectations.
      - Here, `n_points` refers to the *base* Fibonacci count (e.g., 162).
        If antipodal=True, returned size is ~2*n_points (e.g., 324).
    """
    dtype = np.float64

    # --- 1) Deterministic base_n selection ---
    base_n: int
    if n_points is None:
        if angle_resolution is not None:
            base_n = max(8, int(round(4.0 * np.pi / (float(angle_resolution) ** 2))))
        else:
            base_n = 162  # lock default base universe size
    else:
        base_n = int(n_points)

    # Cache key must include antipodal flag and base_n (not final size)
    key = (float(radius),
           float(angle_resolution) if angle_resolution is not None else None,
           int(base_n),
           int(decimals),
           bool(antipodal))

    cached = _SPHERE_CACHE.get(key)
    if cached is None:
        # --- 2) Fibonacci lattice sampling (deterministic) ---
        i = np.arange(base_n, dtype=np.int64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden_angle * i
        z = 1.0 - 2.0 * (i + 0.5) / float(base_n)
        r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))

        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        base = np.stack((x, y, z), axis=1).astype(np.float64, copy=False)

        # --- 3) Normalize (unit), scale, round (BEFORE antipodal mirroring) ---
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        base = base / norms

        if radius != 1.0:
            base = base * float(radius)

        base = np.round(base, int(decimals))
        base[base == 0.0] = 0.0

        # --- 4) Antipodal completion ---
        if antipodal:
            pts = np.vstack([base, -base])
            pts = np.round(pts, int(decimals))
            pts[pts == 0.0] = 0.0
        else:
            pts = base

        # --- 5) Unique (stable) ---
        uniq, idx = np.unique(pts, axis=0, return_index=True)
        pts = uniq[np.sort(idx)]

        _SPHERE_CACHE[key] = pts
        cached = pts

    return [cached[i, :] for i in range(cached.shape[0])]

class Connection:
    def __init__(self, part_name1, part_name2):
        self.part_name1 = part_name1
        self.part_name2 = part_name2
        self.connection_type ="generic"
        #self.main_DOF = np.array([0.0,0.0,0.0])
        self.DOF = []
        self.restrictions = []

    def __str__(self):
        return f"{self.connection_type} connection: {self.part_name1} <-> {self.part_name2}"

    def __repr__(self):
        return f"{self.connection_type} connection: {self.part_name1} <-> {self.part_name2}"

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return False
        state = 0
        if (self.part_name1 == other.part_name1 and
            self.part_name2 == other.part_name2):
            state = 1
        elif (self.part_name1 == other.part_name2 and
              self.part_name2 == other.part_name1):
            state = 2
        if not (state>0 and
                self.connection_type == other.connection_type):
            return False
        if len(self.DOF) != len(other.DOF):
            return False

        for dof1, dof2 in zip(self.DOF, other.DOF):
            if not all(x == y for x, y in zip(dof1, dof2)):
                print(f"{self.part_name1} {self.part_name2} {other.part_name1} {other.part_name2}")
                return False
        return True

    def to_xml_element(self):
        element = ET.Element('connection')
        element.set('type', self.connection_type)
        element.set('part_name1', self.part_name1)
        element.set('part_name2', self.part_name2)
        return element
    @classmethod
    def from_xml_element(cls, conn_element):
        part_name1 = conn_element.attrib['part_name1']
        part_name2 = conn_element.attrib['part_name2']
        return cls(part_name1, part_name2)


class Geometric_Connection(Connection):
    def __init__(self, part_name1, part_name2,points, normals, distances):
        super().__init__(part_name1, part_name2)
        self.points = points
        self.normals = normals
        self.distances = distances
        self.connection_type = "geometric"
        self.calculate_DOF_sphere()

    def __str__(self):
        return f"{self.connection_type}: {self.part_name1} <-> {self.part_name2}"
              # f"with {len(self.points)} points"

    def to_xml_element(self, file_prefix="", base_dir=None):
        element = super().to_xml_element()
        self.points_filename = f"{file_prefix}{self.part_name1}_{self.part_name2}_points.npy"
        np.save(self.points_filename, self.points)
        element.set('points_filename', self.points_filename if base_dir is None else relativize_xml_path(self.points_filename, base_dir))
        self.normals_filename = f"{file_prefix}{self.part_name1}_{self.part_name2}_normals.npy"
        np.save(self.normals_filename, self.normals)
        element.set('normals_filename', self.normals_filename if base_dir is None else relativize_xml_path(self.normals_filename, base_dir))
        self.distances_filename = f"{file_prefix}{self.part_name1}_{self.part_name2}_distances.npy"
        np.save(self.distances_filename, self.distances)
        element.set('distances_filename', self.distances_filename if base_dir is None else relativize_xml_path(self.distances_filename, base_dir))
        return element


    def calculate_DOF(self, resolution=np.pi/16):
        theta = np.arctan2(self.normals[:, 1], self.normals[:, 0])
        phi = np.arccos(self.normals[:, 2])

        # Round the theta and phi values to the desired resolution
        theta_rounded = np.round(theta / resolution) * resolution
        phi_rounded = np.round(phi / resolution) * resolution

        # Convert the 3D polar coordinates back to Cartesian coordinates
        x = np.sin(phi_rounded) * np.cos(theta_rounded)
        y = np.sin(phi_rounded) * np.sin(theta_rounded)
        z = np.cos(phi_rounded)

        # Combine the Cartesian coordinates into a single array
        cartesian_coords = np.stack((x, y, z), axis=1)

        # Remove duplicates from the point set
        unique_coords, indices = np.unique(np.round(cartesian_coords, 3), axis=0, return_index=True)

        self.main_DOF = -np.sum(unique_coords, axis=0)

    def calculate_DOF_sphere(self):
        possible_normals = evenly_spaced_points_on_sphere()
        self.DOF = []
        self.restrictions = []
    
        threshold = len(self.normals) / len(possible_normals) * 0.1
    
        for p in possible_normals:
            found_points = count_points_on_side_of_plane(p, self.normals, 0.2)
            if found_points <= threshold:
                self.DOF.append(p)
    
        # exact complement via quantized key at the SAME decimals
        dof_keys = {tuple(v.tolist()) for v in self.DOF}
        self.restrictions = [p for p in possible_normals if tuple(p.tolist()) not in dof_keys]

    @classmethod
    def from_xml_element(cls, conn_element, base_dir=None):
        part_name1 = conn_element.attrib['part_name1']
        part_name2 = conn_element.attrib['part_name2']
        points = np.load(resolve_xml_path(conn_element.attrib['points_filename'], base_dir))
        normals = np.load(resolve_xml_path(conn_element.attrib['normals_filename'], base_dir))
        distances = np.load(resolve_xml_path(conn_element.attrib['distances_filename'], base_dir))
        return cls(part_name1, part_name2, points, normals, distances)

class Screw_Connection(Connection):
    def __init__(self, part_name1, part_name2, screw_head):
        super().__init__(part_name1, part_name2)
        self.connection_type = "screw"
        self.screw_head = screw_head
        self.DOF=[]
        self.restrictions=evenly_spaced_points_on_sphere()

    def __str__(self):
        return f"{self.connection_type} connection: {self.part_name1} <-> {self.part_name2}"

    def to_xml_element(self):
        element = super().to_xml_element()
        element.set('screw_head', self.screw_head)
        return element

    @classmethod
    def from_xml_element(cls, conn_element):
        part_name1 = conn_element.attrib['part_name1']
        part_name2 = conn_element.attrib['part_name2']
        screw_head = conn_element.attrib['screw_head']
        return cls(part_name1, part_name2, screw_head)

class Fixed_Connection(Connection):
    def __init__(self, part_name1, part_name2):
        super().__init__(part_name1, part_name2)
        self.connection_type = "fixed"
        self.DOF=[]
        self.restrictions=evenly_spaced_points_on_sphere()

    def __str__(self):
        return f"{self.connection_type} connection: {self.part_name1} <-> {self.part_name2}"

    def to_xml_element(self):
        element = super().to_xml_element()
        return element

    @classmethod
    def from_xml_element(cls, conn_element):
        part_name1 = conn_element.attrib['part_name1']
        part_name2 = conn_element.attrib['part_name2']
        return cls(part_name1, part_name2)

class Safety_Ring_Connection(Connection):
    def __init__(self, part_name1, part_name2):
        super().__init__(part_name1, part_name2)
        self.connection_type = "safety_ring"
        self.DOF=[]
        self.restrictions=evenly_spaced_points_on_sphere()

    def __str__(self):
        return f"{self.connection_type} connection: {self.part_name1} <-> {self.part_name2}"

    def to_xml_element(self):
        element = super().to_xml_element()
        return element

    @classmethod
    def from_xml_element(cls, conn_element):
        part_name1 = conn_element.attrib['part_name1']
        part_name2 = conn_element.attrib['part_name2']
        return cls(part_name1, part_name2)