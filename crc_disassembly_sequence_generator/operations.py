import numpy as np
import open3d as o3d
import copy
import dataclasses
import trimesh
from diss_assembly_dataclasses.connection import Geometric_Connection, Screw_Connection
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.cell_state import CellState
from diss_assembly_dataclasses.part import Part
try:
    from utility import check_reachability_o3d_raycast
except Exception:  # legacy fallback
    check_reachability_o3d_raycast = None
try:
    from utility import check_reachability
except Exception:  # legacy fallback
    check_reachability = None
def _safe_check_reachability(mill_volume, part_mesh, direction, visualize=False) -> bool:
    """Compatibility wrapper for legacy and Open3D reachability helpers."""
    fn = check_reachability_o3d_raycast or check_reachability
    if fn is None:
        # Demo-safe fallback. The EMO defense rollout does not use milling.
        return True

    try:
        return bool(fn(mill_volume, part_mesh, direction, visualize=visualize))
    except TypeError:
        return bool(fn(mill_volume, part_mesh, direction))


from scipy.spatial.transform import Rotation as SciRot
from scipy.spatial.transform import Rotation as R


def rgb255_to_rgb01(rgb) -> tuple[float, float, float]:
    rgb = np.asarray(rgb, dtype=np.float64)
    return tuple((rgb / 255.0).tolist())


KIT_DEEP_BLUE = rgb255_to_rgb01([20, 68, 102])
KIT_BLUE_GREEN = rgb255_to_rgb01([0, 131, 143])
KIT_YELLOW = rgb255_to_rgb01([238, 183, 13])
KIT_RED = rgb255_to_rgb01([178, 55, 44])
WBK_GREEN = rgb255_to_rgb01([0, 150, 130])
WBK_GREY = rgb255_to_rgb01([59, 79, 88])
WBK_LIGHT_GREY = rgb255_to_rgb01([196, 208, 207])
WBK_LIGHT_BLUE = rgb255_to_rgb01([60, 152, 217])
WBK_LIGHT_GREEN = rgb255_to_rgb01([39, 255, 226])

VISUAL_THEME = {
    "assembly_context": WBK_LIGHT_GREY,
    "assembly_context_dark": WBK_GREY,
    "moving_part": KIT_YELLOW,
    "direction": KIT_RED,
    "tool": KIT_BLUE_GREEN,
    "milled_main": WBK_GREEN,
    "milled_fragment": WBK_LIGHT_BLUE,
    "mill_volume": KIT_DEEP_BLUE,
}


def _as3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


def o3d_copy_mesh_for_display(
    mesh: o3d.geometry.TriangleMesh,
    color=VISUAL_THEME["assembly_context"],
) -> o3d.geometry.TriangleMesh:
    """Return a copied Open3D mesh with normals and uniform color."""
    if mesh is None:
        raise ValueError("mesh is None")
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError(f"Expected open3d.geometry.TriangleMesh, got {type(mesh)}")
    m = copy.deepcopy(mesh)
    m.compute_vertex_normals()
    m.paint_uniform_color(list(color))
    return m


# Backward-compatible legacy name used by older scene methods.
def o3d_mesh_copy_for_display(mesh: o3d.geometry.TriangleMesh, color=(0.5, 0.5, 0.5), alpha: float | None = None):
    return o3d_copy_mesh_for_display(mesh, color=color)


def o3d_sphere(
    center,
    radius: float = 1.5,
    resolution: int = 10,
    color=VISUAL_THEME["direction"],
) -> o3d.geometry.TriangleMesh:
    c = _as3(center)
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=int(resolution))
    s.translate(c)
    s.compute_vertex_normals()
    s.paint_uniform_color(list(color))
    return s


def o3d_rotation_from_z_to_dir(direction: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps +Z to the given direction."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        return np.eye(3)
    d = d / dn
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0], dtype=np.float64))
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def o3d_make_transform(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = _as3(t)
    return T


def o3d_lineset_from_path(path: np.ndarray, color=VISUAL_THEME["direction"]) -> o3d.geometry.LineSet:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("path must be (N,3) with N>=2")
    lines = [[i, i + 1] for i in range(p.shape[0] - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(p),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([color], dtype=np.float64), (len(lines), 1)))
    return ls


def o3d_lineset_from_direction(origin: np.ndarray, direction: np.ndarray, length: float, color=VISUAL_THEME["direction"]):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("direction must be non-zero")
    d = direction / norm
    end = origin + d * float(length)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([origin, end])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array([color], dtype=np.float64))
    return ls


def o3d_cylinder_between(p0, p1, radius: float = 0.5, resolution: int = 20, color=(1.0, 0.0, 0.0)):
    a = _as3(p0)
    b = _as3(p1)
    v = b - a
    h = float(np.linalg.norm(v))
    if h <= 1e-12:
        return o3d_sphere(a, radius=max(radius, 1e-3), color=color)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=resolution, split=4)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(list(color))
    Rm = o3d_rotation_from_z_to_dir(v)
    cyl.rotate(Rm, center=np.zeros(3))
    cyl.translate((a + b) * 0.5)
    return cyl


def o3d_make_tool_cylinder(
    tcp: np.ndarray,
    direction: np.ndarray,
    *,
    radius: float,
    height: float,
    offset: float,
    color=(0.78, 0.78, 0.40),
    resolution: int = 48,
) -> o3d.geometry.TriangleMesh:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d_unit = d / (np.linalg.norm(d) + 1e-12)
    center = _as3(tcp) + d_unit * float(offset)
    h = float(height)
    tool = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=h, resolution=int(resolution), split=4)
    tool.translate((0.0, 0.0, -0.5 * h))
    tool.compute_vertex_normals()
    tool.paint_uniform_color(list(color))
    tool.transform(o3d_make_transform(o3d_rotation_from_z_to_dir(d_unit), center))
    return tool


def o3d_geoms_from_assembly_context(
    assembly,
    *,
    exclude_names: set[str] | None = None,
    color=VISUAL_THEME["assembly_context"],
) -> list[o3d.geometry.Geometry]:
    geoms: list[o3d.geometry.Geometry] = []
    if assembly is None:
        return geoms
    ex = exclude_names or set()
    for part in assembly.parts.values():
        if part is None or getattr(part, "mesh", None) is None:
            continue
        if getattr(part, "name", None) in ex:
            continue
        if isinstance(part.mesh, o3d.geometry.TriangleMesh):
            geoms.append(o3d_copy_mesh_for_display(part.mesh, color=color))
    return geoms


def o3d_mesh_extent_diag(mesh: o3d.geometry.TriangleMesh) -> float:
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent(), dtype=np.float64)
    return float(np.linalg.norm(extent))


def o3d_scene_extent_diag_from_assembly(assembly: Assembly, exclude_names: set[str] | None = None) -> float:
    mins = []
    maxs = []
    ex = exclude_names or set()
    for part in assembly.parts.values():
        if part is None or getattr(part, "mesh", None) is None:
            continue
        if getattr(part, "name", None) in ex:
            continue
        if not isinstance(part.mesh, o3d.geometry.TriangleMesh):
            continue
        aabb = part.mesh.get_axis_aligned_bounding_box()
        mins.append(np.asarray(aabb.min_bound, dtype=np.float64))
        maxs.append(np.asarray(aabb.max_bound, dtype=np.float64))
    if not mins:
        return 1.0
    return float(np.linalg.norm(np.max(np.vstack(maxs), axis=0) - np.min(np.vstack(mins), axis=0)))


def choose_direction_render_length(
    *,
    part_mesh: o3d.geometry.TriangleMesh,
    assembly: Assembly,
    pullout_distance: float,
    exclude_names: set[str] | None = None,
) -> float:
    part_diag = o3d_mesh_extent_diag(part_mesh)
    scene_diag = o3d_scene_extent_diag_from_assembly(assembly, exclude_names=exclude_names)
    base = 0.8 * float(pullout_distance)
    lower = max(0.6 * part_diag, 1e-3)
    upper = max(0.25 * scene_diag, lower)
    return float(np.clip(base, lower, upper))


def interpolate_polyline(path: np.ndarray, num_frames: int) -> list[np.ndarray]:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] == 0:
        return []
    if p.shape[0] == 1:
        return [p[0].copy() for _ in range(num_frames)]
    seg_lens = np.linalg.norm(p[1:] - p[:-1], axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len < 1e-12:
        return [p[0].copy() for _ in range(num_frames)]
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    samples = np.linspace(0.0, total_len, num_frames)
    out: list[np.ndarray] = []
    seg_idx = 0
    for s in samples:
        while seg_idx < len(seg_lens) - 1 and s > cum[seg_idx + 1]:
            seg_idx += 1
        seg_start = cum[seg_idx]
        seg_end = cum[seg_idx + 1]
        alpha = 0.0 if seg_end <= seg_start else (s - seg_start) / (seg_end - seg_start)
        out.append(((1.0 - alpha) * p[seg_idx] + alpha * p[seg_idx + 1]).astype(np.float64))
    return out


class Operation:
    def __init__(self,  parameter: dict):
        self.data = parameter

    def __eq__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def compare(self, *args, **kwargs):
        raise NotImplementedError

    def check_feasability(self, assembly):
        raise NotImplementedError

    def apply_to_assembly(self, assembly):
        raise NotImplementedError

    def scene(self, assembly: Assembly):
        raise NotImplementedError


def _holder_of(cell_state, part_name):
    """Return the robot holding ``part_name`` in ``cell_state`` (or None)."""
    if cell_state is None:
        return None
    for robot, held in cell_state.holds.items():
        if held == part_name:
            return robot
    return None


def _remove_part_from_assembly(assembly, part_name):
    """Remove a part using the current assembly API when available."""
    work_copy = copy.deepcopy(assembly)
    getter = getattr(work_copy, "get_part_by_name", None)
    part = None
    if callable(getter):
        try:
            part = getter(part_name)
        except Exception:
            part = None

    if part is not None:
        remover = getattr(work_copy, "remove_part", None)
        if callable(remover):
            try:
                remover(part)
                return work_copy
            except Exception:
                pass

    parts = getattr(work_copy, "parts", None)
    if isinstance(parts, dict) and part_name in parts:
        try:
            del parts[part_name]
        except Exception:
            pass
    return work_copy


class MillingOperation(Operation):
    def __init__(self, parameter: dict):
        """
            operation_parameter = {"part_not_milled": part_a,
                                   "part_milled" : part_b,
                                   "mill_volume" : mill_volume,
                                   "mill_diameter" : mill_diameter,
                                   "cylinder_data" : cylinder_max_data,
                                   "resulting_mesh" : resulting_mesh,
                                   "path" : path,
                                   "direction" : direction}
           also contains "connection" and "cost"
        """
        super().__init__(parameter)
        self.data["type"]="milling"
        #self.data = parameter


    def __eq__(self, other):
        if not isinstance(other, MillingOperation):
            return False
        if self.data["inverted"] == other.data["inverted"] and self.data["connection"] == other.data["connection"] \
           and self.data["type"] == other.data["type"]:
            return True
        else:
            return False

    def __str__(self):
        return f'{self.data["type"]} {self.data["part_milled"].name}'

    def compare(self, connection: Geometric_Connection, inverted):
        if self.data["inverted"] == inverted and self.data["connection"] == connection:
            return True
        else:
            return False

    def check_feasability(self, assembly: Assembly):
        #check weather the two parts are of similar size. If so the operation is not feasible --> TODO implement better or more detailed check
        if self.data["similar_radius"] or self.data["no_intersection_found"]:
            return False
        for part in assembly.parts.values():
            #if part.name == "Lower_Lid" and self.data["part_milled"].name == "Upper_Lid":
            #    print(part.mesh.is_watertight)
            #    check_reachability(self.data["mill_volume"], copy.deepcopy(part.mesh), self.data["direction"], visualize=True)
            if part != self.data["part_milled"] and part and \
                    not _safe_check_reachability(self.data["mill_volume"], copy.deepcopy(part.mesh), self.data["direction"], visualize=False):
                return False


        if self.data["cylinder_data"]["height"] > self.data["mill_diameter"] * 5: #TODO global parameter for mill length
            return False
        return True

    def apply_to_assembly(self, assembly: Assembly, cost_milling):
        return self.static_apply_to_assembly(self, assembly, cost_milling)
        """#TODO find better solution than replacing "part_milled" --> currently necessary for sucessfull feasability check
        t_part = self.data["part_milled"]
        if not self.data["inverted"]:
            self.data["part_milled"] = assembly.get_part_by_name(self.data["connection"].part_name2)
        else:
            self.data["part_milled"] = assembly.get_part_by_name(self.data["connection"].part_name1)
        if not (self.check_feasability(assembly)):
            self.data["part_milled"] = t_part
            return -1, assembly
        self.data["part_milled"] = t_part
        assembly_work_copy = copy.deepcopy(assembly)
        if not self.data["inverted"]:
            part_to_be_milled = assembly_work_copy.get_part_by_name(self.data["connection"].part_name2)
            other_part = assembly_work_copy.get_part_by_name(self.data["connection"].part_name1)
        else:
            part_to_be_milled = assembly_work_copy.get_part_by_name(self.data["connection"].part_name1)
            other_part = assembly_work_copy.get_part_by_name(self.data["connection"].part_name2)
        part_to_be_milled.mesh = self.data["part_milled"].mesh
        print(f"applying existing operation: {other_part.name} {part_to_be_milled.name}")
        if self.data["resulting_mesh"] is not None:
            new_part_name = part_to_be_milled.name + "_milled_" + other_part.name +"_2"
            if any(p.name == new_part_name for p in assembly_work_copy.parts):
                assembly_work_copy.remove_part(assembly_work_copy.get_part_by_name(new_part_name))
            #save_path_mesh = assembly.name + "\\" + new_part_name + ".stl"
            #print(save_path_mesh)
            #resulting_mesh1.export(save_path_mesh)
            new_part = Part(new_part_name, "", mesh=self.data["resulting_mesh"])
            assembly_work_copy.add_part(new_part)
            for part in assembly_work_copy.parts.values():
                if part != new_part:
                    assembly_work_copy.check_pair_for_contact([new_part, part])
            existing_connections = assembly_work_copy.get_connections_by_part_name(new_part.name)
            if len(existing_connections) < 1:
                assembly_work_copy.remove_part(new_part)
            elif len(existing_connections) == 1 and not assembly_work_copy.get_part_dof(new_part):
                if new_part.name == existing_connections[0].part_name1:
                    other_part = assembly_work_copy.get_part_by_name(existing_connections[0].part_name2)
                else:
                    other_part = assembly_work_copy.get_part_by_name(existing_connections[0].part_name1)
                other_part.mesh = trimesh.util.concatenate([other_part.mesh, new_part.mesh])
                assembly_work_copy.rename_single_part(other_part, other_part.name + "_combined_" + new_part.name)
        #assembly_work_copy.remove_connection(self.data["connection"])
        if part_to_be_milled.mesh is None:
            assembly_work_copy.remove_part(part_to_be_milled)
        else:
            assembly_work_copy.rename_single_part(part_to_be_milled, part_to_be_milled.name + "_milled_" + other_part.name)
            #assembly_work_copy.check_pair_for_contact([other_part, part_to_be_milled])

            parts_to_recheck = []
            existing_connections = assembly_work_copy.get_connections_by_part_name(part_to_be_milled.name)
            for ex_con in existing_connections:
                if ex_con.connection_type == "geometric":
                    if ex_con.part_name1 == part_to_be_milled.name:
                        parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name2))
                    else:
                        parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name1))
                    assembly_work_copy.remove_connection(ex_con)
            for part in parts_to_recheck:
                assembly_work_copy.check_pair_for_contact([part_to_be_milled, part])

            existing_connections = assembly_work_copy.get_connections_by_part_name(part_to_be_milled.name)
            if len(existing_connections)<1:
                assembly_work_copy.remove_part(part_to_be_milled)
            elif len(existing_connections)==1 and existing_connections[0].connection_type == "geometric" and not assembly_work_copy.get_part_dof(part_to_be_milled):
                if part_to_be_milled.name == existing_connections[0].part_name1:
                    other_part = assembly_work_copy.get_part_by_name(existing_connections[0].part_name2)
                else:
                    other_part = assembly_work_copy.get_part_by_name(existing_connections[0].part_name1)
                other_part.mesh = trimesh.util.concatenate([other_part.mesh, part_to_be_milled.mesh])
                assembly_work_copy.rename_single_part(other_part, other_part.name + "_combined_" + part_to_be_milled.name)
        #assembly_work_copy.show()
        return self.data["cost"], assembly_work_copy"""

    def scene(self, assembly: Assembly = None):
        translation_matrix_tool = trimesh.transformations.translation_matrix(np.array([0, 0, self.data["mill_diameter"] * 5]))
        rotation_matrix_tool = trimesh.geometry.align_vectors([0, 0, 1], self.data["direction"] / np.linalg.norm(self.data["direction"]))
        transformation_matrix_tool = np.dot(np.dot(trimesh.transformations.translation_matrix(np.array(self.data["path"][-1])),
                                                   rotation_matrix_tool), translation_matrix_tool)
        tool = trimesh.creation.cylinder(radius=self.data["mill_diameter"] / 2, height=self.data["mill_diameter"] * 10,
                                         transform=transformation_matrix_tool, sections=64)
        tool.visual.vertex_colors = [200, 200, 100, 255]
        parts_to_show = [ tool, self.data["part_not_milled"].mesh, self.data["part_milled"].mesh]
        if self.data["resulting_mesh"] is not None:
            parts_to_show.append(self.data["resulting_mesh"])
        if assembly is not None:
            work_copy_assembly = copy.deepcopy(assembly)
            work_copy_assembly.remove_part(work_copy_assembly.get_part_by_name(self.data["connection"].part_name1))
            work_copy_assembly.remove_part(work_copy_assembly.get_part_by_name(self.data["connection"].part_name2))
            for part in work_copy_assembly.parts:
                parts_to_show.append(part.mesh)
        for part in parts_to_show:
            part.visual.vertex_colors = [128, 128, 128, 180]
        # Visualization
        """line = trimesh.load_path(self.data["path"])
        line.colors = [[255, 0, 0, 255] for _ in range(len(line.entities))]
        parts_to_show.append(line)"""
        points = self.data["path"]
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            # Create a cylinder with trimesh
            cylinder = trimesh.creation.cylinder(radius=0.5,
                                                 height=segment_length,
                                                 sections=5,
                                                 segment=[start_point, end_point])
            # Set the color of the cylinder
            #cylinder.visual.vertex_colors = [178/255, 55/255, 44/255, 255/255]
            cylinder.visual.vertex_colors = [255,0,0,255]
            parts_to_show.append(cylinder)
        print(f"visualizing")
        return trimesh.Scene(parts_to_show)
    @staticmethod
    def static_apply_to_assembly(operation, assembly: Assembly, cost_milling):#TODO large part of this code is duplicate to the actual planning (i.e. checking screw connections) --> NOT A GOOD SOLUTION
        #TODO find better solution than replacing "part_milled" --> currently necessary for sucessfull feasability check
        t_part = operation.data["part_milled"]
        if not operation.data["inverted"]:
            operation.data["part_milled"] = assembly.get_part_by_name(operation.data["connection"].part_name2)
        else:
            operation.data["part_milled"] = assembly.get_part_by_name(operation.data["connection"].part_name1)
        if not (operation.check_feasability(assembly)):
            operation.data["part_milled"] = t_part
            return -1, None
        operation.data["part_milled"] = t_part
        operation.data["cost"] = cost_milling
        assembly_work_copy = copy.deepcopy(assembly)


        if not operation.data["inverted"]:
            part_to_be_milled = assembly_work_copy.get_part_by_name(operation.data["connection"].part_name2)
            other_part = assembly_work_copy.get_part_by_name(operation.data["connection"].part_name1)
        else:
            part_to_be_milled = assembly_work_copy.get_part_by_name(operation.data["connection"].part_name1)
            other_part = assembly_work_copy.get_part_by_name(operation.data["connection"].part_name2)
        part_to_be_milled.mesh = operation.data["part_milled"].mesh
        print(f"applying existing operation: {other_part.name} {part_to_be_milled.name}")
        connections = assembly_work_copy.get_connections_by_part_name(part_to_be_milled.name)
        sc_to_recheck = [conn for conn in connections if conn.connection_type == 'screw']

        if operation.data["resulting_mesh"] is not None:
            new_part_name = part_to_be_milled.name + "_milled_" + other_part.name.split("_milled_")[0] +"_2" #TODO: discarding everything after _milled_ might not be the best solution
            if any(p.name == new_part_name for p in assembly_work_copy.parts):
                assembly_work_copy.remove_part(assembly_work_copy.get_part_by_name(new_part_name))
            #save_path_mesh = assembly.name + "\\" + new_part_name + ".stl"
            #print(save_path_mesh)
            #resulting_mesh1.export(save_path_mesh)
            new_part = Part(new_part_name, "", mesh=operation.data["resulting_mesh"])
            assembly_work_copy.add_part(new_part)
            for part in assembly_work_copy.parts.values():
                if part != new_part:
                    assembly_work_copy.check_pair_for_contact_new([new_part, part])

        assembly_work_copy.remove_connection(operation.data["connection"])
        if part_to_be_milled.mesh is None:
            assembly_work_copy.remove_part(part_to_be_milled)
        else:
            assembly_work_copy.rename_single_part(part_to_be_milled, part_to_be_milled.name + "_milled_" +
                                                  other_part.name.split("_milled_")[0])
            #assembly_work_copy.check_pair_for_contact([other_part, part_to_be_milled])

            parts_to_recheck = [other_part]
            existing_connections = assembly_work_copy.get_connections_by_part_name(part_to_be_milled.name)
            for ex_con in existing_connections:
                if ex_con.connection_type == "geometric":
                    if ex_con.part_name1 == part_to_be_milled.name:
                        parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name2))
                    else:
                        parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name1))
                    assembly_work_copy.remove_connection(ex_con)
            for part in parts_to_recheck:
                assembly_work_copy.check_pair_for_contact_new([part_to_be_milled, part])


        for sc in sc_to_recheck:
            connections = assembly_work_copy.get_connections_by_part_name(sc.part_name2)
            gc_part = [conn for conn in connections if conn.connection_type == 'geometric']
            if sc.part_name1 == part_to_be_milled.name and operation.data[
                "resulting_mesh"] is not None:  # milled part was the screw
                connections = assembly_work_copy.get_connections_by_part_name(new_part.name)
                gc_new_part = [conn for conn in connections if conn.connection_type == 'geometric']
                common_elements = [item for item in gc_new_part if item in gc_part]
                if len(common_elements) > 0:  # there is a geometric connection beteween the new part and the part the screw was screwed into
                    sc.part_name1 = new_part.name
            else:
                connections = assembly_work_copy.get_connections_by_part_name(sc.part_name1)
                gc_screw = [conn for conn in connections if conn.connection_type == 'geometric']
                common_elements = [item for item in gc_screw if item in gc_part]
                if len(common_elements) == 0 and operation.data["resulting_mesh"] is not None:
                    sc.part_name2 = new_part.name  # TODO: maybe check that there actually is a connection

        #assembly_work_copy.show()
        assembly_work_copy.check_geometric_connections()
        return operation.data["cost"], assembly_work_copy

class ScrewingOperation(Operation):
    def __init__(self, parameter: dict):
        super().__init__(parameter)
        self.data["type"] = "screwing"

    def __eq__(self, other):
        return isinstance(other, ScrewingOperation) and self.data.get("connection") == other.data.get("connection")

    def __str__(self):
        conn = self.data.get("connection", None)
        screw_name = getattr(conn, "part_name1", self.data.get("screw_part_name", "?"))
        return f'{self.data["type"]} {screw_name}'

    def compare(self, connection: Screw_Connection):
        return self.data.get("connection") == connection

    def check_feasability(self, assembly):
        return True

    def apply_to_assembly(self, assembly):
        pass

    def _screw_name(self) -> str | None:
        conn = self.data.get("connection", None)
        name = getattr(conn, "part_name1", None)
        if isinstance(name, str):
            return name
        name = self.data.get("screw_part_name", None)
        return name if isinstance(name, str) else None

    def scene_o3d(self, assembly: Assembly | None = None):
        geoms: list[o3d.geometry.Geometry] = []
        if assembly is None:
            return geoms

        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            return geoms

        screw_name = self._screw_name()
        if not isinstance(screw_name, str):
            return geoms

        geoms.extend(o3d_geoms_from_assembly_context(assembly, exclude_names={screw_name}))
        screw_part = assembly.get_part_by_name(screw_name)
        if screw_part is None or not isinstance(screw_part.mesh, o3d.geometry.TriangleMesh):
            return geoms

        grasp_pos = path[1] if path.shape[0] >= 2 else path[0]
        delta = path[-1] - grasp_pos
        moving = o3d_copy_mesh_for_display(screw_part.mesh, color=VISUAL_THEME["moving_part"])
        moving.translate(delta)
        geoms.append(moving)

        direction = np.asarray(self.data.get("direction", np.zeros(3)), dtype=np.float64)
        dn = np.linalg.norm(direction)
        if dn > 1e-12:
            direction = direction / dn
            pullout_distance = float(np.linalg.norm(path[-1] - grasp_pos))
            dir_len = choose_direction_render_length(
                part_mesh=screw_part.mesh,
                assembly=assembly,
                pullout_distance=pullout_distance,
                exclude_names={screw_name},
            )
            marker_radius = max(0.08 * dir_len, 1e-3)
            geoms.append(o3d_lineset_from_direction(path[-1], direction, dir_len))
            geoms.append(o3d_sphere(path[-1], radius=marker_radius))
        return geoms

    def animation_frames_o3d(self, assembly: Assembly, num_frames: int = 60) -> list[list[o3d.geometry.Geometry]]:
        frames: list[list[o3d.geometry.Geometry]] = []
        if assembly is None:
            return frames

        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            return frames

        screw_name = self._screw_name()
        if not isinstance(screw_name, str):
            return frames

        screw_part = assembly.get_part_by_name(screw_name)
        if screw_part is None or not isinstance(screw_part.mesh, o3d.geometry.TriangleMesh):
            return frames

        direction = np.asarray(self.data.get("direction", np.zeros(3)), dtype=np.float64)
        dn = np.linalg.norm(direction)
        if dn < 1e-12:
            return frames
        direction = direction / dn

        static_geoms = o3d_geoms_from_assembly_context(assembly, exclude_names={screw_name})
        grasp_pos = path[1] if path.shape[0] >= 2 else path[0]
        pullout_distance = float(np.linalg.norm(path[-1] - grasp_pos))
        dir_len = choose_direction_render_length(
            part_mesh=screw_part.mesh,
            assembly=assembly,
            pullout_distance=pullout_distance,
            exclude_names={screw_name},
        )
        marker_radius = max(0.08 * dir_len, 1e-3)

        if path.shape[0] >= 3:
            n_approach = max(1, num_frames // 3)
            n_pull = max(1, num_frames - n_approach)
            approach_positions = interpolate_polyline(path[:2], num_frames=n_approach)
            pull_positions = interpolate_polyline(path[1:], num_frames=n_pull)
        else:
            approach_positions = []
            pull_positions = interpolate_polyline(path, num_frames=num_frames)

        for _ in approach_positions:
            geoms = [copy.deepcopy(g) for g in static_geoms]
            geoms.append(o3d_copy_mesh_for_display(screw_part.mesh, color=VISUAL_THEME["moving_part"]))
            geoms.append(o3d_lineset_from_direction(grasp_pos, direction, dir_len))
            geoms.append(o3d_sphere(grasp_pos, radius=marker_radius))
            frames.append(geoms)

        for pos in pull_positions:
            geoms = [copy.deepcopy(g) for g in static_geoms]
            delta = np.asarray(pos, dtype=np.float64) - np.asarray(grasp_pos, dtype=np.float64)
            moving = o3d_copy_mesh_for_display(screw_part.mesh, color=VISUAL_THEME["moving_part"])
            moving.translate(delta)
            geoms.append(moving)
            geoms.append(o3d_lineset_from_direction(np.asarray(pos, dtype=np.float64), direction, dir_len))
            geoms.append(o3d_sphere(np.asarray(pos, dtype=np.float64), radius=marker_radius))
            frames.append(geoms)
        return frames

    def show_animation_o3d(self, assembly: Assembly, num_frames: int = 60, frame_dt: float = 0.05, window_name: str = "Screwing animation") -> None:
        frames = self.animation_frames_o3d(assembly, num_frames=num_frames)
        if not frames:
            print("[ScrewingOperation] No animation frames available.")
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        for frame_geoms in frames:
            vis.clear_geometries()
            for g in frame_geoms:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            import time
            time.sleep(frame_dt)
        vis.run()
        vis.destroy_window()

class GrippingAndRemovingOperation(Operation):
    def __init__(self, parameter: dict):
        super().__init__(parameter)
        self.data["type"] = "gripping"

    def __eq__(self, other):
        return isinstance(other, GrippingAndRemovingOperation) and self.data.get("part_name") == other.data.get("part_name")

    def __str__(self):
        return f'{self.data["type"]} {self.data["part_name"]}'

    def compare(self, part_name: str):
        return self.data.get("part_name") == part_name

    def check_feasability(self, assembly):
        """
        Feasibility check for gripping removal.

        The check is intentionally layered:

        1) Short local translation:
           catches immediate collisions close to the current state.

        2) Full-path stepped actual-mesh translation:
           translates the real target mesh along the planned pull-out segment.
           This is the hard path check for hollow / cover-like parts.

        3) Projected-footprint ray bundle:
           conservative swept-corridor diagnostic. It remains a hard check for
           ordinary solid parts, but may be diagnostic-only when
           data["ray_bundle_required"] is False.

        This avoids treating empty interior space of lids, covers, shells, or
        ring-like parts as a filled solid projected footprint.
        """
        part_name = self.data.get("part_name", None)
        direction = np.asarray(self.data.get("direction", np.zeros(3)), dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        mode = str(self.data.get("ray_bundle_sampling_mode", "projected_footprint"))

        self.data["last_feasibility_debug"] = {
            "check": "gripping_feasibility",
            "part_name": part_name,
            "direction_raw": direction.tolist(),
            "direction_norm": norm,
            "local_translation_checked": False,
            "stepped_path_checked": False,
            "path_ray_bundle_checked": False,
            "ray_bundle_mode": mode,
            "ray_bundle_required": bool(self.data.get("ray_bundle_required", True)),
        }

        if not isinstance(part_name, str):
            self.data["last_feasibility_debug"].update({
                "feasible": False,
                "reason": "invalid_part_name",
            })
            return False

        if norm < 1e-9 or not np.isfinite(norm):
            self.data["last_feasibility_debug"].update({
                "feasible": False,
                "reason": "zero_or_nonfinite_direction",
            })
            raise ValueError("Operation direction vector is zero or non-finite")

        dir_unit = direction / norm

        # ------------------------------------------------------------
        # 1) Local translation feasibility check
        # ------------------------------------------------------------
        delta_trans = float(self.data.get("delta_trans", 5.0))
        translation = dir_unit * delta_trans
        depth_threshold = float(self.data.get("depth_threshold", delta_trans / 2.0))

        try:
            local_feasible, _ = assembly.is_translation_feasible(
                moving_part_name=part_name,
                translation=translation,
                ignore_self=True,
                return_contacts=True,
                depth_threshold=depth_threshold,
            )
            local_feasible = bool(local_feasible)
        except Exception as exc:
            self.data["last_feasibility_debug"].update({
                "local_translation_checked": True,
                "feasible": False,
                "reason": "single_step_translation_exception",
                "error": repr(exc),
                "delta_trans": delta_trans,
                "translation": translation.tolist(),
                "depth_threshold": depth_threshold,
            })
            return False

        self.data["last_feasibility_debug"].update({
            "local_translation_checked": True,
            "local_translation_feasible": bool(local_feasible),
            "delta_trans": delta_trans,
            "translation": translation.tolist(),
            "depth_threshold": depth_threshold,
        })

        if not local_feasible:
            self.data["last_feasibility_debug"].update({
                "feasible": False,
                "reason": "single_step_translation_collision",
            })
            print(
                f"[GrippingAndRemovingOperation] infeasible (single-step): "
                f"moving {part_name} by {delta_trans:.1f} along {dir_unit}"
            )
            return False

        # ------------------------------------------------------------
        # 2) Full-path stepped actual-mesh translation check
        # ------------------------------------------------------------
        stepped_path_feasible = self._check_removal_path_stepped_translation(
            assembly=assembly,
            part_name=part_name,
        )

        if not stepped_path_feasible:
            self.data["last_feasibility_debug"].update({
                "feasible": False,
                "reason": "removal_path_stepped_translation_collision",
            })
            return False

        # ------------------------------------------------------------
        # 3) Projected-footprint ray-bundle check
        # ------------------------------------------------------------
        ray_bundle_required = bool(self.data.get("ray_bundle_required", True))

        path_feasible = self._check_removal_path_projected_footprint_rays(
            assembly=assembly,
            part_name=part_name,
        )

        if not path_feasible and ray_bundle_required:
            self.data["last_feasibility_debug"].update({
                "feasible": False,
                "reason": "removal_path_projected_footprint_hit",
            })
            return False

        if not path_feasible and not ray_bundle_required:
            self.data["last_feasibility_debug"].update({
                "ray_bundle_warning_ignored": True,
                "ray_bundle_warning_reason": "actual_mesh_stepped_path_feasible",
            })
            print(
                f"[GrippingAndRemovingOperation] projected-footprint warning ignored for {part_name}: "
                f"actual-mesh stepped path is feasible."
            )

        self.data["last_feasibility_debug"].update({
            "feasible": True,
            "reason": None,
        })
        return True

    def _check_removal_path_stepped_translation(self, assembly, part_name: str) -> bool:
        """
        Check the full pull-out path by translating the actual target mesh in
        small increments along the planned removal segment.

        Unlike the projected-footprint ray bundle, this does not fill hollow
        interior regions. It is therefore better suited for lids, covers,
        shells, and ring-like components.
        """
        dbg = self.data.setdefault("last_feasibility_debug", {})
        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=float)

        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            dbg.update({
                "stepped_path_checked": False,
                "stepped_path_reason": "invalid_path",
                "path_shape": list(path.shape),
            })
            return False

        grasp_pos = np.asarray(path[1] if path.shape[0] >= 2 else path[0], dtype=float).reshape(3)
        end_pos = np.asarray(path[-1], dtype=float).reshape(3)

        pull_vec = end_pos - grasp_pos
        pull_len = float(np.linalg.norm(pull_vec))

        if pull_len < 1e-9 or not np.isfinite(pull_len):
            dbg.update({
                "stepped_path_checked": False,
                "stepped_path_reason": "zero_or_nonfinite_pullout_length",
                "pull_length": pull_len,
            })
            return False

        pull_dir = pull_vec / pull_len

        step_size = float(self.data.get("stepped_path_step_size", 5.0))
        if step_size <= 1e-9 or not np.isfinite(step_size):
            step_size = 5.0

        depth_threshold = float(
            self.data.get(
                "stepped_path_depth_threshold",
                max(1.0, 0.5 * step_size),
            )
        )

        n_steps = max(1, int(np.ceil(pull_len / step_size)))

        for k in range(1, n_steps + 1):
            dist = min(float(k) * step_size, pull_len)
            translation = pull_dir * dist

            try:
                # return_contacts=True is needed: depth_threshold only takes
                # effect when contacts are requested (AT 282, AT 288).
                feasible, _ = assembly.is_translation_feasible(
                    moving_part_name=part_name,
                    translation=translation,
                    ignore_self=True,
                    return_contacts=True,
                    depth_threshold=depth_threshold,
                )
                feasible = bool(feasible)
            except Exception as exc:
                dbg.update({
                    "stepped_path_checked": True,
                    "stepped_path_feasible": False,
                    "stepped_path_reason": "translation_exception",
                    "stepped_path_error": repr(exc),
                    "stepped_path_step": int(k),
                    "stepped_path_num_steps": int(n_steps),
                    "stepped_path_distance": float(dist),
                    "stepped_path_step_size": float(step_size),
                    "stepped_path_depth_threshold": float(depth_threshold),
                })
                return False

            if not feasible:
                dbg.update({
                    "stepped_path_checked": True,
                    "stepped_path_feasible": False,
                    "stepped_path_reason": "translation_collision",
                    "stepped_path_step": int(k),
                    "stepped_path_num_steps": int(n_steps),
                    "stepped_path_distance": float(dist),
                    "stepped_path_translation": translation.tolist(),
                    "stepped_path_step_size": float(step_size),
                    "stepped_path_depth_threshold": float(depth_threshold),
                })
                print(
                    f"[GrippingAndRemovingOperation] infeasible (stepped actual-mesh path): "
                    f"moving {part_name}, step={k}/{n_steps}, "
                    f"dist={dist:.3f} / {pull_len:.3f}, dir={pull_dir}"
                )
                return False

        dbg.update({
            "stepped_path_checked": True,
            "stepped_path_feasible": True,
            "stepped_path_reason": None,
            "stepped_path_num_steps": int(n_steps),
            "stepped_path_step_size": float(step_size),
            "stepped_path_depth_threshold": float(depth_threshold),
            "stepped_path_pull_length": float(pull_len),
        })
        return True

    def _check_removal_path_projected_footprint_rays(self, assembly, part_name: str) -> bool:
        dbg = self.data.setdefault("last_feasibility_debug", {})
        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "invalid_path", "path_shape": list(path.shape)})
            return False

        part = assembly.get_part_by_name(part_name)
        if part is None or getattr(part, "mesh", None) is None:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "missing_part_or_mesh"})
            return False

        grasp_pos = np.asarray(path[1] if path.shape[0] >= 2 else path[0], dtype=float).reshape(3)
        end_pos = np.asarray(path[-1], dtype=float).reshape(3)
        ray_vec = end_pos - grasp_pos
        ray_len = float(np.linalg.norm(ray_vec))
        if ray_len < 1e-9 or not np.isfinite(ray_len):
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "zero_or_nonfinite_pullout_length", "ray_length": ray_len})
            return False
        ray_dir = ray_vec / ray_len

        vertices = np.asarray(part.mesh.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "invalid_part_vertices"})
            return False

        ray_origins, footprint_debug = self._make_projected_footprint_ray_origins(vertices=vertices, center=grasp_pos, direction=ray_dir)
        if ray_origins.shape[0] == 0:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "no_ray_origins_generated", **footprint_debug})
            return False

        origin_offset = float(self.data.get("ray_origin_offset", 1.0))
        hit_margin = float(self.data.get("ray_hit_margin", 1.0))
        ray_origins = ray_origins + ray_dir.reshape(1, 3) * origin_offset
        effective_ray_len = max(ray_len - origin_offset, 0.0)

        if effective_ray_len <= 1e-9:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "effective_ray_too_short", "ray_length": ray_len, "origin_offset": origin_offset, **footprint_debug})
            return True

        scene, geometry_ids, added = self._build_obstacle_raycast_scene(assembly=assembly, moving_part_name=part_name)
        if added == 0:
            dbg.update({
                "path_ray_bundle_checked": True,
                "path_ray_bundle_reason": "no_obstacle_meshes",
                "ray_length": ray_len,
                "effective_ray_length": effective_ray_len,
                "origin_offset": origin_offset,
                "hit_margin": hit_margin,
                "hit": False,
                "blocking_hit": False,
                **footprint_debug,
            })
            return True

        ray_dirs = np.tile(ray_dir.reshape(1, 3), (ray_origins.shape[0], 1))
        rays = np.concatenate([ray_origins.astype(np.float32), ray_dirs.astype(np.float32)], axis=1)
        try:
            ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        except Exception as exc:
            dbg.update({"path_ray_bundle_checked": False, "path_ray_bundle_reason": "ray_bundle_cast_exception", "error": repr(exc), "num_rays": int(ray_origins.shape[0]), **footprint_debug})
            return False

        t_hit = ans["t_hit"].numpy()
        hit_mask = np.isfinite(t_hit)
        max_allowed_t = max(effective_ray_len - hit_margin, 0.0)
        blocking_mask = hit_mask & (t_hit <= max_allowed_t)
        blocking_hit = bool(np.any(blocking_mask))

        first_blocking_t = None
        first_blocking_ray_index = None
        first_blocking_geometry = None
        if blocking_hit:
            first_blocking_ray_index = int(np.flatnonzero(blocking_mask)[0])
            first_blocking_t = float(t_hit[first_blocking_ray_index])
            try:
                gid = int(ans["geometry_ids"].numpy()[first_blocking_ray_index])
                first_blocking_geometry = geometry_ids.get(gid, str(gid))
            except Exception:
                first_blocking_geometry = None

        dbg.update({
            "path_ray_bundle_checked": True,
            "path_ray_bundle_reason": None if not blocking_hit else "blocking_hit",
            "ray_length": float(ray_len),
            "effective_ray_length": float(effective_ray_len),
            "origin_offset": float(origin_offset),
            "hit_margin": float(hit_margin),
            "num_hits": int(np.sum(hit_mask)),
            "num_blocking_hits": int(np.sum(blocking_mask)),
            "hit": bool(np.any(hit_mask)),
            "blocking_hit": bool(blocking_hit),
            "first_blocking_t": first_blocking_t,
            "first_blocking_ray_index": first_blocking_ray_index,
            "first_blocking_geometry": first_blocking_geometry,
            **footprint_debug,
        })

        if blocking_hit:
            print(
                f"[GrippingAndRemovingOperation] infeasible (projected-footprint ray bundle): "
                f"moving {part_name}, rays={ray_origins.shape[0]}, "
                f"hit={first_blocking_geometry} at t={first_blocking_t:.3f} / {effective_ray_len:.3f}"
            )
            return False
        return True

    def _make_projected_footprint_ray_origins(self, *, vertices: np.ndarray, center: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, dict]:
        vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
        center = np.asarray(center, dtype=float).reshape(3)
        direction = np.asarray(direction, dtype=float).reshape(3)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        u, v = self._orthonormal_basis_perpendicular_to(direction)
        rel = vertices - center.reshape(1, 3)
        pts2 = np.column_stack([rel @ u, rel @ v])
        mode = str(self.data.get("ray_bundle_sampling_mode", "projected_footprint"))
        max_rays = int(self.data.get("ray_bundle_max_rays", 200))

        if pts2.shape[0] == 0:
            return np.zeros((0, 3), dtype=float), {"footprint_reason": "empty_projected_vertices", "ray_bundle_mode": mode}

        min_xy = np.min(pts2, axis=0)
        max_xy = np.max(pts2, axis=0)
        extent = max_xy - min_xy
        footprint_diag = float(np.linalg.norm(extent))

        if mode in {"boundary_vertices", "vertices_only", "hollow_boundary"}:
            origins2 = self._downsample_2d_points_with_center(
                pts2=pts2,
                center2=np.array([0.0, 0.0], dtype=float),
                max_points=max_rays,
                include_center=False,
            )
            origins3 = center.reshape(1, 3) + origins2[:, 0:1] * u.reshape(1, 3) + origins2[:, 1:2] * v.reshape(1, 3)
            return origins3, {
                "footprint_reason": "boundary_vertices_sampling",
                "ray_bundle_mode": mode,
                "footprint_diag": footprint_diag,
                "footprint_extent": extent.tolist(),
                "max_rays": max_rays,
                "num_projected_vertices": int(pts2.shape[0]),
                "num_rays_before_cap": int(pts2.shape[0]),
                "num_rays": int(origins3.shape[0]),
            }

        if footprint_diag <= 1e-9 or not np.isfinite(footprint_diag):
            return center.reshape(1, 3), {
                "footprint_reason": "degenerate_projected_footprint_center_only",
                "ray_bundle_mode": mode,
                "footprint_diag": footprint_diag,
                "num_projected_vertices": int(pts2.shape[0]),
                "num_rays_before_cap": 1,
                "num_rays": 1,
            }

        spacing = float(self.data.get("ray_bundle_spacing", max(2.0, min(8.0, 0.08 * footprint_diag))))
        if spacing <= 1e-9 or not np.isfinite(spacing):
            spacing = max(2.0, min(8.0, 0.08 * footprint_diag))

        hull = self._convex_hull_2d(pts2)
        if hull.shape[0] < 3:
            origins2 = self._downsample_2d_points_with_center(
                pts2=pts2,
                center2=np.array([0.0, 0.0], dtype=float),
                max_points=max_rays,
            )
            origins3 = center.reshape(1, 3) + origins2[:, 0:1] * u.reshape(1, 3) + origins2[:, 1:2] * v.reshape(1, 3)
            return origins3, {
                "footprint_reason": "degenerate_hull_vertex_sampling",
                "ray_bundle_mode": mode,
                "footprint_diag": footprint_diag,
                "footprint_extent": extent.tolist(),
                "ray_spacing": spacing,
                "max_rays": max_rays,
                "num_projected_vertices": int(pts2.shape[0]),
                "num_hull_vertices": int(hull.shape[0]),
                "num_rays_before_cap": int(origins2.shape[0]),
                "num_rays": int(origins3.shape[0]),
            }

        xs = np.arange(min_xy[0], max_xy[0] + 0.5 * spacing, spacing, dtype=float)
        ys = np.arange(min_xy[1], max_xy[1] + 0.5 * spacing, spacing, dtype=float)
        grid = []
        for xx in xs:
            for yy in ys:
                p = np.array([xx, yy], dtype=float)
                if self._point_in_convex_polygon_2d(p, hull):
                    grid.append(p)
        origins2 = np.asarray(grid, dtype=float) if grid else np.zeros((0, 2), dtype=float)
        origins2 = np.vstack([origins2, np.array([[0.0, 0.0]], dtype=float)])
        vertex_budget = min(32, max(8, max_rays // 5)) if max_rays > 0 else 32
        vertex_samples = self._downsample_2d_points_with_center(
            pts2=pts2,
            center2=np.array([0.0, 0.0], dtype=float),
            max_points=vertex_budget,
            include_center=False,
        )
        if vertex_samples.shape[0] > 0:
            origins2 = np.vstack([origins2, vertex_samples])
        if origins2.shape[0] > 1:
            rounded = np.round(origins2, decimals=6)
            _, idx = np.unique(rounded, axis=0, return_index=True)
            origins2 = origins2[np.sort(idx)]
        num_before_cap = int(origins2.shape[0])
        if max_rays is not None and max_rays > 0 and origins2.shape[0] > max_rays:
            center2 = np.array([0.0, 0.0], dtype=float)
            d_center = np.linalg.norm(origins2 - center2.reshape(1, 2), axis=1)
            center_idx = int(np.argmin(d_center))
            other_idx = [i for i in range(origins2.shape[0]) if i != center_idx]
            keep_n_other = max(0, max_rays - 1)
            if keep_n_other > 0 and len(other_idx) > 0:
                sampled_pos = np.linspace(0, len(other_idx) - 1, keep_n_other).round().astype(int)
                keep_idx = [center_idx] + [other_idx[int(j)] for j in sampled_pos]
            else:
                keep_idx = [center_idx]
            origins2 = origins2[np.asarray(sorted(set(keep_idx)), dtype=int)]
        origins3 = center.reshape(1, 3) + origins2[:, 0:1] * u.reshape(1, 3) + origins2[:, 1:2] * v.reshape(1, 3)
        return origins3, {
            "footprint_reason": "convex_hull_grid_sampling",
            "ray_bundle_mode": mode,
            "footprint_diag": footprint_diag,
            "footprint_extent": extent.tolist(),
            "ray_spacing": spacing,
            "max_rays": max_rays,
            "num_projected_vertices": int(pts2.shape[0]),
            "num_hull_vertices": int(hull.shape[0]),
            "num_rays_before_cap": int(num_before_cap),
            "num_rays": int(origins3.shape[0]),
        }

    def _convex_hull_2d(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape((-1, 2))
        if pts.shape[0] <= 1:
            return pts.copy()
        pts = np.unique(np.round(pts, decimals=9), axis=0)
        if pts.shape[0] <= 2:
            return pts.copy()
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)
        return np.asarray(lower[:-1] + upper[:-1], dtype=float)

    def _point_in_convex_polygon_2d(self, point: np.ndarray, polygon: np.ndarray, eps: float = 1e-9) -> bool:
        p = np.asarray(point, dtype=float).reshape(2)
        poly = np.asarray(polygon, dtype=float).reshape((-1, 2))
        if poly.shape[0] < 3:
            return False
        signs = []
        for i in range(poly.shape[0]):
            a = poly[i]
            b = poly[(i + 1) % poly.shape[0]]
            edge = b - a
            rel = p - a
            cross = float(edge[0] * rel[1] - edge[1] * rel[0])
            signs.append(0.0 if abs(cross) <= eps else np.sign(cross))
        nonzero = [s for s in signs if s != 0.0]
        if not nonzero:
            return True
        return all(s == nonzero[0] for s in nonzero)

    def _downsample_2d_points_with_center(self, *, pts2: np.ndarray, center2: np.ndarray, max_points: int, include_center: bool = True) -> np.ndarray:
        pts2 = np.asarray(pts2, dtype=float).reshape((-1, 2))
        center2 = np.asarray(center2, dtype=float).reshape(2)
        if max_points <= 0:
            return np.zeros((0, 2), dtype=float)
        if pts2.shape[0] == 0:
            return center2.reshape(1, 2) if include_center else np.zeros((0, 2), dtype=float)
        rel = pts2 - center2.reshape(1, 2)
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        radii = np.linalg.norm(rel, axis=1)
        order = np.lexsort((-radii, angles))
        pts_sorted = pts2[order]
        rounded = np.round(pts_sorted, decimals=6)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        pts_sorted = pts_sorted[np.sort(idx)]
        budget = max_points - 1 if include_center else max_points
        budget = max(0, budget)
        if pts_sorted.shape[0] > budget and budget > 0:
            idx = np.linspace(0, pts_sorted.shape[0] - 1, budget).round().astype(int)
            pts_sorted = pts_sorted[idx]
        elif budget == 0:
            pts_sorted = np.zeros((0, 2), dtype=float)
        if include_center:
            pts_sorted = np.vstack([center2.reshape(1, 2), pts_sorted])
        return pts_sorted

    def _orthonormal_basis_perpendicular_to(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(direction, dtype=float).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12 or not np.isfinite(n):
            raise ValueError("Cannot build perpendicular basis from zero/non-finite direction")
        d = d / n
        helper = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(np.dot(d, np.array([1.0, 0.0, 0.0])))) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(d, helper)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(d, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def _build_obstacle_raycast_scene(self, assembly, moving_part_name: str):
        scene = o3d.t.geometry.RaycastingScene()
        geometry_ids: dict[int, str] = {}
        added = 0
        for p in getattr(assembly, "parts", {}).values():
            name = getattr(p, "name", None)
            if name == moving_part_name:
                continue
            mesh = getattr(p, "mesh", None)
            if mesh is None or not isinstance(mesh, o3d.geometry.TriangleMesh):
                continue
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                continue
            try:
                gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                geometry_ids[int(gid)] = str(name)
                added += 1
            except Exception:
                continue
        return scene, geometry_ids, added

    def apply_to_assembly(self, assembly):
        pass

    def scene_o3d(self, assembly: Assembly | None = None):
        geoms: list[o3d.geometry.Geometry] = []
        if assembly is None:
            return geoms
        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            return geoms
        part_name = self.data.get("part_name", None)
        if not isinstance(part_name, str):
            return geoms
        geoms.extend(o3d_geoms_from_assembly_context(assembly, exclude_names={part_name}))
        part = assembly.get_part_by_name(part_name)
        if part is not None and isinstance(part.mesh, o3d.geometry.TriangleMesh):
            grasp_pos = path[1] if path.shape[0] >= 2 else path[0]
            delta = path[-1] - grasp_pos
            moving = o3d_copy_mesh_for_display(part.mesh, color=VISUAL_THEME["moving_part"])
            moving.translate(delta)
            geoms.append(moving)
            direction = np.asarray(self.data.get("render_direction", self.data.get("direction", np.zeros(3))), dtype=np.float64)
            dn = np.linalg.norm(direction)
            if dn > 1e-12:
                direction = direction / dn
                current_pos = path[-1]
                pullout_distance = float(np.linalg.norm(path[-1] - grasp_pos))
                dir_len = choose_direction_render_length(part_mesh=part.mesh, assembly=assembly, pullout_distance=pullout_distance, exclude_names={part_name})
                marker_radius = max(0.08 * dir_len, 1e-3)
                geoms.append(o3d_lineset_from_direction(current_pos, direction, dir_len))
                geoms.append(o3d_sphere(current_pos, radius=marker_radius))
        return geoms

    def animation_frames_o3d(self, assembly: Assembly, num_frames: int = 60) -> list[list[o3d.geometry.Geometry]]:
        frames: list[list[o3d.geometry.Geometry]] = []
        if assembly is None:
            return frames
        path = np.asarray(self.data.get("path", np.zeros((0, 3))), dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            return frames
        part_name = self.data.get("part_name", None)
        if not isinstance(part_name, str):
            return frames
        part = assembly.get_part_by_name(part_name)
        if part is None or not isinstance(part.mesh, o3d.geometry.TriangleMesh):
            return frames
        direction = np.asarray(self.data.get("render_direction", self.data.get("direction", np.zeros(3))), dtype=np.float64)
        dn = np.linalg.norm(direction)
        if dn < 1e-12:
            return frames
        direction = direction / dn
        static_geoms = o3d_geoms_from_assembly_context(assembly, exclude_names={part_name})
        grasp_pos = path[1] if path.shape[0] >= 2 else path[0]
        pullout_distance = float(np.linalg.norm(path[-1] - grasp_pos))
        dir_len = choose_direction_render_length(part_mesh=part.mesh, assembly=assembly, pullout_distance=pullout_distance, exclude_names={part_name})
        marker_radius = max(0.08 * dir_len, 1e-3)
        if path.shape[0] >= 3:
            n_approach = max(1, num_frames // 3)
            n_pull = max(1, num_frames - n_approach)
            approach_positions = interpolate_polyline(path[:2], num_frames=n_approach)
        else:
            n_pull = num_frames
            approach_positions = []
        pull_positions = [np.asarray(grasp_pos, dtype=np.float64) + direction * s for s in np.linspace(0.0, pullout_distance, n_pull)]

        for _ in approach_positions:
            geoms = [copy.deepcopy(g) for g in static_geoms]
            geoms.append(o3d_copy_mesh_for_display(part.mesh, color=VISUAL_THEME["moving_part"]))
            geoms.append(o3d_lineset_from_direction(grasp_pos, direction, dir_len))
            geoms.append(o3d_sphere(grasp_pos, radius=marker_radius))
            frames.append(geoms)

        for pos in pull_positions:
            geoms = [copy.deepcopy(g) for g in static_geoms]
            delta = np.asarray(pos, dtype=np.float64) - np.asarray(grasp_pos, dtype=np.float64)
            moving = o3d_copy_mesh_for_display(part.mesh, color=VISUAL_THEME["moving_part"])
            moving.translate(delta)
            geoms.append(moving)
            geoms.append(o3d_lineset_from_direction(np.asarray(pos, dtype=np.float64), direction, dir_len))
            geoms.append(o3d_sphere(np.asarray(pos, dtype=np.float64), radius=marker_radius))
            frames.append(geoms)
        return frames

    def show_animation_o3d(self, assembly: Assembly, num_frames: int = 60, frame_dt: float = 0.05, window_name: str = "Gripping animation") -> None:
        frames = self.animation_frames_o3d(assembly, num_frames=num_frames)
        if not frames:
            print("[GrippingAndRemovingOperation] No animation frames available.")
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        for frame_geoms in frames:
            vis.clear_geometries()
            for g in frame_geoms:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            import time
            time.sleep(frame_dt)
        vis.run()
        vis.destroy_window()

class GrippingOperation(Operation):
    def __init__(self, parameter: dict):
        """
            operation_parameter = { "part_name": name,
                                    "cost": cost,
                                    "direction": direction,
                                    "path": path})
        """
        super().__init__(parameter)
        self.data["type"] = "gripping"
        if "agent" not in self.data:
            self.data["agent"] = "robot_1"

    def __eq__(self, other):
        if not isinstance(other, GrippingOperation):
            return False
        if (self.data["part_name"] == other.data["part_name"] and
                self.data["type"] == other.data["type"] and
                self.data.get("grasp_id") == other.data.get("grasp_id") and
                self.data.get("agent") == other.data.get("agent")):
            return True
        else:
            return False

    def __str__(self):
        return f'{self.data["type"]} {self.data["part_name"]} (grasp {self.data.get("grasp_id")}) by {self.data.get("agent")}'

    def compare(self, part_name: str):
        if self.data["part_name"] == part_name:
            return True
        else:
            return False
   
    def check_feasability(self, assembly, cell_state: CellState | None = None):
        part_name = self.data.get("part_name")
        if not isinstance(part_name, str) or not part_name:
            return False

        # Teil muss überhaupt im Assembly existieren.
        part_getter = getattr(assembly, "get_part_by_name", None)
        if callable(part_getter):
            try:
                if part_getter(part_name) is None:
                    return False
            except Exception:
                return False
        elif part_name not in getattr(assembly, "parts", {}):
            return False

        # Agent muss frei sein.
        agent_name = self.data.get("agent", "robot_1")
        if cell_state is not None and agent_name in cell_state.holds:
            return False

        # Wenn das Teil bereits von irgendjemandem gehalten wird, kann nicht erneut gegriffen werden.
        if _holder_of(cell_state, part_name) is not None:
            return False

        # Optionaler Richtung-/Pfad-Check, falls semantisch angegeben.
        direction = self.data.get("direction")
        if direction is not None:
            try:
                direction_arr = np.asarray(direction, dtype=float)
                if direction_arr.size == 0 or not np.all(np.isfinite(direction_arr)):
                    return False
            except Exception:
                return False

        return True

    def apply_to_assembly(self, assembly, cell_state: CellState | None = None, **kwargs):
        assembly_work_copy = copy.deepcopy(assembly)
        target_name = self.data.get("part_name")
        agent_name = self.data.get("agent", "robot_1")

        if not isinstance(target_name, str) or not target_name:
            return -1, assembly

        part_getter = getattr(assembly_work_copy, "get_part_by_name", None)
        if callable(part_getter):
            try:
                if part_getter(target_name) is None:
                    return -1, assembly
            except Exception:
                return -1, assembly
        elif target_name not in getattr(assembly_work_copy, "parts", {}):
            return -1, assembly

        if _holder_of(cell_state, target_name) is not None:
            return -1, assembly

        last = dict(getattr(assembly_work_copy, 'last_regrasp', {}))
        last.pop(agent_name, None)
        assembly_work_copy.last_regrasp = last

        return self.data.get("cost", 1.0), assembly_work_copy

    def apply_to_cell_state(self, cell_state: CellState) -> CellState:
        """The robot now holds the part; the hold is cell state, not product state."""
        holds = dict(cell_state.holds)
        holds[self.data.get("agent", "robot_1")] = self.data["part_name"]
        return dataclasses.replace(cell_state, holds=holds)

    def scene(self, assembly: Assembly):
        mesh_list = []
        for p in assembly.parts.values():
            if p.mesh:
                m = copy.deepcopy(p.mesh)
                m.transform(p.get_global_pose()) 
                mesh_list.append(m)

        target_pose = None
        if "pose_world" in self.data:
            target_pose = self.data["pose_world"]
        elif "l_pose_m" in self.data:
            part_obj = assembly.get_part_by_name(self.data["part_name"])
            if part_obj:
                l_pose = self.data["l_pose_m"].copy()
                
                
                if np.max(np.abs(l_pose[:3, 3])) < 5.0: # Wenn kleiner als 5mm, sind es wohl Meter
                    l_pose[:3, 3] *= 1000.0
                
                target_pose = part_obj.get_global_pose() @ l_pose

        if target_pose is not None:
            g_type = self.data.get("gripper_type", "gripper")
            tcp_offset = self.data.get("tcp_offset_mm", 294.0)
            if g_type == "gripper_TG":
                m_body, m_f_l, m_f_r = GripperAssemblyTG.get_tg_meshes()
            elif g_type == "gripper_GG":
                m_body, m_f_l, m_f_r = GripperAssemblyGG.get_gg_meshes()
            else:
                m_body, m_f_l, m_f_r = GripperAssembly.get_egu70_meshes()
            gripper_cls = _GRIPPER_SCENE_MAP.get(g_type, GripperAssembly)
            width = self.data.get("opening_width", 50.0)

            mb_mat, mfl_mat, mfr_mat = gripper_cls.get_kinematics(target_pose, width, tcp_offset_mm=tcp_offset)

            m_body.transform(mb_mat)
            m_f_l.transform(mfl_mat)
            m_f_r.transform(mfr_mat)

            m_body.paint_uniform_color([0.2, 0.2, 0.2])
            mesh_list.extend([m_body, m_f_l, m_f_r])


        if "path" in self.data:
            path_pts = np.array(self.data["path"])
            if path_pts.size > 0 and np.all(np.isfinite(path_pts)):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(path_pts)
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]]) # Rot (bessere Sichtbarkeit)
                mesh_list.append(line_set)

                dot = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
                dot.paint_uniform_color([1, 0, 0])
                dot.translate(path_pts[0]) 
                mesh_list.append(dot)
                
        return mesh_list


class RemoveOperation(Operation):
    def __init__(self, parameter: dict):
        """
        parameter = { "part_name": name_des_teils, 
                      "agent": "robot_1",
                      "cost": 2.0 } # Oft teurer, da Bewegung involviert
        """
        super().__init__(parameter)
        self.data["type"] = "removing"
        if "agent" not in self.data:
            raise ValueError(f"FEHLER: RemoveOperation für {self.data.get('part_name')} ohne Agent erstellt!")

    def __str__(self):
        return f'removing {self.data["part_name"]} from assembly'

    def __eq__(self, other):
        if not isinstance(other, RemoveOperation):
            return False
        return (self.data["part_name"] == other.data["part_name"] and 
                self.data["agent"] == other.data["agent"])

    def check_feasability(self, assembly, cell_state: CellState | None = None):
        part_name = self.data["part_name"]
        agent_name = self.data["agent"]

        holder = _holder_of(cell_state, part_name)
        if holder is not None and holder != agent_name:
            return False

        for conn in getattr(assembly, "connections", []):
            if getattr(conn, "connection_type", None) == 'screw':
                if conn.part_name1 == part_name or conn.part_name2 == part_name:
                    part = assembly.get_part_by_name(conn.part_name1)
                    if part is not None:
                        return False

        my_part = None
        getter = getattr(assembly, "get_part_by_name", None)
        if callable(getter):
            try:
                my_part = getter(part_name)
            except Exception:
                my_part = None
        direction = self.data.get("direction")
        if my_part is None or direction is None:
            return True
        direction = np.array(direction, dtype=float)
        if np.linalg.norm(direction) < 0.001:
            return True
        direction = direction / np.linalg.norm(direction)

        try:
            scene = o3d.t.geometry.RaycastingScene()
            parts_in_scene = 0
            for part in assembly.parts.values():
                if part.name == part_name:
                    continue
                try:
                    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(part.mesh)
                    scene.add_triangles(t_mesh)
                    parts_in_scene += 1
                except Exception:
                    pass

            if parts_in_scene == 0:
                return True

            pcd = my_part.mesh.sample_points_uniformly(number_of_points=500)
            origins = np.asarray(pcd.points) + (direction * 0.002)
            dirs = np.tile(direction, (len(origins), 1))
            rays = np.hstack([origins, dirs]).astype(np.float32)
            rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.float32)
            ans = scene.cast_rays(rays_tensor)
            hit_distances = ans['t_hit'].numpy()

            bbox = my_part.mesh.get_axis_aligned_bounding_box()
            extent = np.max(bbox.get_extent())
            move_distance = max(extent * 2.0, 0.15)

            valid_mask = np.isfinite(hit_distances)
            blocked_ratio = np.sum(valid_mask & (hit_distances < move_distance)) / len(origins)
            return blocked_ratio <= 0.10
        except Exception:
            return True

    def apply_to_assembly(self, assembly, cell_state: CellState | None = None, **kwargs):
        assembly_work_copy = copy.deepcopy(assembly)
        part_name = self.data["part_name"]
        agent_name = self.data["agent"]

        getter = getattr(assembly_work_copy, "get_part_by_name", None)
        part = None
        if callable(getter):
            try:
                part = getter(part_name)
            except Exception:
                part = None

        if part is not None:
            remover = getattr(assembly_work_copy, "remove_part", None)
            if callable(remover):
                try:
                    remover(part)
                except Exception:
                    pass
            parts = getattr(assembly_work_copy, "parts", None)
            if isinstance(parts, dict) and part_name in parts:
                try:
                    del parts[part_name]
                except Exception:
                    pass

        last = dict(getattr(assembly_work_copy, 'last_regrasp', {}))
        last.pop(agent_name, None)
        if hasattr(assembly_work_copy, 'last_regrasp'):
            assembly_work_copy.last_regrasp = last

        return self.data.get("cost", 1.0), assembly_work_copy

    def apply_to_cell_state(self, cell_state: CellState) -> CellState:
        """The removing robot lets go of the part."""
        holds = {r: p for r, p in cell_state.holds.items() if r != self.data["agent"]}
        return dataclasses.replace(cell_state, holds=holds)

    def scene(self, assembly: Assembly):
        # Visualisierung
        mesh_list = []
        parts_iterator = assembly.parts.values()
        part_name = self.data["part_name"]

        for p in parts_iterator:
            if p.mesh:
                m = copy.deepcopy(p.mesh)
                m.transform(p.get_global_pose())
                if p.name == part_name:
                    direction = np.array(self.data.get("direction", [0, 0, 1]), dtype=float)
                    dn = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([0, 0, 1])
                    m.translate(dn * 50.0)
                    m.paint_uniform_color([1.0, 0.0, 0.0]) 
                else:
                    m.paint_uniform_color([0.5, 0.5, 0.5])
                mesh_list.append(m)
        return mesh_list


class RegraspOperation(Operation):
    def __init__(self, parameter: dict):
        """
        Atomic release + re-grip at a different grasp pose on the same part.
        parameter = {
            "part_name": str,
            "agent": str,
            "new_grasp_id": str/int,
            "new_l_pose_m": 4x4 ndarray (object-to-gripper transform),
            "new_opening_width": float (mm),
            "cost": float
        }
        """
        super().__init__(parameter)
        self.data["type"] = "regripping"
        if "agent" not in self.data:
            self.data["agent"] = "robot_1"

    def __str__(self):
        return f'regrasp {self.data["part_name"]} with {self.data["agent"]} -> grip {self.data.get("new_grasp_id")}'

    def __eq__(self, other):
        if not isinstance(other, RegraspOperation):
            return False
        return (self.data["part_name"] == other.data["part_name"] and
                self.data["agent"] == other.data["agent"] and
                self.data.get("new_grasp_id") == other.data.get("new_grasp_id"))

    def check_feasability(self, assembly, cell_state: CellState | None = None):
        part_name = self.data["part_name"]
        agent_name = self.data["agent"]
        holder = _holder_of(cell_state, part_name)
        if holder is not None and holder != agent_name:
            return False

        if getattr(assembly, 'last_regrasp', {}).get(agent_name) == part_name:
            return False

        if cell_state is not None and any(r != agent_name for r in cell_state.holds):
            return False
        return True

    def apply_to_assembly(self, assembly, cell_state: CellState | None = None, **kwargs):
        assembly_work_copy = copy.deepcopy(assembly)
        agent_name = self.data["agent"]
        part_name = self.data["part_name"]

        last = dict(getattr(assembly_work_copy, 'last_regrasp', {}))
        last[agent_name] = part_name
        if hasattr(assembly_work_copy, 'last_regrasp'):
            assembly_work_copy.last_regrasp = last

        return self.data.get("cost", 20.0), assembly_work_copy

    def apply_to_cell_state(self, cell_state: CellState) -> CellState:
        """The robot lets go and takes the part again with the new grasp."""
        holds = dict(cell_state.holds)
        holds[self.data["agent"]] = self.data["part_name"]
        return dataclasses.replace(cell_state, holds=holds)

    def scene(self, assembly):
        mesh_list = []
        for part in assembly.parts.values():
            if part.mesh:
                m = copy.deepcopy(part.mesh)
                m.transform(part.get_global_pose())
                if part.name == self.data.get("part_name"):
                    m.paint_uniform_color([0.2, 0.6, 1.0])
                else:
                    m.paint_uniform_color([0.6, 0.6, 0.6])
                mesh_list.append(m)

        # Show the new grip position
        new_l_pose = self.data.get("new_l_pose_m")
        part_obj = assembly.get_part_by_name(self.data.get("part_name"))
        if new_l_pose is not None and part_obj is not None:
            l_pose = np.array(new_l_pose).copy()
            if np.max(np.abs(l_pose[:3, 3])) < 5.0:
                l_pose[:3, 3] *= 1000.0
            target_pose = part_obj.get_global_pose() @ l_pose
            g_type = self.data.get("gripper_type", "gripper")
            tcp_offset = self.data.get("tcp_offset_mm", 294.0)
            if g_type == "gripper_TG":
                m_body, m_f_l, m_f_r = GripperAssemblyTG.get_tg_meshes()
            elif g_type == "gripper_GG":
                m_body, m_f_l, m_f_r = GripperAssemblyGG.get_gg_meshes()
            else:
                m_body, m_f_l, m_f_r = GripperAssembly.get_egu70_meshes()
            gripper_cls = _GRIPPER_SCENE_MAP.get(g_type, GripperAssembly)
            width = self.data.get("new_opening_width", 50.0)
            mb_mat, mfl_mat, mfr_mat = gripper_cls.get_kinematics(target_pose, width, tcp_offset_mm=tcp_offset)
            m_body.transform(mb_mat)
            m_f_l.transform(mfl_mat)
            m_f_r.transform(mfr_mat)
            m_body.paint_uniform_color([1.0, 0.5, 0.0])  # Orange = neue Greifposition
            mesh_list.extend([m_body, m_f_l, m_f_r])

        return mesh_list


class SafetyRingOperation(Operation):
    def __init__(self, parameter: dict):
        """
            operation_parameter = { "part_name": name,
                                    "cost": cost,
                                    "direction": direction,
                                    "grip_widths": grip_width for every point in path
                                    "path": path,
                                    "orientations": orientation for every point in path
                                    "approach_index": index to point at which approach is finished
                                    })
        """
        super().__init__(parameter)
        self.data["type"] = "SafetyRing"

    def __eq__(self, other):
        if not isinstance(other, SafetyRingOperation):
            return False
        if self.data["part_name"] == other.data["part_name"] and self.data["type"] == other.data["type"]:
            return True
        else:
            return False

    def __str__(self):
        return f'{self.data["type"]} {self.data["part_name"]}'

    def compare(self, part_name: str):
        if self.data["part_name"] == part_name:
            return True
        else:
            return False

    def check_feasability(self, assembly):
        points = self.data["path"]
        path_finger_1 = []
        path_finger_2 = []
        for p, o, gw in zip(points, self.data["orientations"], self.data["grip_widths"]):
            rot = R.from_matrix(o)  # Convert the rotation matrix to a Rotation object
            # Local x-axis unit vector
            local_ax = np.array([1, 0, 0])
            # Rotate the local x-axis vector into world space
            world_direction = rot.apply(local_ax)

            path_finger_1.append(p + gw / 2 * world_direction)
            path_finger_2.append(p - gw / 2 * world_direction)
        for part in assembly.parts.values():
            if part.name != self.data["part_name"]:
                ray_origins = path_finger_1 + path_finger_2
                intersects = list(
                    part.mesh.ray.intersects_location(ray_origins=ray_origins,
                                                      ray_directions=[self.data["direction"] for _ in range(len(ray_origins))])[0])
                if len(intersects) > 0:
                    print("SafetyRingOperation not feasible due to intersection with part: ", part.name)
                    return False

        return True
        """radius = 2.5
        height = 50

        # Translation: shift 10 along x, and half-height up along z
        translation_matrix1 = trimesh.transformations.translation_matrix([self.data["grip_width"]/2, 0, height / 2])
        translation_matrix2 = trimesh.transformations.translation_matrix([-self.data["grip_width"]/2, 0, height / 2])
        # Create two cylinders along the z-axis at origin initially
        cylinder1 = trimesh.creation.cylinder(radius=radius, height=height, transform=translation_matrix1,sections=32)
        cylinder2 = trimesh.creation.cylinder(radius=radius, height=height, transform=translation_matrix2, sections=32)

        # Convert the 3x3 rotation to a 4x4 homogeneous matrix
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = self.data["orientations"][self.data["approach_index"]]
        cylinder1.apply_transform(rotation_matrix_4x4)
        cylinder2.apply_transform(rotation_matrix_4x4)

        cylinder1.apply_translation(self.data["path"][self.data["approach_index"]])
        cylinder2.apply_translation(self.data["path"][self.data["approach_index"]])


        # Convert euler to rotation matrix
        rotation_matrix = R.from_quat(orientation_quat).as_matrix()

        # Local x-axis for displacement (first column of rotation matrix)
        local_x_axis = rotation_matrix[:, 0]

        # Compute displacement vectors
        displacement = (separation / 2.0) * local_x_axis

        # Apply rotation and translation to both cylinders
        cylinder1.apply_transform(trimesh.transformations.rotation_matrix(
            angle=0, direction=[0, 0, 1], point=[0, 0, 0]))  # No rotation needed, just a placeholder
        cylinder2.apply_transform(trimesh.transformations.rotation_matrix(
            angle=0, direction=[0, 0, 1], point=[0, 0, 0]))  # No rotation needed, just a placeholder

        # Rotate cylinders
        cylinder1.apply_transform(np.vstack((np.column_stack((rotation_matrix, [0, 0, 0])), [0, 0, 0, 1])))
        cylinder2.apply_transform(np.vstack((np.column_stack((rotation_matrix, [0, 0, 0])), [0, 0, 0, 1])))

        # Translate cylinders to final positions
        cylinder1.apply_translation(position + displacement)
        cylinder2.apply_translation(position - displacement)

        # Combine both cylinders into a single mesh for visualization or further processing
        combined = trimesh.util.concatenate([cylinder1, cylinder2]) """
        return True

    def apply_to_assembly(self, assembly):
        # Define how to apply screwing to the assembly
        pass

    def scene(self, assembly: Assembly):
        # Create copies of the meshes to avoid modifying the original meshes
        work_copy_assembly = copy.deepcopy(assembly)
        mesh_nontranslated = copy.deepcopy(work_copy_assembly.get_part_by_name(self.data["part_name"]).mesh)
        mesh_nontranslated.visual.vertex_colors = [128, 128, 128, 200]
        # Translate the screw according to the planned path
        work_copy_assembly.get_part_by_name(self.data["part_name"]).mesh.apply_translation(self.data["path"][-1]-self.data["path"][-2])

        # Create a scene
        mesh_list = [mesh_nontranslated]
        for part in work_copy_assembly.parts:
            part.mesh.visual.vertex_colors = [128, 128, 128, 180]
            mesh_list.append(part.mesh)

        points = self.data["path"]
        path_finger_1 = []
        path_finger_2 = []
        for p, o, gw in zip(points, self.data["orientations"], self.data["grip_widths"]):
            rot = R.from_matrix(o)  # Convert the rotation matrix to a Rotation object
            # Local x-axis unit vector
            local_ax = np.array([1, 0, 0])
            # Rotate the local x-axis vector into world space
            world_direction = rot.apply(local_ax)

            path_finger_1.append(p + gw / 2 * world_direction)
            path_finger_2.append(p - gw / 2 * world_direction)
            """sphere = trimesh.creation.icosphere(radius=1.5, subdivisions=1)
            sphere.vertices += p + gw/2 * local_x
            sphere.visual.vertex_colors = [255,0,0,255] # Set the color to red (RGBA format)
            sphere = trimesh.creation.icosphere(radius=1.5, subdivisions=1)
            sphere.vertices += p + gw / 2 * local_x
            sphere.visual.vertex_colors = [255, 0, 0, 255]  # Set the color to red (RGBA format)
            mesh_list.append(sphere)"""
        for p in path_finger_1:
            sphere = trimesh.creation.icosphere(radius=0.75, subdivisions=1)
            sphere.vertices += p
            sphere.visual.vertex_colors = [255, 0, 0, 255]  # Set the color to red (RGBA format)
            mesh_list.append(sphere)
        for p in path_finger_2:
            sphere = trimesh.creation.icosphere(radius=0.75, subdivisions=1)
            sphere.vertices += p
            sphere.visual.vertex_colors = [255, 0, 0, 255]  # Set the color to red (RGBA format)
            mesh_list.append(sphere)
        # Add each segment as a cylinder between consecutive points
        for i in range(len(path_finger_1) - 1):
            start_point = path_finger_1[i]
            end_point = path_finger_1[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            # Create a cylinder with trimesh
            cylinder = trimesh.creation.cylinder(radius=0.5,
                                                 height=segment_length,
                                                 sections=5,
                                                 segment=[start_point, end_point])
            # Set the color of the cylinder
            # cylinder.visual.vertex_colors = [178/255, 55/255, 44/255, 255/255]
            cylinder.visual.vertex_colors = [255, 0, 0, 255]
            mesh_list.append(cylinder)
        for i in range(len(path_finger_2) - 1):
            start_point = path_finger_2[i]
            end_point = path_finger_2[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            # Create a cylinder with trimesh
            cylinder = trimesh.creation.cylinder(radius=0.5,
                                                 height=segment_length,
                                                 sections=5,
                                                 segment=[start_point, end_point])
            # Set the color of the cylinder
            # cylinder.visual.vertex_colors = [178/255, 55/255, 44/255, 255/255]
            cylinder.visual.vertex_colors = [255, 0, 0, 255]
            mesh_list.append(cylinder)
        """for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            # Create a cylinder with trimesh
            cylinder = trimesh.creation.cylinder(radius=0.5,
                                              height=segment_length,
                                              sections=5,
                                              segment=[start_point, end_point])
            # Set the color of the cylinder
            # cylinder.visual.vertex_colors = [178/255, 55/255, 44/255, 255/255]
            cylinder.visual.vertex_colors = [255, 0, 0, 255]
            mesh_list.append(cylinder)"""

        #trimesh.Scene(mesh_list).show()
        return trimesh.Scene(mesh_list)

