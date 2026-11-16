import numpy as np
import copy
import networkx as nx
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part
#from part import Part
from diss_assembly_dataclasses.connection import Safety_Ring_Connection, Screw_Connection, Geometric_Connection, Fixed_Connection
#from connection import Safety_Ring_Connection, Screw_Connection, Geometric_Connection
#from connection import evenly_spaced_points_on_sphere

from crc_disassembly_sequence_generator.utility import generate_helix_path, distance_along_vector, \
                    distance_point_to_line, fit_cylinder, fit_min_max_cylinder, UniqueIDgenerator,\
                    fit_min_max_cylinder_along_axis, fit_minimum_volume_cylinder_along_axis, \
                    generate_spiral_above_origin, generate_double_spiral_above_origin,\
                    rotation_matrix_from_zx, rotation_matrix_from_zy, \
                    calculate_new_chord_length, compute_center_shift

try:
    from utility import check_reachability_o3d_raycast as check_reachability
except ImportError:
    try:
        from utility import check_reachability
    except ImportError:
        check_reachability = None

def _safe_check_reachability(mill_volume, part_mesh, direction, visualize=False) -> bool:
    if check_reachability is None:
        # Demo-safe fallback. The EMO defense rollout does not use milling.
        return True

    try:
        return bool(
            check_reachability(
                mill_volume,
                part_mesh,
                direction,
                visualize=visualize,
            )
        )
    except TypeError:
        return bool(check_reachability(mill_volume, part_mesh, direction))

from crc_disassembly_sequence_generator.graspplanning import find_pertruding_grip_points
from crc_disassembly_sequence_generator.operations import Operation, GrippingAndRemovingOperation, MillingOperation, ScrewingOperation, SafetyRingOperation
import pickle
import trimesh
import open3d as o3d
import os
import shutil
from scipy.spatial.transform import Rotation as R
from crc_disassembly_sequence_generator.colors import get_random_color, wbk_color_dict


import multiprocessing

import json


#from assembly
"""from part import Part
from connection import Connection, Geometric_Connection, Screw_Connection, evenly_spaced_points_on_sphere, \
    Fixed_Connection, Safety_Ring_Connection
import xml.etree.ElementTree as ET
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import math
import itertools
import copy
import trimesh"""


from pathlib import Path
base_path = Path(__file__).parent.parent.parent.resolve()
def resolve_path(relative_path):
    """Resolve a relative path to an absolute path."""
    return (base_path / relative_path).resolve()

def load_cell_settings():
    settings_path = Path(__file__).resolve().parent / 'cell_settings.json'
    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find cell_settings.json at {settings_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in cell_settings.json at {settings_path}")
        return {}

# ______only nexessary for generating graphs:
import matplotlib.pyplot as plt
from collections import deque
import time
import datetime

from networkx.drawing.nx_agraph import graphviz_layout
"""try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or PyDotPlus")"""





def snap_direction_to_axis(direction: np.ndarray) -> np.ndarray:
    """Snap a vector to the nearest world axis (render direction; main-axis candidates)."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = d / n
    axis = int(np.argmax(np.abs(d)))
    out = np.zeros(3, dtype=np.float64)
    out[axis] = 1.0 if d[axis] >= 0.0 else -1.0
    return out


MAIN_AXIS_SNAP_ANGLE_DEG = 15.0


def add_main_axis_candidates(
    free_directions: list[np.ndarray],
    outward: np.ndarray | None = None,
    max_angle_deg: float = MAIN_AXIS_SNAP_ANGLE_DEG,
) -> list[np.ndarray]:
    """Put the main axes next to tilted free directions in front of the free directions.

    The core samples free directions on a sphere, so none is exactly a main axis
    and a part pulled out of a bore never gets the bore axis (AT 288). Every free
    direction within ``max_angle_deg`` of a main axis (plus or minus x, y, z)
    adds that axis. Axis candidates come first, the sampled free directions
    stay as fallbacks in the order they came in (the caller sorts them outward
    first), duplicates are dropped. The axes are ordered by ``outward`` (most
    outward first) when it is given.
    """
    cos_limit = float(np.cos(np.radians(max_angle_deg)))
    axes: list[np.ndarray] = []
    for d in free_directions:
        v = np.asarray(d, dtype=float).reshape(3)
        a = snap_direction_to_axis(v)
        if float(np.dot(v, a)) < cos_limit * float(np.linalg.norm(v)):
            continue
        if not any(float(np.linalg.norm(a - e)) < 1e-6 for e in axes):
            axes.append(a)
    if outward is not None:
        axes.sort(key=lambda a: float(np.dot(a, outward)), reverse=True)
    rest = [
        d for d in free_directions
        if not any(float(np.linalg.norm(np.asarray(d, dtype=float) - a)) < 1e-6 for a in axes)
    ]
    return axes + rest


def _connection_pair_matches(conn, a: str, b: str, connection_type: str | None = None) -> bool:
    if connection_type is not None and getattr(conn, "connection_type", None) != connection_type:
        return False
    p1 = getattr(conn, "part_name1", None)
    p2 = getattr(conn, "part_name2", None)
    return (p1 == a and p2 == b) or (p1 == b and p2 == a)


def _has_fixed_connection(assembly, a: str, b: str) -> bool:
    return any(_connection_pair_matches(conn, a, b, connection_type="fixed") for conn in getattr(assembly, "connections", []))


def _add_fixed_if_missing(assembly, a: str, b: str) -> bool:
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b or a == b:
        return False
    if _has_fixed_connection(assembly, a, b):
        return False
    assembly.add_connection(Fixed_Connection(a, b))
    return True


def _current_part_names(assembly) -> set[str]:
    parts = getattr(assembly, "parts", {})
    if isinstance(parts, dict):
        return {str(name) for name in parts.keys()}
    names: set[str] = set()
    try:
        for p in parts:
            name = getattr(p, "name", None)
            if isinstance(name, str):
                names.add(name)
    except Exception:
        pass
    return names


def _geometric_neighbor_names_in_current_state(assembly, part_name: str) -> list[str]:
    current_parts = _current_part_names(assembly)
    if part_name not in current_parts:
        return []
    neighbors: set[str] = set()
    for conn in getattr(assembly, "connections", []):
        if getattr(conn, "connection_type", None) != "geometric":
            continue
        p1 = getattr(conn, "part_name1", None)
        p2 = getattr(conn, "part_name2", None)
        if p1 == part_name and isinstance(p2, str) and p2 in current_parts:
            neighbors.add(p2)
        elif p2 == part_name and isinstance(p1, str) and p1 in current_parts:
            neighbors.add(p1)
    neighbors.discard(part_name)
    return sorted(neighbors)


def add_fixed_connections_to_current_geometric_neighbors(assembly, part_name: str) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Legacy helper kept for old experiments; the new freeze logic should not call this on the real assembly."""
    current_parts = sorted(_current_part_names(assembly))
    current_geometric_neighbors = _geometric_neighbor_names_in_current_state(assembly, part_name)
    added: list[tuple[str, str]] = []
    for neighbor in current_geometric_neighbors:
        if _add_fixed_if_missing(assembly, part_name, neighbor):
            added.append((part_name, neighbor))
    _clear_assembly_geometry_caches(assembly)
    return added, current_parts, current_geometric_neighbors


def _is_secondary_path_blocked_failure(failure: dict) -> bool:
    return (
        failure.get("reason") == "check_feasability_failed"
        and failure.get("feasibility_reason") == "removal_path_projected_footprint_hit"
        and failure.get("local_translation_feasible") is True
        and failure.get("path_ray_bundle_checked") is True
        and failure.get("path_ray_bundle_reason") == "blocking_hit"
    )


def _is_promising_gripping_and_removing_candidate_failure(failure: dict) -> bool:
    return failure.get("reason") == "check_feasability_failed" and failure.get("local_translation_feasible") is True


def should_lock_part_after_all_secondary_path_failures(candidate_failures: list[dict]) -> bool:
    promising_failures = [f for f in candidate_failures if _is_promising_gripping_and_removing_candidate_failure(f)]
    if not promising_failures:
        return False
    return all(_is_secondary_path_blocked_failure(f) for f in promising_failures)


def _secondary_path_failures_blocked_by_part(candidate_failures: list[dict], blocker_part_name: str) -> bool:
    promising_failures = [f for f in candidate_failures if _is_promising_gripping_and_removing_candidate_failure(f)]
    if not promising_failures:
        return False
    for failure in promising_failures:
        if not _is_secondary_path_blocked_failure(failure):
            return False
        if failure.get("first_blocking_geometry", None) != blocker_part_name:
            return False
    return True


def _clear_assembly_geometry_caches(assembly) -> None:
    for attr, value in (("_fcl_objects", {}), ("_fcl_cache", None)):
        if hasattr(assembly, attr):
            try:
                setattr(assembly, attr, value)
            except Exception:
                pass


def _normalize_vec3(v) -> np.ndarray | None:
    try:
        arr = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(arr))
    except Exception:
        return None
    if n <= 1e-9 or not np.isfinite(n):
        return None
    return arr / n


def _get_direction_freezes(assembly) -> list[dict]:
    freezes = getattr(assembly, "_direction_specific_freezes", None)
    if freezes is None:
        freezes = []
        setattr(assembly, "_direction_specific_freezes", freezes)
    return freezes


def _is_direction_temporarily_frozen(
    assembly,
    *,
    target_part_name: str,
    direction: np.ndarray,
    cos_tol: float = 0.9848,
) -> tuple[bool, dict | None]:
    direction = _normalize_vec3(direction)
    if direction is None:
        return False, None

    active_freezes: list[dict] = []
    matched: dict | None = None

    for fr in list(_get_direction_freezes(assembly)):
        blocker = fr.get("blocker_part_name")
        if isinstance(blocker, str) and blocker:
            try:
                blocker_present = assembly.get_part_by_name(blocker) is not None
            except Exception:
                blocker_present = blocker in getattr(assembly, "parts", {})
            if not blocker_present:
                continue

        active_freezes.append(fr)

        if fr.get("target_part_name") != target_part_name:
            continue
        fr_dir = _normalize_vec3(fr.get("direction"))
        if fr_dir is None:
            continue
        if float(np.dot(direction, fr_dir)) >= cos_tol:
            matched = fr

    setattr(assembly, "_direction_specific_freezes", active_freezes)
    return matched is not None, matched


def _add_direction_specific_freeze(
    assembly,
    *,
    target_part_name: str,
    direction: np.ndarray,
    blocker_part_name: str,
    reason: str,
) -> None:
    direction = _normalize_vec3(direction)
    if direction is None:
        return

    freezes = _get_direction_freezes(assembly)
    for fr in freezes:
        old_dir = _normalize_vec3(fr.get("direction"))
        if (
            fr.get("target_part_name") == target_part_name
            and fr.get("blocker_part_name") == blocker_part_name
            and old_dir is not None
            and float(np.dot(direction, old_dir)) >= 0.9848
        ):
            return

    freezes.append({
        "target_part_name": target_part_name,
        "blocker_part_name": blocker_part_name,
        "direction": direction.tolist(),
        "reason": reason,
    })


def _looks_like_hollow_cover(part_name: str) -> bool:
    name = str(part_name).lower()
    return any(token in name for token in ("lid", "cover", "deckel", "cap"))


def graph_layout(graph):
    """Node positions for plotting: graphviz 'dot' if available, otherwise a layered layout (root on top)."""
    try:
        return graphviz_layout(graph, prog="dot")
    except Exception:  # pygraphviz / Graphviz not installed (common on Windows)
        calculate_max_steps(graph)
        pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
        return {node: (x, -y) for node, (x, y) in pos.items()}


class HierarchicalDisassemblySequencePlanner:
    def __init__(self, assembly: Assembly, cost_milling: float = 10.0, cost_screwing: float=5.0,
                 cost_gripping_and_removing: float=0.0, cost_safety_ring: float=5.0, mill_diameter : float = 6.0, path_to_working_dir = None,
                 available_screwdrivers = ['H5', 'T6']):
        """ Carefull: working_dir is completely erased!!!"""
        self.cost_milling = cost_milling
        self.cost_screwing = cost_screwing
        self.cost_gripping_and_removing = cost_gripping_and_removing
        self.cost_safety_ring = cost_safety_ring
        self.mill_diameter = mill_diameter
        self.available_screwdrivers = available_screwdrivers
        self.id_generator = UniqueIDgenerator()
        self.graph = nx.DiGraph()
        self.assembly_states = {}

        initial_id = self.id_generator.get_id()
        self.unexplored_nodes = [initial_id]
        self.assembly_states[initial_id]=copy.deepcopy(assembly)
        self.graph.add_node(initial_id, cost = 0, cheapest_sequence_to_node = [], explored=False)
        self.calculated_milling_operations = []

        self.defined_gripping_and_removing_operations = []

        if path_to_working_dir is None:
            path_to_working_dir = os.path.dirname(os.path.abspath(__file__))
        self.working_dir = str(path_to_working_dir) + "/Working_dir"
        
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.mkdir(self.working_dir)


    def plot_graph(self, highlights = None):
        if highlights is not None:
            node_color = ["green" if node in highlights else "white" for node in self.graph.nodes()]
        else:
            node_color = ["white" for node in self.graph.nodes()]

        pos = graph_layout(self.graph)
        layers = calculate_max_steps(self.graph)
        #pos = nx.spring_layout(self.graph)
        #pos = nx.multipartite_layout(self.graph, subset_key="layer", align="horizontal")
        # flip y axis
        #for key, values in pos.items():
        #    pos[key] = [-1 * value for value in values]

        #pos = nx.spring_layout(self.graph, pos=pos)
        labels = {}
        for node in self.graph.nodes:
            labels[node] = node
        fig = plt.figure(figsize=(8, 8))
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=6)
        nx.draw_networkx_nodes(self.graph, pos, node_size=100, node_color=node_color, edgecolors='black')
        nx.draw_networkx_edges(self.graph, pos, arrows=True, arrowsize=8)

        #plt.show()
        return fig

    def visualize_sequence(self, sequence, angle = 125.0, distance = 80, file_path = None):
        scene = self.assembly_states[sequence[0]].scene()
        # Create the camera transformation
        translation_matrix = trimesh.transformations.translation_matrix(scene.centroid)
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 180.0 * angle, [1, 0, 0])
        translation_matrix2 = trimesh.transformations.translation_matrix([0, 0, distance])
        transform = np.dot(np.dot(translation_matrix, rotation_matrix), translation_matrix2)
        scene.camera_transform = transform
        if file_path is None:
            scene.show()
        else:
            img = scene.save_image((1920,1080))
            # Save the image to a file
            pathtmp = os.path.join(file_path, f'state_{sequence[0]}.png')
            with open(pathtmp, 'wb') as f:
                f.write(img)
        for i in range(1, len(sequence)):
            scene = self.graph[sequence[i-1]][sequence[i]]["operation"].scene(self.assembly_states[sequence[i-1]])
            scene.camera_transform = transform
            if file_path is None:
                scene.show()
            else:
                img = scene.save_image((1920, 1080))
                # Save the image to a file
                pathtmp = os.path.join(file_path, f'operation_{sequence[i-1]}_{sequence[i]}.png')
                with open(pathtmp, 'wb') as f:
                    f.write(img)
            scene = self.assembly_states[sequence[i]].scene()
            scene.camera_transform = transform
            if file_path is None:
                scene.show()
            else:
                img = scene.save_image((1920, 1080))
                # Save the image to a file
                pathtmp = os.path.join(file_path, f'state_{sequence[i]}.png')
                with open(pathtmp, 'wb') as f:
                    f.write(img)

    def export_sequence(self, sequence, file_path):
        self.assembly_states[sequence[0]].export_full(f'{file_path}/assembly_states/state_{sequence[0]}.xml')
        for i in range(1, len(sequence)):
            self.assembly_states[sequence[i]].export_full(f'{file_path}/assembly_states/state_{sequence[i]}.xml')
            op = self.graph[sequence[i-1]][sequence[i]]["operation"]
            with open(f'{file_path}/sequence.txt', "a") as file:
                file.write(f'{sequence[i-1]}_{sequence[i]} {str(op)}\n')
            with open(f"{file_path}/{sequence[i-1]}_{sequence[i]}_tool_path.txt", "w") as file:
                direction = op.data["direction"]
                for point in op.data["path"]:
                    file.write(f"{point[0]} {point[1]} {point[2]} {direction[0]} {direction[1]} {direction[2]}\n")

    def export_states(self, file_path):
        for i in range(0, len(self.assembly_states)):
            self.assembly_states[i].export_full(f'{file_path}/assembly_states/state_{i}.xml')

    def export_gripping_and_removing_operations(self, file_path: str = "gripping_and_removing_operations.pkl"):
        """Export defined_gripping_and_removing_operations to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.defined_gripping_and_removing_operations, f)
        print(f"Gripping operations exported to {file_path}")

    def import_gripping_and_removing_operations(self, file_path: str = "gripping_and_removing_operations.pkl"):
        """Import defined_gripping_and_removing_operations from a pickle file."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.defined_gripping_and_removing_operations = pickle.load(f)
            print(f"Gripping operations imported from {file_path}")
        else:
            raise FileNotFoundError(f"No such file: {file_path}")

    def save_graph_as_pkl(self, file_name=""):
        labels = {}
        for node in self.graph.nodes:
            labels[node] = node
        dump_graph = copy.deepcopy(self.graph)
        # save graph and labels to file
        with open(file_name, 'wb') as f:
            pickle.dump((dump_graph, labels), f)

    def explore_node(self, node_id, multiprocess = True, reduce_screwing = True):
        grippable = False
        multiprocess_apply = multiprocess
        exploration_results = []
        if self.cost_gripping_and_removing>=0:
            for part in self.assembly_states[node_id].parts.values():
                operation, new_assembly = self.plan_gripping_and_removing(self.assembly_states[node_id], part, self.cost_gripping_and_removing)
                if operation.data["cost"]>=0:
                    exploration_results.append([operation, new_assembly, node_id])
                    #self.update_graph_with_operation(operation, new_assembly, node_id)
                    grippable=True
                   # new_assembly.show()
                else:
                    print(f"part {part.name} not grippable")
        if not grippable and self.cost_screwing>=0:
            #applied_operations = []
            #applied_op_screw_heads = set()
            #cheapest_sequence = self.graph.nodes[node_id]["cheapest_sequence_to_node"]
            """for i in range(len(cheapest_sequence)-1):
                applied_operations.append(self.graph.get_edge_data(cheapest_sequence[i], cheapest_sequence[i+1])["operation"])
            if len(cheapest_sequence)>0:
                applied_operations.append(self.graph.get_edge_data(cheapest_sequence[-1], node_id)["operation"])
            for ao in applied_operations:
                if ao.data["type"]=="screwing":
                    applied_op_screw_heads.add(ao.data["connection"].screw_head)"""
            connections = self.assembly_states[node_id].connections
            sc_connections = [conn for conn in connections if conn.connection_type == 'screw']
            #preferred_sc_op = []
            #other_sc_op = []
            for sc in sc_connections:
                if sc.screw_head in self.available_screwdrivers:
                    operation, new_assembly = self.plan_screwing(self.assembly_states[node_id], sc, self.cost_screwing)
                    if operation.data["cost"]>=0:
                        #if sc.screw_head in applied_op_screw_heads:
                        #    preferred_sc_op.append([operation, new_assembly, node_id])
                        #else:
                        #    other_sc_op.appen([operation, new_assembly, node_id])
                        exploration_results.append([operation, new_assembly, node_id])
                        #self.update_graph_with_operation(operation, new_assembly, node_id)
                       # new_assembly.show()
            """for entry in preferred_sc_op:
                self.update_graph_with_operation(entry[0], entry[1], entry[2])
            if len(preferred_sc_op)==0:
                for entry in other_sc_op:
                    self.update_graph_with_operation(entry[0], entry[1], entry[2])"""
        if not grippable and self.cost_safety_ring>=0:
            connections = self.assembly_states[node_id].connections
            sr_connections = [conn for conn in connections if conn.connection_type == 'safety_ring']
            for src in sr_connections:
                operation, new_assembly = self.plan_safety_ring_disassembly(self.assembly_states[node_id], src, self.cost_safety_ring)
                if operation.data["cost"]>=0:
                    exploration_results.append([operation, new_assembly, node_id])
                    #self.update_graph_with_operation(operation, new_assembly, node_id)

        if not grippable and self.cost_milling>=0:
            datasets = []
            results = []
            datasets_apply = []
            for gc in self.assembly_states[node_id].connections:
                if gc.connection_type == "geometric":
                    found_inverted = False
                    found_non_inverted = False
                    for op in self.calculated_milling_operations:
                        if op.data["connection"]==gc:
                            if op.data["inverted"]:
                                found_inverted=True
                            else:
                                found_non_inverted=True
                            if multiprocess_apply:
                                #print("creating dataset")
                                datasets_apply.append([copy.deepcopy(op), self.assembly_states[node_id], self.cost_milling])
                            else:
                                #apply existing operations without multitrheading
                                t_cost, t_assembly = op.apply_to_assembly(self.assembly_states[node_id], self.cost_milling)
                                if t_cost>=0:
                                    self.update_graph_with_operation(op, t_assembly, node_id) #ToDo change!!

                    if multiprocess:
                        if not found_non_inverted:
                            datasets.append([self.assembly_states[node_id], gc, self.mill_diameter, self.cost_milling, False])
                        if not found_inverted:
                            datasets.append([self.assembly_states[node_id], gc, self.mill_diameter, self.cost_milling, True])
                    else:
                        if not found_non_inverted:
                            results.append(HierarchicalDisassemblySequencePlanner.plan_milling(self.assembly_states[node_id], gc, self.mill_diameter, self.cost_milling, False))
                        if not found_inverted:
                            results.append(HierarchicalDisassemblySequencePlanner.plan_milling(self.assembly_states[node_id], gc, self.mill_diameter, self.cost_milling, True))

            pool = multiprocessing.Pool(processes=32)
            if multiprocess_apply:
                # apply existing operations first:
                results_apply = pool.starmap(
                    MillingOperation.static_apply_to_assembly,
                    datasets_apply)
                print("calculated application results")
                #pool.close()  # Close the pool, preventing new tasks from being submitted
                #pool.join()  # Wait for all worker processes to finish
                for res, data in zip(results_apply, datasets_apply):
                    if isinstance(res, Exception):
                        print(f"Task failed with exception: {res}")
                    elif res[0] >= 0:
                        exploration_results.append([data[0], res[1], node_id])
                        #self.update_graph_with_operation(data[0], res[1], node_id)
            if multiprocess:
                #pool = multiprocessing.Pool(processes=32)
                results = pool.starmap(HierarchicalDisassemblySequencePlanner.plan_milling, datasets)
                #data = datasets[0]
                #results = pool.starmap(HierarchicalDisassemblySequencePlanner.plan_milling,[data])
            pool.close()  # Close the pool, preventing new tasks from being submitted
            pool.join()  # Wait for all worker processes to finish
            print("done joining")
            for res in results:
                if res is not None:
                    if res[0].data["cost"]>=0:
                        #check wether screw was milled
                        #for con in res[1].get_connections_by_part_name(res[0].data["part_milled"].name):
                        #    if con.connection_type == "screw" and con.part_name1 == res[0].data["part_milled"].name:
                        #        con.screw_head='X'
                        #self.update_graph_with_operation(res[0], res[1], node_id)
                        exploration_results.append([res[0], res[1], node_id])
                    if res[0] not in self.calculated_milling_operations:
                        self.calculated_milling_operations.append(res[0])
                        #scene = res[0].scene()
                        #scene.show()
        return exploration_results

        #self.graph.nodes[node_id]["explored"]=True
        #self.unexplored_nodes.remove(node_id)
    def update_graph_with_exploration_results(self, node, results):
        #Filter operations according to set Rules
        applied_operations = []
        applied_op_screw_heads = set()
        planned_op_screw_heads = set()
        cheapest_sequence = self.graph.nodes[node]["cheapest_sequence_to_node"]
        for i in range(len(cheapest_sequence)-1):
            applied_operations.append(self.graph.get_edge_data(cheapest_sequence[i], cheapest_sequence[i+1])["operation"])
        if len(cheapest_sequence)>0:
            applied_operations.append(self.graph.get_edge_data(cheapest_sequence[-1], node)["operation"])
        for ao in applied_operations:
            if ao.data["type"]=="screwing":
                applied_op_screw_heads.add(ao.data["connection"].screw_head)
        for operation_result in results:
            if operation_result[0].data["type"] == "screwing":
                planned_op_screw_heads.add(operation_result[0].data["connection"].screw_head)
            if operation_result[0].data["type"] == "gripping":
                for op in self.defined_gripping_and_removing_operations:
                    if op.data["part_name"] == operation_result[0].data["part_name"] and \
                    np.allclose(op.data["direction"], operation_result[0].data["direction"]):
                        operation_result[0] = copy.deepcopy(op)
                        operation_result[0].data["cost"] = self.cost_gripping_and_removing
                        break

        intersection = planned_op_screw_heads.intersection(applied_op_screw_heads)
        if intersection:
            #print(f"{applied_op_screw_heads} {planned_op_screw_heads}")
            for operation_result in results[:]: #[:] necessary to copy list
                if operation_result[0].data["type"] == "screwing" and \
                   operation_result[0].data["connection"].screw_head not in intersection:
                    results.remove(operation_result)
            #print("done")

        #done filtering

        for operation_result in results:
            if len(operation_result)==3:
                self.update_graph_with_operation(operation_result[0], operation_result[1], operation_result[2])
                operation_result[1].save_part_meshes(self.working_dir)
                operation_result[1].delete_part_meshes()
            else:
                print("cant update, no valid Ooperation")

        self.graph.nodes[node]["explored"] = True
        self.assembly_states[node].save_part_meshes(self.working_dir)
        # self.assembly_states[result[0][0][2]].export_full(f"{working_dir}/as{result[0][0][2]}.xml")
        self.assembly_states[node].delete_part_meshes()
        self.unexplored_nodes.remove(node)



    def update_graph_with_operation(self, operation: Operation, new_assembly: Assembly, node_id):
        already_in_graph = False
        follow_node_id = 0
        for key in self.assembly_states:
            a = self.assembly_states[key]
            if a == new_assembly:
                already_in_graph = True
                follow_node_id = key
                break
        if not already_in_graph:
            follow_node_id = self.id_generator.get_id()
            self.graph.add_node(follow_node_id, cost_to_node=0, cheapest_sequence_to_node=[], explored=False)
            self.assembly_states[follow_node_id] = new_assembly
            self.unexplored_nodes.append(follow_node_id)

        self.graph.add_edge(node_id, follow_node_id, operation=operation)
        self.update_node(follow_node_id)
        print(f"{operation} | {operation.data['cost']} | Nodes in Graph: {len(self.graph.nodes)}")

    def update_node(self, node_id):
        pred_id = self.find_cheapest_predecessor(node_id)
        self.graph.nodes[node_id]["cost"]= self.graph.nodes[pred_id]["cost"] + \
                                           self.graph[pred_id][node_id]["operation"].data["cost"]
        self.graph.nodes[node_id]["cheapest_sequence_to_node"] = self.graph.nodes[pred_id]["cheapest_sequence_to_node"]\
                                                                 + [pred_id]

    def find_cheapest_predecessor(self, node_id):
        predecessors=list(self.graph.predecessors(node_id))
        temp_cost = float("inf")
        cheapest_pred_id = -1
        for i in predecessors:
            if self.graph.nodes[i]["cost"]<temp_cost:
                temp_cost=self.graph.nodes[i]["cost"]
                cheapest_pred_id = i
        return cheapest_pred_id
    def get_cheapest_unexplored_node(self):
        cheapest_node=-1
        cost = float("inf")
        for n in self.unexplored_nodes:
            if self.graph.nodes[n]["cost"]<cost:
                cheapest_node=n
                cost=self.graph.nodes[n]["cost"]
        return cheapest_node
    def get_cheapest_unexplored_nodes(self):
        cheapest_nodes=[]
        cost = float("inf")
        for n in self.unexplored_nodes:
            if self.graph.nodes[n]["cost"]==cost:
                cheapest_nodes.append(n)
            elif self.graph.nodes[n]["cost"]<cost:
                cheapest_nodes=[n]
                cost=self.graph.nodes[n]["cost"]
        return cheapest_nodes
    def get_cheapest_n_unexplored_nodes(self, n=32):
        self.unexplored_nodes.sort(key=lambda node: self.graph.nodes[node]['cost'])
        print(self.unexplored_nodes)
        return self.unexplored_nodes[:min(n, len(self.unexplored_nodes))]

    def generate_full_graph(self, max_iterations=500):
        for _ in range(max_iterations):
            cheapest_node=self.get_cheapest_unexplored_node()
            result = self.explore_node(cheapest_node)
            self.update_graph_with_exploration_results(cheapest_node, result)
            if len(self.unexplored_nodes)==0:
                print("explored all nodes!")
                break
    def explore_all_nodes_with_parts(self, parts_to_be_disassembled: list[Part], max_iterations: int=100):
        #pool = multiprocessing.Pool(processes=32)

        start = time.time()
        for i in range(max_iterations):
            datasets = []
            cheapest_nodes = self.get_cheapest_n_unexplored_nodes(32) #self.get_cheapest_unexplored_nodes()
            found_target = False
            for node in cheapest_nodes:
                found_part = False
                for part in parts_to_be_disassembled:
                    if any(part.soft_compare(p) for p in self.assembly_states[node].parts.values()) or \
                            part in self.assembly_states[node].parts.values():
                        datasets.append([node, self.assembly_states[node], self.cost_gripping_and_removing, self.cost_screwing,
                                         self.cost_milling, self.cost_safety_ring, self.mill_diameter, self.calculated_milling_operations,
                                         self.available_screwdrivers])
                        found_part = True
                        break
                if not found_part:
                    found_target = True
            if len(datasets)==0 or found_target == True:
                return 1
            for ds in datasets:
                print(ds[0])
            print(f"iteration {i} | time:{time.time()-start} | current cost {self.graph.nodes[cheapest_nodes[0]]['cost']}")
            if len(datasets) <=4:
                for dataset in datasets:
                    result=self.explore_node(dataset[0])
                    self.update_graph_with_exploration_results(dataset[0], result)
                    #self.assembly_states[dataset[0]].save_part_meshes(working_dir)#export_full(f"{working_dir}/as{dataset[0]}.xml")
                    #self.assembly_states[dataset[0]].delete_part_meshes()
            else:
                chunk_size = 16

                def chunked_datasets(datasets, chunk_size):
                    """Yield successive chunk_size chunks from datasets."""
                    for i in range(0, len(datasets), chunk_size):
                        yield datasets[i:i + chunk_size]

                for dataset_chunk in chunked_datasets(datasets, chunk_size):
                    with multiprocessing.Pool(processes=16) as pool:
                        results_search = pool.starmap(
                            HierarchicalDisassemblySequencePlanner.static_explore_node,
                            dataset_chunk)
                    print("calculated sth")
                    for result, dataset in zip(results_search, dataset_chunk):
                        #1 for operation_result in result[0]:
                        #1     self.update_graph_with_operation(operation_result[0], operation_result[1], operation_result[2])
                        #1     operation_result[1].save_part_meshes(working_dir)
                        #1     operation_result[1].delete_part_meshes()
                        self.update_graph_with_exploration_results(dataset[0], result[0])
                        for op in result[1]:
                            if op not in self.calculated_milling_operations:
                                self.calculated_milling_operations.append(op)
                        #1 self.graph.nodes[dataset[0]]["explored"] = True
                        #1 self.assembly_states[dataset[0]].save_part_meshes(working_dir)
                        #self.assembly_states[result[0][0][2]].export_full(f"{working_dir}/as{result[0][0][2]}.xml")
                        #1 self.assembly_states[dataset[0]].delete_part_meshes()
                        #1 self.unexplored_nodes.remove(dataset[0])
                #for con in res[1].get_connections_by_part_name(res[0].data["part_milled"].name):
                #    if con.connection_type == "screw" and con.part_name1 == res[0].data["part_milled"].name:
                #        con.screw_head = 'X'
        return -1

    def find_all_sequences_to_target(self, parts_to_be_disassembled: list[Part], source_node=0):
        candidates = []
        paths = []
        for node in self.graph.nodes:
            all_parts_disassembled = True
            for part in parts_to_be_disassembled:
                if any(part.soft_compare(p) for p in self.assembly_states[node].parts.values()) or \
                        part in self.assembly_states[node].parts.values():
                    all_parts_disassembled = False
                    break
            if all_parts_disassembled:
                candidates.append(node)
        if len(candidates)>0:
            for path in nx.all_simple_paths(self.graph, source=source_node, target=candidates, cutoff=None):
                if path[-2] not in candidates:
                    paths.append(path)
        return paths, candidates
    def calculate_costs_for_sequence(self, sequence):
        cost = 0
        for i in range(len(sequence)-1):
            cost += self.graph[sequence[i]][sequence[i+1]]["operation"].data["cost"]
        return cost
    def plan_path_to_target(self, parts_to_be_disassembled: list[Part], max_iterations: int=100, result_path = None):
        #check if nodes exist with the parts already disassembled
        candidates = []
        for node in self.graph.nodes:
            all_parts_disassembled = True
            for part in parts_to_be_disassembled:
                if any(part.soft_compare(p) for p in self.assembly_states[node].parts.values()) or \
                        part in self.assembly_states[node].parts.values():
                    all_parts_disassembled = False
                    break
            if all_parts_disassembled:
                candidates.append(node)
        if len(candidates)>0:
            cheapest_candidate = candidates[0]
            for i in range(1, len(candidates)):
                if self.graph.nodes[candidates[i]]['cost']<self.graph.nodes[cheapest_candidate]['cost']:
                    cheapest_candidate = i
            return cheapest_candidate
        for i in range(max_iterations):
            current_node = self.get_cheapest_unexplored_node()
            if current_node <0:
                print("all nodes explored")
                break
            #print(f"{i} current node: {current_node}")
            try:
                print(f"{i} current node: {current_node} current cost: {self.graph.nodes[current_node]['cost']}" 
                     f" remaining parts: {self.assembly_states[current_node].generate_name_string()}")
            except:
                print("eeeeeeeeerooor")
            all_parts_disassembled = True
            for part in parts_to_be_disassembled:
                if any(part.soft_compare(p) for p in self.assembly_states[current_node].parts.values()) or \
                        part in self.assembly_states[current_node].parts.values():
                    all_parts_disassembled = False
                    break
            if result_path is not None:
                if not os.path.exists(f'{result_path}/sequence_plots'):
                    os.makedirs(f'{result_path}/sequence_plots')
                if not os.path.exists(f'{result_path}/DOF_plots'):
                    os.makedirs(f'{result_path}/DOF_plots')
                fig = self.plot_graph(
                    highlights=self.graph.nodes[current_node]['cheapest_sequence_to_node'] + [current_node])
                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
                # Save the figure with timestamp in the filename
                filename = f'{result_path}/sequence_plots/plot_{timestamp}.png'
                fig.savefig(filename)
                plt.close(fig)
                """figs = self.assembly_states[current_node].generate_figures()#generate_surface_plot()
                for fig in figs:
                    fig.savefig(f'{result_path}/DOF_plots/{timestamp}_sp.png')
                    plt.close(fig)"""
                """figures = self.assembly_states[current_node].generate_figures(2, 2)
                i = 0
                for fig in figures:
                    fig.savefig(f'{result_path}/DOF_plots/{timestamp}_dof{i}.png')
                    i += 1
                    plt.close(fig)"""

            if all_parts_disassembled:
                return current_node
            result = self.explore_node(current_node)
            self.update_graph_with_exploration_results(current_node, result)
            if result_path is not None:
                fig = self.plot_graph(
                    highlights=self.graph.nodes[current_node]['cheapest_sequence_to_node'] + [current_node])
                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
                # Save the figure with timestamp in the filename
                filename = f'{result_path}/sequence_plots/plot_{timestamp}.png'
                fig.savefig(filename)
                plt.close(fig)
                if len(self.assembly_states[current_node].parts) > 1:
                    fig = self.assembly_states[current_node].generate_surface_plot()
                    fig.savefig(f'{result_path}/DOF_plots/{timestamp}_sp.png')
                    plt.close(fig)
                """figures = self.assembly_states[current_node].generate_figures(2, 2)
                i = 0
                for fig in figures:
                    fig.savefig(f'{result_path}/DOF_plots/{timestamp}_dof{i}.png')
                    i += 1
                    plt.close(fig)"""

        return -1
    def end_nodes(self):
        return  [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]
    @staticmethod
    def plan_gripping_and_removing(
        assembly: Assembly,
        part: Part,
        cost_gripping_and_removing,
        max_grip_width: float = 50.0 * 999,
        *,
        _allow_secondary_path_lock: bool = True,
        _allow_reverse_secondary_path_lock: bool = True,
    ):
        cell_data = load_cell_settings()
        gripper_start_offset = float(cell_data.get("gripping_start_offset", 100.0))
        pullout_distance_factor = float(cell_data.get("gripping_pullout_factor", 2.1))
        minimal_pullout_value = float(cell_data.get("gripping_minimal_pullout_value", 5.0))
        approaching_feed_rate = float(cell_data.get("feed_rate_gripping_and_removing_approaching", 2000.0))
        leaving_feed_rate = float(cell_data.get("feed_rate_gripping_and_removing_leaving", 1000.0))

        # Forward secondary lock is now diagnostic-only: it must not mutate the real assembly.
        lock_part_after_all_secondary_path_failures = (
            bool(cell_data.get("lock_part_after_all_secondary_path_failures", True))
            and bool(_allow_secondary_path_lock)
        )
        reverse_secondary_path_lock_enabled = (
            bool(cell_data.get("reverse_secondary_path_lock_enabled", True))
            and bool(_allow_reverse_secondary_path_lock)
        )

        def _fail(msg: str, extra: dict | None = None):
            print(msg)
            data = {
                "part_name": part.name,
                "type": "gripping",
                "cost": -1,
                "direction": np.array([0.0, 0.0, 0.0]),
                "path": [],
                "failure_reason": msg,
            }
            if isinstance(extra, dict):
                data.update(extra)
            return GrippingAndRemovingOperation(data), assembly

        try:
            DOF_part = assembly.get_part_dof(part)
        except Exception as e:
            return _fail(
                f"get_part_dof failed for part {part.name}: {repr(e)}",
                {"failure_stage": "get_part_dof_exception"},
            )

        if not DOF_part:
            return _fail(
                f"Part {part.name} has no DOF, cannot plan gripping operation",
                {"failure_stage": "no_dof"},
            )

        cand: list[np.ndarray] = []
        invalid_dof_count = 0
        for d in DOF_part:
            try:
                v = np.asarray(d, dtype=float).reshape(3)
            except Exception:
                invalid_dof_count += 1
                continue
            n = float(np.linalg.norm(v))
            if n > 1e-9 and np.all(np.isfinite(v)):
                cand.append(v / n)
            else:
                invalid_dof_count += 1

        if not cand:
            return _fail(
                f"Part {part.name} has no usable DOF vectors after normalization",
                {"failure_stage": "no_usable_dof", "raw_dof_count": len(DOF_part), "invalid_dof_count": invalid_dof_count},
            )

        cand2: list[np.ndarray] = []
        for d in cand:
            if not any(float(np.linalg.norm(d - e)) < 1e-6 for e in cand2):
                cand2.append(d)

        if not cand2:
            return _fail(
                f"Part {part.name} has no finite DOF vectors after duplicate filtering",
                {"failure_stage": "no_finite_dof", "raw_dof_count": len(DOF_part), "invalid_dof_count": invalid_dof_count},
            )

        def _o3d_vertices(mesh):
            try:
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    v = np.asarray(mesh.vertices, dtype=float)
                    if v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0:
                        return v
            except Exception:
                pass
            v = getattr(mesh, "vertices", None)
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            if v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0:
                return v
            return None

        part_v = _o3d_vertices(getattr(part, "mesh", None))
        if part_v is None:
            return _fail(
                f"Part {part.name} mesh is not an Open3D TriangleMesh or has no vertices",
                {"failure_stage": "invalid_part_mesh"},
            )

        vs_all = []
        for p in assembly.parts.values():
            v = _o3d_vertices(getattr(p, "mesh", None))
            if v is not None:
                vs_all.append(v)
        if not vs_all:
            return _fail("Assembly has no meshes with vertices; cannot fit cylinder", {"failure_stage": "invalid_assembly_meshes"})

        complete_v = np.vstack(vs_all)
        if complete_v.shape[0] == 0:
            return _fail("Assembly merged vertices empty; cannot fit cylinder", {"failure_stage": "empty_assembly_vertices"})

        try:
            _, cylinder_part_data, _, _ = fit_cylinder(part_v, 32)
        except Exception as e:
            return _fail(
                f"fit_cylinder failed for part {part.name}: {repr(e)}",
                {"failure_stage": "fit_cylinder_exception", "error": repr(e)},
            )

        radius = float(cylinder_part_data.get("radius", 0.0))
        height = float(cylinder_part_data.get("height", 0.0))
        tf = np.asarray(cylinder_part_data.get("transform", np.eye(4)), dtype=float)
        if height <= 0.0 or tf.shape != (4, 4):
            return _fail(
                f"Cylinder fit invalid for part {part.name}: height={height}, tf_shape={tf.shape}",
                {"failure_stage": "invalid_cylinder_fit", "radius": radius, "height": height, "tf_shape": list(tf.shape)},
            )

        center = tf[:3, 3]
        assembly_center = np.mean(complete_v, axis=0)
        outward = center - assembly_center
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1e-9:
            outward = outward / outward_norm
            cand2 = sorted(cand2, key=lambda d: float(np.dot(d, outward)), reverse=True)

        cand2 = add_main_axis_candidates(cand2, outward if outward_norm > 1e-9 else None)

        candidate_failures: list[dict] = []
        cover_like = _looks_like_hollow_cover(part.name)

        for direction in cand2:
            direction = np.asarray(direction, dtype=float).reshape(3)
            dn = float(np.linalg.norm(direction))
            if dn < 1e-9 or not np.all(np.isfinite(direction)):
                candidate_failures.append({"direction": direction.tolist(), "reason": "zero_or_nonfinite_direction"})
                continue
            direction = direction / dn

            is_frozen, freeze_info = _is_direction_temporarily_frozen(
                assembly,
                target_part_name=part.name,
                direction=direction,
            )
            if is_frozen:
                candidate_failures.append({
                    "direction": direction.tolist(),
                    "reason": "direction_temporarily_frozen",
                    "freeze_info": freeze_info,
                })
                continue

            try:
                render_direction = snap_direction_to_axis(direction)
            except Exception:
                render_direction = direction.copy()

            try:
                if radius < max_grip_width:
                    start_point = center + direction * (height * 0.5 + gripper_start_offset)
                    mid_point = center + direction * (height * 0.5)
                    end_point = center + direction * max(height * (0.5 + pullout_distance_factor), minimal_pullout_value)
                else:
                    tri_mesh = getattr(part, "trimesh_mesh", None) or getattr(part, "mesh_trimesh", None) or getattr(part, "tri_mesh", None)
                    if tri_mesh is None:
                        candidate_failures.append({
                            "direction": direction.tolist(),
                            "reason": "radius_too_large_no_trimesh_grip_fallback",
                            "radius": radius,
                            "max_grip_width": max_grip_width,
                        })
                        continue
                    try:
                        mid_point, depth = find_pertruding_grip_points(tri_mesh, direction, max_grip_width)
                    except Exception as e:
                        candidate_failures.append({
                            "direction": direction.tolist(),
                            "reason": "find_pertruding_grip_points_exception",
                            "error": repr(e),
                            "radius": radius,
                            "max_grip_width": max_grip_width,
                        })
                        continue
                    if mid_point is None:
                        candidate_failures.append({
                            "direction": direction.tolist(),
                            "reason": "no_alternative_grip_point",
                            "radius": radius,
                            "max_grip_width": max_grip_width,
                        })
                        continue
                    mid_point = np.asarray(mid_point, dtype=float).reshape(3) + float(depth) * direction
                    start_point = mid_point + direction * 100.0
                    end_point = mid_point + direction * height * 2.6

                path = np.array([start_point, mid_point, end_point], dtype=float)
                path_length = gripper_start_offset + max(pullout_distance_factor * height, minimal_pullout_value)
                gripping_path_duration = round(
                    gripper_start_offset / approaching_feed_rate * 60.0
                    + max(pullout_distance_factor * height, minimal_pullout_value) / leaving_feed_rate * 60.0,
                    3,
                )

                op = GrippingAndRemovingOperation({
                    "part_name": part.name,
                    "type": "gripping",
                    "cost": cost_gripping_and_removing,
                    "direction": direction,
                    "render_direction": render_direction,
                    "path": path,
                    "path_length": path_length,
                    "path_duration": gripping_path_duration,
                    # Hollow / cover-like parts use the actual-mesh stepped path as
                    # the hard feasibility gate. The projected footprint remains
                    # diagnostic-only because it can fill empty interior space.
                    "ray_bundle_required": not cover_like,
                    "ray_bundle_sampling_mode": "boundary_vertices" if cover_like else "projected_footprint",
                    "ray_bundle_max_rays": 96 if cover_like else 200,
                    "stepped_path_step_size": 5.0,
                    "stepped_path_depth_threshold": 2.5,
                    "dof_candidate_count": len(cand2),
                    "raw_dof_count": len(DOF_part),
                    "invalid_dof_count": invalid_dof_count,
                    "selected_dof_direction": direction,
                    "outward_sort_used": bool(outward_norm > 1e-9),
                    "outward_direction": outward if outward_norm > 1e-9 else None,
                    "candidate_failures_before_selected": list(candidate_failures),
                    "gripping_radius": radius,
                    "gripping_height": height,
                    "max_grip_width": float(max_grip_width),
                    "hollow_cover_like_relaxation": bool(cover_like),
                })

                try:
                    feasible = bool(op.check_feasability(assembly))
                except Exception as e:
                    candidate_failures.append({"direction": direction.tolist(), "reason": "check_feasability_exception", "error": repr(e)})
                    continue

                if not feasible:
                    feasibility_debug = getattr(op, "data", {}).get("last_feasibility_debug", {})
                    candidate_failures.append({
                        "direction": direction.tolist(),
                        "reason": "check_feasability_failed",
                        "feasibility_reason": feasibility_debug.get("reason"),
                        "local_translation_feasible": feasibility_debug.get("local_translation_feasible"),
                        "path_ray_bundle_checked": feasibility_debug.get("path_ray_bundle_checked"),
                        "path_ray_bundle_reason": feasibility_debug.get("path_ray_bundle_reason"),
                        "ray_bundle_mode": feasibility_debug.get("ray_bundle_mode"),
                        "num_rays": feasibility_debug.get("num_rays"),
                        "num_blocking_hits": feasibility_debug.get("num_blocking_hits"),
                        "first_blocking_geometry": feasibility_debug.get("first_blocking_geometry"),
                        "first_blocking_t": feasibility_debug.get("first_blocking_t"),
                        "stepped_path_checked": feasibility_debug.get("stepped_path_checked"),
                        "stepped_path_feasible": feasibility_debug.get("stepped_path_feasible"),
                        "stepped_path_reason": feasibility_debug.get("stepped_path_reason"),
                        "stepped_path_step": feasibility_debug.get("stepped_path_step"),
                        "stepped_path_num_steps": feasibility_debug.get("stepped_path_num_steps"),
                        "ray_bundle_required": feasibility_debug.get("ray_bundle_required"),
                        "ray_bundle_warning_ignored": feasibility_debug.get("ray_bundle_warning_ignored"),
                    })
                    continue

                if reverse_secondary_path_lock_enabled:
                    reverse_lock_checked_neighbors: list[str] = []
                    reverse_lock_blocking_neighbors: list[str] = []
                    reverse_lock_freeze_relations: list[tuple[str, str]] = []
                    reverse_lock_neighbor_debug: list[dict] = []
                    current_neighbors = _geometric_neighbor_names_in_current_state(assembly, part.name)

                    for neighbor_name in current_neighbors:
                        neighbor_part = assembly.get_part_by_name(neighbor_name)
                        if neighbor_part is None:
                            continue
                        reverse_lock_checked_neighbors.append(neighbor_name)
                        try:
                            reverse_op, _reverse_asm = HierarchicalDisassemblySequencePlanner.plan_gripping_and_removing(
                                assembly=assembly,
                                part=neighbor_part,
                                cost_gripping_and_removing=cost_gripping_and_removing,
                                max_grip_width=max_grip_width,
                                _allow_secondary_path_lock=False,
                                _allow_reverse_secondary_path_lock=False,
                            )
                        except Exception as exc:
                            reverse_lock_neighbor_debug.append({"neighbor": neighbor_name, "stage": "reverse_plan_gripping_and_removing_exception", "error": repr(exc)})
                            continue

                        reverse_data = getattr(reverse_op, "data", {})
                        if not isinstance(reverse_data, dict):
                            reverse_lock_neighbor_debug.append({"neighbor": neighbor_name, "stage": "reverse_plan_gripping_and_removing_no_data"})
                            continue

                        reverse_candidate_failures = reverse_data.get("candidate_failures", [])
                        reverse_blocked_by_current_part = (
                            reverse_data.get("failure_stage") == "no_feasible_dof_candidate"
                            and isinstance(reverse_candidate_failures, list)
                            and _secondary_path_failures_blocked_by_part(reverse_candidate_failures, blocker_part_name=part.name)
                        )

                        reverse_lock_neighbor_debug.append({
                            "neighbor": neighbor_name,
                            "reverse_failure_stage": reverse_data.get("failure_stage"),
                            "reverse_failure_reason": reverse_data.get("failure_reason"),
                            "reverse_dof_candidate_count": reverse_data.get("dof_candidate_count"),
                            "reverse_candidate_failures_count": len(reverse_candidate_failures) if isinstance(reverse_candidate_failures, list) else None,
                            "reverse_blocked_by_current_part": bool(reverse_blocked_by_current_part),
                        })

                        if reverse_blocked_by_current_part:
                            reverse_lock_blocking_neighbors.append(neighbor_name)
                            _add_direction_specific_freeze(
                                assembly,
                                target_part_name=part.name,
                                direction=direction,
                                blocker_part_name=neighbor_name,
                                reason="reverse_secondary_path_lock",
                            )
                            reverse_lock_freeze_relations.append((part.name, neighbor_name))

                    if reverse_lock_blocking_neighbors:
                        print(
                            f"[plan_gripping_and_removing] direction-specific freeze rejected removal of {part.name}: "
                            f"blocking neighbors={reverse_lock_blocking_neighbors}, frozen={reverse_lock_freeze_relations}"
                        )
                        return _fail(
                            f"Removal of {part.name} rejected by direction-specific secondary freeze.",
                            {
                                "failure_stage": "direction_specific_freeze",
                                "reverse_secondary_path_lock_enabled": bool(reverse_secondary_path_lock_enabled),
                                "reverse_secondary_path_lock_applied": False,
                                "direction_specific_freeze_enabled": True,
                                "direction_specific_freeze_applied": bool(reverse_lock_blocking_neighbors),
                                "direction_specific_freeze_target_part": part.name,
                                "direction_specific_freeze_direction": direction.tolist(),
                                "direction_specific_freeze_blockers": reverse_lock_blocking_neighbors,
                                "direction_specific_freeze_relations": [[a, b] for a, b in reverse_lock_freeze_relations],
                                "reverse_secondary_path_lock_rejected_part": part.name,
                                "reverse_secondary_path_lock_checked_neighbors": reverse_lock_checked_neighbors,
                                "reverse_secondary_path_lock_blocking_neighbors": reverse_lock_blocking_neighbors,
                                "reverse_secondary_path_lock_added_fixed_connections": [],
                                "reverse_secondary_path_lock_added_count": 0,
                                "reverse_secondary_path_lock_neighbor_debug": reverse_lock_neighbor_debug,
                                "would_have_selected_direction": direction.tolist(),
                                "would_have_selected_path_length": float(path_length),
                                "would_have_selected_path_duration": float(gripping_path_duration),
                                "dof_candidate_count": len(cand2),
                                "raw_dof_count": len(DOF_part),
                                "invalid_dof_count": invalid_dof_count,
                                "candidate_failures_before_selected": list(candidate_failures),
                                "gripping_radius": radius,
                                "gripping_height": height,
                                "max_grip_width": float(max_grip_width),
                            },
                        )

                new_assembly = copy.copy(assembly)
                new_assembly.remove_part(part)
                print(
                    f"gripping path length: {path_length} mm, "
                    f"gripper_start_offset: {gripper_start_offset} mm & "
                    f"pullout_distance_factor: {pullout_distance_factor}*height {height} = {height * pullout_distance_factor} mm"
                )
                print(f"gripping path duration: {gripping_path_duration} seconds")
                print(f"[plan_gripping_and_removing] selected feasible DOF for {part.name}: {direction.tolist()} out of {len(cand2)} candidates")
                return op, new_assembly

            except Exception as e:
                candidate_failures.append({"direction": direction.tolist(), "reason": "candidate_planning_exception", "error": repr(e)})
                continue

        secondary_path_lock_added: list[tuple[str, str]] = []
        secondary_path_lock_current_parts: list[str] = []
        secondary_path_lock_current_geometric_neighbors: list[str] = []
        secondary_path_lock_should_apply = (
            lock_part_after_all_secondary_path_failures
            and should_lock_part_after_all_secondary_path_failures(candidate_failures)
        )

        if secondary_path_lock_should_apply:
            # Diagnostic-only temporary freeze probe. Do not mutate the real assembly.
            freeze_probe_assembly = copy.copy(assembly)
            (
                secondary_path_lock_added,
                secondary_path_lock_current_parts,
                secondary_path_lock_current_geometric_neighbors,
            ) = add_fixed_connections_to_current_geometric_neighbors(
                assembly=freeze_probe_assembly,
                part_name=part.name,
            )
            if secondary_path_lock_added:
                print(
                    f"[plan_gripping_and_removing] temporary secondary freeze for {part.name}: "
                    f"would add fixed connections {secondary_path_lock_added}"
                )
            else:
                print(
                    f"[plan_gripping_and_removing] temporary secondary freeze triggered for {part.name}, "
                    f"but no fixed connections would be added. "
                    f"current_geometric_neighbors={secondary_path_lock_current_geometric_neighbors}"
                )

        return _fail(
            f"Part {part.name} has DOF candidates, but none passed gripping feasibility. num_candidates={len(cand2)}",
            {
                "failure_stage": "no_feasible_dof_candidate",
                "dof_candidate_count": len(cand2),
                "raw_dof_count": len(DOF_part),
                "invalid_dof_count": invalid_dof_count,
                "candidate_failures": candidate_failures,
                "gripping_radius": radius,
                "gripping_height": height,
                "max_grip_width": float(max_grip_width),
                "secondary_path_lock_enabled": bool(lock_part_after_all_secondary_path_failures),
                "secondary_path_lock_should_apply": bool(secondary_path_lock_should_apply),
                "secondary_path_lock_applied": False,
                "secondary_path_lock_current_parts": secondary_path_lock_current_parts,
                "secondary_path_lock_current_geometric_neighbors": secondary_path_lock_current_geometric_neighbors,
                "secondary_path_lock_added_fixed_connections": [],
                "secondary_path_lock_added_count": 0,
                "temporary_secondary_freeze": bool(secondary_path_lock_should_apply),
                "temporary_secondary_freeze_would_add_fixed_connections": [[a, b] for a, b in secondary_path_lock_added],
                "temporary_secondary_freeze_would_add_count": len(secondary_path_lock_added),
            },
        )

    @staticmethod
    def plan_safety_ring_disassembly(assembly: Assembly, connection: Safety_Ring_Connection, cost_safety_ring, max_grip_width=50.0*999):
        part = assembly.get_part_by_name(connection.part_name1)
        min_cylinder_data = trimesh.bounds.minimum_cylinder(part.mesh)
        rotation_matrix = min_cylinder_data["transform"][:3, :3]
        normal = rotation_matrix @ np.array([0, 0, 1])

        # slice part at mid height
        slice = part.mesh.section(plane_origin=min_cylinder_data["transform"][:3, 3], plane_normal=normal)

        # search for circles
        circles = []
        temp_colors = []
        for i in range(len(slice.entities)):
            points = slice.vertices[slice.entities[i].points]

            # Compute centroid and distances to centroid
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)

            # Check circularity via standard deviation of radii
            radius_mean = distances.mean()
            radius_std = distances.std()

            # Set a tolerance for circularity (e.g., std deviation less than 5%)
            print(radius_std / radius_mean)
            if radius_std / radius_mean < 0.05 and radius_mean < 3:  # TODO: implement as dynamic tolerance setting for radius depending on size of part
                circles.append({
                    'center': centroid,
                    'radius': radius_mean,
                    #'entity': slice.entities[i]
                })
                temp_colors.append(wbk_color_dict["green"])
            else:
                temp_colors.append(wbk_color_dict["blue"])
        slice.colors = temp_colors
        #set mesh color
        part.mesh.visual.vertex_colors = [[128, 128, 128, 180] for _ in range(len(part.mesh.vertices))]

        # ---- Add plane geometry for visualization ----
        plane_center = min_cylinder_data["transform"][:3, 3]
        plane_normal = normal
        plane_size = min_cylinder_data["radius"]*2.1  # Adjust size as needed

        # Create a square plane mesh centered on the slicing point
        plane = trimesh.creation.box(extents=[plane_size, plane_size, 0.01])
        plane.apply_translation(-plane.center_mass)

        # Align plane to slicing normal
        T = trimesh.geometry.align_vectors([0, 0, 1], plane_normal)
        plane.apply_transform(T)

        # Move to slice position
        plane.apply_translation(plane_center)

        # Set transparent color
        plane.visual.face_colors = np.append(wbk_color_dict['red'], 60)  # RGBA

        # ---- Show everything in scene ----
        #scene = trimesh.Scene([part.mesh, slice, plane])
        #scene = trimesh.Scene([part.mesh, slice])
        #scene.show()

        # check wether there are exactly two circles left TODO: check wether there are safety rings with more than two holes
        if len(circles) != 2 or abs(circles[0]['radius'] - circles[1]['radius']) > 0.25:
            print("Did not find exactly two circles with similar radii")
            operation = SafetyRingOperation({"part_name": part.name,
                                             "cost": -1,
                                             "direction": np.array([0, 0, 0]),
                                             "grip_widths": [],
                                             "orientations": [],
                                             "approach_index": -1,
                                             "path": []})
            return operation, assembly

        hole_positions = [circle['center'] for circle in circles]
        hole_radius = circles[0]['radius']

        segments = []
        """for circle in circles:
            # Create a new Path3D object with the same vertices but only one entity
            new_path = trimesh.path.Path3D(vertices=slice.vertices, entities=[circle["entity"]])
            new_path.colors = [wbk_color_dict["green"] for _ in range(len(new_path.entities))]
            segments.append(new_path)"""
        """for entity in slice.entities:
            slice.colors[entity.points] = wbk_color_dict["green"]
            # Create a new Path3D object with the same vertices but only one entity
            new_path = trimesh.path.Path3D(vertices=slice.vertices, entities=[entity])
            new_path.colors = [wbk_color_dict["blue"] for _ in range(len(new_path.entities))]
            segments.append(new_path)"""



        connections = assembly.get_connections_by_part_name(part.name)
        connections.remove(connection)
        ignore_connections = [connection]
        for conn in connections:
            if conn.connection_type == "geometric" and \
                    (conn.part_name1 == connection.part_name1 and
                    conn.part_name2 == connection.part_name2) or \
                    (conn.part_name1 == connection.part_name2 and
                    conn.part_name2 == connection.part_name1):
                ignore_connections.append(conn)

        DOF_part = assembly.get_part_dof(part, ignored_connections=ignore_connections)

        if min(np.linalg.norm(DOF_part - normal, axis=1)) < 0.05:
            direction = normal
        elif min(np.linalg.norm(DOF_part + normal, axis=1)) < 0.05:
            direction = -normal
        else:
            print("Could not find movable direction")
            operation = SafetyRingOperation({"part_name": part.name,
                                             "cost": -1,
                                             "direction": np.array([0, 0, 0]),
                                             "grip_widths": [],
                                             "orientations": [],
                                             "approach_index": -1,
                                             "path": []})
            return operation, assembly

        center_point_holes = (hole_positions[0] + hole_positions[1]) / 2
        base_grip_width = np.linalg.norm(hole_positions[0] - hole_positions[1])
        direction_forwards = min_cylinder_data["transform"][:3, 3] - center_point_holes
        direction_forwards = direction_forwards / np.linalg.norm(direction_forwards)
        base_orn = rotation_matrix_from_zy(direction, direction_forwards)



        x, y, z = generate_double_spiral_above_origin(1.0, hole_radius, 0.5, 4.0)
        spiral_path = np.array([x, y, z]).transpose()
        transform = np.eye(4)
        transform[:3, :3] = base_orn
        transform[:3, 3] = center_point_holes
        spiral_path = np.dot(transform,np.hstack((spiral_path, np.ones((len(spiral_path), 1)))).T).T[:, :3]

        _, cylinder_data, _, _ = fit_minimum_volume_cylinder_along_axis(part.mesh.vertices, direction, 32)
        center_shaft = cylinder_data["transform"][:3, 3]
        current_radius = np.linalg.norm(center_point_holes - center_shaft)
        target_radius = current_radius * 1.15
        final_grip_width = calculate_new_chord_length(current_radius, base_grip_width, target_radius)
        #generate array of grip widths same length as path
        grip_widths = np.array([base_grip_width for _ in range(len(spiral_path)+1)])
        grip_widths_gripping_and_removing = np.linspace(base_grip_width, final_grip_width, 10)
        grip_widths = np.concatenate([grip_widths, grip_widths_gripping_and_removing, [final_grip_width]])

        #center_shift = compute_center_shift(current_radius, base_grip_width, target_radius, final_grip_width)
        grip_path = np.array([spiral_path[-1] for _ in(range(10))])
        #grip_path = np.linspace(spiral_path[-1], spiral_path[-1]+center_shift*direction_forwards, num=10)
        # Compute center shift for each grip width
        grip_path = np.array([
            spiral_path[-1] + 0.5*compute_center_shift(current_radius, base_grip_width, target_radius,
                                                   gw) * direction_forwards
            for gw in grip_widths_gripping_and_removing
        ])
        start_point = center_point_holes + direction * 100
        end_point = grip_path[-1] + direction * 100

        path = np.vstack([
            start_point.reshape(1, 3),  # Reshape to (1, 3) so it can be stacked
            spiral_path,  # Spiral path points
            grip_path,
            end_point.reshape(1, 3)  # Reshape to (1, 3) so it can be stacked
        ])



        #rotation offset to allign with actual gripper TCP
        #rotation_offset = R.from_euler('z', 180, degrees=True)
        #orientation = np.dot(base_orn, rotation_offset.as_matrix())
        orientations = np.array([base_orn for _ in range(len(path))])
        approach_index = len(path)-2


        new_assembly = copy.copy(assembly)#used to be deepcopy
        new_assembly.remove_part(part)



        operation = SafetyRingOperation({"part_name": part.name,
                                         "cost": cost_safety_ring,
                                         "direction": direction,
                                         "grip_widths": grip_widths,
                                         "orientations": orientations,
                                         "approach_index": approach_index,
                                         "path": path})
        if not operation.check_feasability(assembly):
            operation = SafetyRingOperation({"part_name": part.name,
                                             "cost": -1,
                                             "direction": np.array([0, 0, 0]),
                                             "grip_widths": [],
                                             "orientations": [],
                                             "approach_index": -1,
                                             "path": []})
            return operation, assembly
        return operation, new_assembly

    @staticmethod
    def plan_screwing(assembly: Assembly, connection: Screw_Connection, cost_screwing):
        cell_data = load_cell_settings()
        screwdriver_start_offset = float(cell_data.get("screwing_start_offset", 100.0))
        pullout_distance_factor = float(cell_data.get("screwing_pullout_factor", 1.1))
        minimal_pullout_value = float(cell_data.get("screwing_minimal_pullout_value", 5.0))
        approaching_feed_rate = float(cell_data.get("feed_rate_screwing_approaching", 2000.0))
        leaving_feed_rate = float(cell_data.get("feed_rate_screwing_leaving", 1000.0))

        new_assembly = copy.copy(assembly)
        new_assembly.remove_connection(connection)
        screw_part = new_assembly.get_part_by_name(connection.part_name1)
        direction = None

        def _vertices_from_mesh(mesh):
            if mesh is None:
                return None
            try:
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    v = np.asarray(mesh.vertices, dtype=float)
                    if v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0:
                        return v
            except Exception:
                pass
            v = getattr(mesh, "vertices", None)
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            if v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0:
                return v
            return None

        def _pca_axis(vertices: np.ndarray) -> np.ndarray | None:
            if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3:
                return None
            X = np.asarray(vertices, dtype=float)
            X = X[np.all(np.isfinite(X), axis=1)]
            if X.shape[0] < 10:
                return None
            Y = X - X.mean(axis=0, keepdims=True)
            C = (Y.T @ Y) / max(1, Y.shape[0] - 1)
            try:
                w, V = np.linalg.eigh(C)
            except Exception:
                return None
            axis = V[:, int(np.argmax(w))]
            n = float(np.linalg.norm(axis))
            if n < 1e-9:
                return None
            axis = axis / n
            if float(np.dot(axis, np.array([0.0, 0.0, 1.0]))) < 0.0:
                axis = -axis
            return axis

        if screw_part is not None:
            direction = _pca_axis(_vertices_from_mesh(getattr(screw_part, "mesh", None)))

        if direction is None and screw_part is not None:
            try:
                DOF_part = new_assembly.get_part_dof(screw_part)
            except Exception:
                DOF_part = None
            if DOF_part:
                d = np.mean(DOF_part, axis=0)
                n = float(np.linalg.norm(d))
                if n > 1e-9:
                    direction = d / n

        if direction is None:
            direction = np.array([0.0, 0.0, 0.0], dtype=float)
        direction[np.abs(direction) < 1e-8] = 0.0

        print(f"computing screwing {connection} direction: {direction}")
        if float(np.linalg.norm(direction)) < 0.1 or screw_part is None:
            print(f"no direction detectable for {connection.part_name1}")
            return ScrewingOperation({"cost": -1, "connection": connection, "direction": direction, "path": np.array([])}), new_assembly

        screw_vtx = _vertices_from_mesh(getattr(screw_part, "mesh", None))
        if screw_vtx is None:
            return ScrewingOperation({"cost": -1, "connection": connection, "direction": direction, "path": np.array([])}), new_assembly

        try:
            _, cylinder_data, _, _ = fit_minimum_volume_cylinder_along_axis(screw_vtx, direction, 32)
        except TypeError:
            _, cylinder_data, _, _ = fit_minimum_volume_cylinder_along_axis(screw_vtx, direction)

        height = float(cylinder_data["height"])
        center = np.asarray(cylinder_data["transform"][:3, 3], dtype=float)
        start_point = center + direction * (height * 0.5 + screwdriver_start_offset)
        mid_point = center + direction * (height * 0.5)
        end_point = center + direction * max(height * (0.5 + pullout_distance_factor), minimal_pullout_value)
        path = np.vstack([start_point.reshape(1, 3), mid_point.reshape(1, 3), end_point.reshape(1, 3)])

        path_length = screwdriver_start_offset + max(pullout_distance_factor * height, minimal_pullout_value)
        screwing_path_duration = round(
            screwdriver_start_offset / approaching_feed_rate * 60.0
            + max(pullout_distance_factor * height, minimal_pullout_value) / leaving_feed_rate * 60.0,
            3,
        )

        new_assembly.remove_part(screw_part)
        operation = ScrewingOperation({
            "cost": cost_screwing,
            "connection": connection,
            "direction": direction,
            "path": path,
            "path_length": path_length,
            "path_duration": screwing_path_duration,
        })
        return operation, new_assembly

    @staticmethod
    def plan_milling(assembly: Assembly, connection: Geometric_Connection, mill_diameter=5.0, cost_milling=10.0, invert=False):
        """
        Plan geometric milling for two parts based on the provided connection and mill diameter.

        Args:
            connection (Geometric_Connection): A geometric connection object detailing how two parts are connected.
            assembly (Assembly): An assembly object containing parts information.
            mill_diameter (float, optional): Diameter of the mill. Defaults to 5.0.

        This function determines the milling paths for two connected parts. It computes the intersection
        points between the parts and then plans the milling paths in the form of helical paths.
        """

        def compute_milling(part_a, part_b, direction, assembly):
            #start_time = time.time()
            cell_data = load_cell_settings()
            feed_rate = cell_data["feed_rate_milling"]
            cylinder_a, cylinder_a_data, _, _ = fit_minimum_volume_cylinder_along_axis(part_a.mesh.vertices, direction)
            cylinder_b, cylinder_b_data, _, _ = fit_minimum_volume_cylinder_along_axis(part_b.mesh.vertices, direction)

            """for data_set in self.calculated_milling_data:
                if part_a.name == data_set[0] and part_b.name == data_set[1]:
                    helix_radius = data_set[2]
                    cylinder_max_data = data_set[3]
                    already_calculated=True
                    break"""


            intersect_points = []
            #part_a.mesh.visual.vertex_colors = [128, 128, 128, 100]
            part_b.mesh.visual.vertex_colors = [128, 128, 128, 100]
            #scene = trimesh.Scene([part_a.mesh, part_b.mesh])
            #scene.show()
            for point in part_a.mesh.vertices:
                intersects = part_b.mesh.ray.intersects_location(ray_origins=[point], ray_directions=[direction])
                #####################
                #p = trimesh.load_path(np.array([[point, point + direction * 50]]))
                #p.colors = [[238, 183, 13, 50] for _ in range(len(p.entities))]
                #scene.add_geometry(p)
                #####################
                for intersect in intersects[0]:
                    intersect_points.append(intersect)
                ####################################################################
                #    sphere = trimesh.creation.icosphere(radius=1.5, subdivisions=0)
                #    sphere.vertices += intersect
                #    sphere.visual.face_colors = [178, 55, 44, 255]  # Set the color to red (RGBA format)
                #    scene.add_geometry(sphere)
                #VISUALIZATION######################################################
            #scene.show()
           # print(f"calculating min max cylinders {time.time()-start_time}")
            """if len(intersect_points)<=0:
                part_a.mesh.visual.vertex_colors = [178, 55, 44, 255]
                scene = trimesh.Scene([part_a.mesh, part_b.mesh])
                for point in part_a.mesh.vertices:
                    p = trimesh.load_path(np.array([[point, point + direction * 50]]))
                    scene.add_geometry(p)
                scene.show()"""

            """cylinder_max, cylinder_max_data, cylinder_min, cylinder_min_radius, _, _ = fit_min_max_cylinder(
                intersect_points)"""
            if len(intersect_points) <= 0:
                print(f"No intersection points found for milling {part_a.name} and {part_b.name}.")
                operation_parameter = {"part_not_milled": part_a,
                                       "part_milled": part_b,
                                       "mill_volume": None,
                                       "mill_diameter": mill_diameter,
                                       "cylinder_data": None,
                                       "resulting_mesh": None,
                                       "path": [[]],
                                       "direction": direction,
                                       "no_intersection_found": True,
                                       "similar_radius": False}
                operation = MillingOperation(operation_parameter)
                return False, operation

            cylinder_max, cylinder_max_data, cylinder_min, cylinder_min_radius, _, _ = fit_min_max_cylinder_along_axis(
                intersect_points, direction)
            #TODO: mesh is already "cylinder_max"?
            cylinder_max_mesh = trimesh.creation.cylinder(radius=cylinder_max_data["radius"], height=cylinder_max_data["height"], transform=cylinder_max_data["transform"])
            cylinder_max_mesh.visual.vertex_colors = [178, 55, 44, 100]
            cylinder_min_mesh = trimesh.creation.cylinder(radius=cylinder_min_radius, height=cylinder_max_data["height"]+0.2, transform=cylinder_max_data["transform"])
            cylinder_min_mesh.visual.vertex_colors = [20, 68, 102, 150]
            #scene.add_geometry(cylinder_max_mesh)
            #scene.add_geometry(cylinder_min_mesh)
            """scene = trimesh.Scene([part_a.mesh, part_b.mesh, cylinder_max_mesh, cylinder_min_mesh])
            for intersect in intersect_points:
                ####################################################################
                sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=0)
                sphere.vertices += intersect
                sphere.visual.face_colors = [178, 55, 44, 255]  # Set the color to red (RGBA format)
                scene.add_geometry(sphere)
            scene.show()"""

            # important for future use: if top and bottom points of cylinder are used, they might need to be swapped
            cylinder_up = np.dot(cylinder_max_data["transform"][:3, :3], [0, 0, 1])
            if np.dot(cylinder_up, direction) / np.linalg.norm(cylinder_up) / np.linalg.norm(
                    direction) < 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                cylinder_max_data["transform"] = np.dot(cylinder_max_data["transform"], rotation_matrix)

            #print(f"calculating helix radius {time.time()-start_time}")
            helix_radius = determine_helix_radius(cylinder_a_data["radius"], cylinder_b_data["radius"],
                                                  cylinder_max_data["radius"], cylinder_min_radius, mill_diameter,
                                                  offset = 0.2)

           # print(f"calculating milling volume {time.time()-start_time}")
            mill_volume = create_mill_volume(helix_radius, cylinder_max_data, mill_diameter, offset=0.1)
            """####Debug - Not all mesh are volumes - Prints kann man auch auskommentieren
            print(f"Active Part: '{part_b.name}'")
            print(f"part_b.mesh.is_watertight: {part_b.mesh.is_watertight}")
            print(f"mill_volume.is_watertight: {mill_volume.is_watertight}")
            print(f"part_b.mesh.volume: {part_b.mesh.volume}")
            print(f"mill_volume.volume: {mill_volume.volume}")"""

            #print(trimesh.Trimesh == trimesh.base.Trimesh)
            
            #print(type(part_b.mesh))
            if not part_b.mesh.is_watertight:
                print("Warning: part_b.mesh is not watertight. Attempting to fix it.")
                fixed_mesh = part_b.mesh.fill_holes()
                if isinstance(fixed_mesh, trimesh.Trimesh):
                    part_b.mesh = fixed_mesh
                    print("is fixed")
                else:
                    print("Warning: fill_holes() failed. Using convex hull as a fallback.")
                    part_b.mesh = part_b.mesh.convex_hull  # Last resort fix - Diese Fkt behebt das Problem

            """result_difference = part_b.mesh.difference(mill_volume)
            if not isinstance(result_difference, trimesh.Trimesh):
                print("Boolean difference operation failed!")
                print("Returned:", result_difference)
                return False  # or handle the failure gracefully"""
            """scene = trimesh.Scene()
            scene.add_geometry(part_b.mesh)
            scene.add_geometry(mill_volume)
            scene.show()"""
            result_difference = part_b.mesh.difference(mill_volume)
            if isinstance(result_difference, trimesh.Scene):
                resulting_meshes = []
            else:
                resulting_meshes = part_b.mesh.difference(mill_volume).split(only_watertight=False)
            old_meshes = []
            new_meshes = []
            check_cylinder = trimesh.creation.cylinder(helix_radius, cylinder_max_data["height"] + 1.0,
                                                       transform=cylinder_max_data["transform"])
            """mill_volume.visual.vertex_colors = [178, 55, 44, 255]
            scene = trimesh.Scene([mill_volume, part_b.mesh])
            scene.show()

            scene = trimesh.Scene([mill_volume, result_difference])
            scene.show()"""
        # print(f"calculating resulting meshes {time.time()-start_time}")
            for m in resulting_meshes:
                if check_cylinder.contains(m.vertices).any():
                    if len(m.vertices)>min(len(part_b.mesh.vertices), 100): #TODO: needs to be parameter or  Dynamic value!!!
                        new_meshes.append(m)
                else:
                    if len(m.vertices) > min(len(part_b.mesh.vertices), 100):
                        old_meshes.append(m)
        #print(f"merging {time.time()-start_time}")
            if len(old_meshes) > 0:
                part_b.mesh = trimesh.util.concatenate(old_meshes)
                part_b.mesh.visual.vertex_colors = [0, 150, 130, 255]
                if len(new_meshes) > 0:
                    resulting_mesh = trimesh.util.concatenate(new_meshes)
                    resulting_mesh.visual.vertex_colors = [20, 68, 102, 255]
                else:
                    resulting_mesh = None
            elif len(new_meshes) > 0:
                part_b.mesh = trimesh.util.concatenate(new_meshes)
                part_b.mesh.visual.vertex_colors = [0, 150, 130, 255]
                resulting_mesh = None
            else:
                part_b.mesh = None
                resulting_mesh = None

            helix_radius = determine_helix_radius(cylinder_a_data["radius"], cylinder_b_data["radius"],
                                                  cylinder_max_data["radius"], cylinder_min_radius, mill_diameter,
                                                  offset = -0.2)
            mill_volume = create_mill_volume(helix_radius+0.4, cylinder_max_data, mill_diameter,offset=-0.1)

            #print(f"generate path {time.time()-start_time}")
            helix_radius = determine_helix_radius(cylinder_a_data["radius"], cylinder_b_data["radius"],
                                                  cylinder_max_data["radius"], cylinder_min_radius, mill_diameter,
                                                  offset = 1.0) #TODO: check if offset is correct --> 4 for milling dummy 1 for motor
            #x, y, z = generate_helix_path(1.0, helix_radius, 0.5, cylinder_max_data["height"])#TODO: point dist wrong!
            x, y, z = generate_helix_path(1.0, helix_radius, 2.5, cylinder_max_data["height"]+1.0)#TODO: point dist wrong! 
            # distance_between_points: Controls the density of points along the path, radius of the helix (distance from central axis); pitch: Vertical distance covered in one complete revolution; height: Total vertical height of the helix
            path = np.array([x, y, z]).transpose()
            path = np.dot(cylinder_max_data["transform"],
                          np.hstack((path, np.ones((len(path), 1)))).T).T[:, :3]
            #line = trimesh.load_path(path)
            # line.apply_transform(cylinder_max_data["transform"])
            if part_b.mesh is not None:
                part_b.mesh.visual.vertex_colors = [0, 150, 130, 255]
            path_length = helix_with_endcircle_length(helix_radius, 2.5, cylinder_max_data["height"]+1.0) # in mm
            milling_path_duration = round(helix_with_endcircle_length(helix_radius, 2.5, cylinder_max_data["height"]+1.0) / feed_rate * 60,3) # Speed mm/min -> multiply with 60 to get seconds
            #print(f"checking feasibility {time.time()-start_time}")
            operation_parameter = {"part_not_milled": part_a,
                                   "part_milled": part_b,
                                   "mill_volume": mill_volume,
                                   "mill_diameter": mill_diameter,
                                   "cylinder_data": cylinder_max_data,
                                   "resulting_mesh": resulting_mesh,
                                   "path": path,
                                   "direction": direction,
                                   "path_length": path_length, #(mm)
                                   "path_duration": milling_path_duration} #(s)
            
            print(f"milling path length: {path_length} units")
            print(f"milling path duration: {milling_path_duration} seconds")
            #check wheather parts are of similar size --> these are filtered out in feasability check
            if 0.99 < cylinder_a_data["radius"] / cylinder_b_data["radius"] <1.01 :
                operation_parameter["similar_radius"] = True
            else:
                operation_parameter["similar_radius"] = False
            operation_parameter["no_intersection_found"] = False
            operation = MillingOperation(operation_parameter)
            feasible = operation.check_feasability(assembly)
            """for part in assembly.parts.values():
                if part != part_b and not _safe_check_reachability(mill_volume, part.mesh, direction):
                    #print(f"collision part {part.name}")
                    part.mesh.visual.vertex_colors = [255, 100, 100, 255]
                    parts_to_show.append(part.mesh)
                    feasible = False

            if cylinder_max_data["height"]>mill_diameter * 5:
                feasible = False"""

            """translation_matrix_tool = trimesh.transformations.translation_matrix(np.array([0, 0, mill_diameter * 5]))
            rotation_matrix_tool = trimesh.geometry.align_vectors([0, 0, 1], direction / np.linalg.norm(direction))
            transformation_matrix_tool = np.dot(np.dot(trimesh.transformations.translation_matrix(np.array(path[0])),
                                                       rotation_matrix_tool), translation_matrix_tool)
            tool = trimesh.creation.cylinder(radius=mill_diameter / 2, height=mill_diameter * 10,
                                             transform=transformation_matrix_tool, sections=64)
            tool.visual.vertex_colors = [200, 200, 100, 255]
            parts_to_show = [tool, part_a.mesh, part_b.mesh]
            if resulting_mesh is not None:
                parts_to_show.append(resulting_mesh)
            #parts_to_show.append(tool)
            # Visualization
            line.colors = [[255, 0, 0, 255] for _ in range(len(line.entities))]
            parts_to_show.append(line)
            print(f"visualizing")
            scene = trimesh.Scene(parts_to_show)
            scene.show()"""
            """if not already_calculated:
                self.calculated_milling_data.append([part_a.name, part_b.name, helix_radius, cylinder_max_data])"""
            #print(f"done {time.time()-start_time}")
            #print(f"computed milling {part_a.name} {part_b.name}")
            return feasible, operation#resulting_mesh, mill_volume, line

        def determine_helix_radius(radius_a, radius_b, max_radius, min_radius, mill_diameter, offset = 0.2):
            # check wich part is smaller and use corresponding rasius as a border for milling
            #TODO HELIX RADIUS!!!! --> if something went wrong
            """if radius_a <= radius_b:
                return max(max_radius - mill_diameter / 2 + 0.2, 0.1)
            return max(min_radius + mill_diameter / 2 - 0.2, 0.1)"""
            if radius_a <= radius_b:
                return max(max_radius - mill_diameter / 2 + offset, 0.1)
            return max(min_radius + mill_diameter / 2 - offset, 0.1)
        
        
        def helix_with_endcircle_length(radius, pitch, height):
            num_turns = height / pitch
            helix_length = np.sqrt((2 * np.pi * radius)**2 + pitch**2) * num_turns

            # Circle at bottom
            circle_length = 2 * np.pi * radius

            total_length = helix_length + circle_length

            return total_length


        def create_mill_volume(helix_radius, cylinder_data, mill_diameter, offset=0.1):
            mill_cyl_1 = trimesh.creation.cylinder(helix_radius - mill_diameter / 2,
                                                   cylinder_data["height"] + offset + 0.1, sections = 128)
            mill_cyl_2 = trimesh.creation.cylinder(helix_radius + mill_diameter / 2, cylinder_data["height"] + offset, sections = 128)
            mill_volume = mill_cyl_2.difference(mill_cyl_1)
            #print(cylinder_data["transform"]) Debug - Not all mesh are volumes
            mill_volume.apply_transform(cylinder_data["transform"])
            return mill_volume

        if len(connection.DOF)==0 or np.linalg.norm(np.mean(connection.DOF, axis=0)) < 0.25:
            print(f"not computing milling {connection.part_name1} {connection.part_name2}")
            print(len(connection.DOF))
            print(np.mean(connection.DOF, axis=0))
            return
        assembly_work_copy = copy.deepcopy(assembly)
        if not invert:
            part1 = assembly_work_copy.get_part_by_name(connection.part_name1)
            part2 = assembly_work_copy.get_part_by_name(connection.part_name2)
            direction = np.mean(connection.DOF, axis=0)
            direction = direction / np.linalg.norm(direction)
        else:
            part1 = assembly_work_copy.get_part_by_name(connection.part_name2)
            part2 = assembly_work_copy.get_part_by_name(connection.part_name1)
            direction = - np.mean(connection.DOF, axis=0)
            direction = direction / np.linalg.norm(direction)
        feasible, operation = compute_milling(part1, part2, direction, assembly_work_copy)
        operation.data["connection"]=connection
        operation.data["inverted"]=invert
        if feasible:#TODO large part of this code is duplicate to apply to assembly code! (i.e. checking screw connections) --> NOT A GOOD SOLUTION
            connections = assembly_work_copy.get_connections_by_part_name(part2.name)
            sc_to_recheck = [conn for conn in connections if conn.connection_type == 'screw']

            if operation.data["resulting_mesh"] is not None:
                new_part_name = part2.name + "_milled_" + part1.name.split("_milled_")[0] +"_2"
                if any(p.name == new_part_name for p in assembly_work_copy.parts.values()):
                    assembly_work_copy.remove_part(assembly_work_copy.get_part_by_name(new_part_name))
                #save_path_mesh = assembly.name + "\\" + new_part_name + ".stl"
                #print(save_path_mesh)
                #resulting_mesh1.export(save_path_mesh)
                new_part = Part(new_part_name, "", mesh=operation.data["resulting_mesh"])
                assembly_work_copy.add_part(new_part)
                for part in assembly_work_copy.parts.values():
                    if part != new_part:
                        assembly_work_copy.check_pair_for_contact_new([new_part, part])
            assembly_work_copy.remove_connection(connection)
            if part2.mesh is None:
                assembly_work_copy.remove_part(part2)
            else:
                assembly_work_copy.rename_single_part(part2, part2.name + "_milled_" + part1.name.split("_milled_")[0])
                parts_to_recheck = [part1]
                existing_connections = assembly_work_copy.get_connections_by_part_name(part2.name)
                #con_befor = len(assembly_work_copy.connections)
                screw_conntections_to_recheck = []
                for ex_con in existing_connections:
                    if ex_con.connection_type == "geometric":
                        if ex_con.part_name1 == part2.name:
                            parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name2))
                        else:
                            parts_to_recheck.append(assembly_work_copy.get_part_by_name(ex_con.part_name1))
                        assembly_work_copy.remove_connection(ex_con)

                for part in parts_to_recheck:
                    assembly_work_copy.check_pair_for_contact_new([part2, part])
            # assembly_work_copy.get_connections_by_part_name(part2.name
                #print(f"{len(assembly_work_copy.connections)}\{con_befor} connections exist")
                #part2.name = part2.name + "_milled_" + part1.name
                #assembly_work_copy.check_pair_for_contact([part1, part2])

            for sc in sc_to_recheck:
                connections = assembly_work_copy.get_connections_by_part_name(sc.part_name2)
                gc_part = [conn for conn in connections if conn.connection_type == 'geometric']
                if sc.part_name1 == part2.name and operation.data["resulting_mesh"] is not None: #milled part was the screw
                    connections = assembly_work_copy.get_connections_by_part_name(new_part.name)
                    gc_new_part = [conn for conn in connections if conn.connection_type == 'geometric']
                    common_elements = [item for item in gc_new_part if item in gc_part]
                    if len(common_elements)>0: #there is a geometric connection beteween the new part and the part the screw was screwed into
                        sc.part_name1 = new_part.name
                else:
                    connections = assembly_work_copy.get_connections_by_part_name(sc.part_name1)
                    gc_screw = [conn for conn in connections if conn.connection_type == 'geometric']
                    common_elements = [item for item in gc_screw if item in gc_part]
                    if len(common_elements)==0 and operation.data["resulting_mesh"] is not None:
                        sc.part_name2=new_part.name #TODO: maybe check that there actually is a connection


            operation.data["cost"]=cost_milling
            assembly_work_copy.check_geometric_connections()
            return [operation, assembly_work_copy]
        else:
            operation.data["cost"]=-1
            return [operation, assembly_work_copy]
    @staticmethod
    def wrapper_function_operation_application(operation, assembly):
        try:
            t_cost, t_assembly = operation.apply_to_assembly(assembly)
        except:
            t_cost = -1
            t_assembly = None
            print("error in application to assembly")
        return t_cost, t_assembly
    @staticmethod
    def static_explore_node(node_id, assembly_state, cost_gripping_and_removing, cost_screwing, cost_milling, cost_safety_ring, mill_diameter,
                            calculated_milling_operations, available_screwdrivers):
        print(f"exploring node {node_id} on process {os.getpid()}")
        grippable = False
        exploration_results = []
        new_milling_operations = []
        if cost_gripping_and_removing>=0:
            for part in assembly_state.parts.values():
                operation, new_assembly = HierarchicalDisassemblySequencePlanner.plan_gripping_and_removing(assembly_state, part, cost_gripping_and_removing)
                if operation.data["cost"]>=0:
                    #planner.update_graph_with_operation(operation, new_assembly, node_id)
                    exploration_results.append([operation, new_assembly, node_id])
                    grippable=True
                else:
                    print(f"part {part.name} not grippable")

                   # new_assembly.show()
        if cost_screwing>=0:
            for sc in assembly_state.connections:
                if sc.connection_type == "screw" and sc.screw_head in available_screwdrivers:
                    operation, new_assembly = HierarchicalDisassemblySequencePlanner.plan_screwing(assembly_state, sc, cost_screwing)
                    if operation.data["cost"]>=0:
                        #planner.update_graph_with_operation(operation, new_assembly, node_id)
                        exploration_results.append([operation, new_assembly, node_id])
                       # new_assembly.show()
        if cost_safety_ring>=0:
            connections = assembly_state.connections
            sr_connections = [conn for conn in connections if conn.connection_type == 'safety_ring']
            for src in sr_connections:
                operation, new_assembly = HierarchicalDisassemblySequencePlanner.plan_safety_ring_disassembly(assembly_state, src, cost_safety_ring)
                if operation.data["cost"]>=0:
                    exploration_results.append([operation, new_assembly, node_id])
                    #self.update_graph_with_operation(operation, new_assembly, node_id)

        if not grippable and cost_milling>=0:
            datasets = []
            results = []
            datasets_apply = []
            for gc in assembly_state.connections:
                if gc.connection_type == "geometric":
                    found_inverted = False
                    found_non_inverted = False
                    for op in calculated_milling_operations:
                        if op.data["connection"]==gc:
                            if op.data["inverted"]:
                                found_inverted=True
                            else:
                                found_non_inverted=True
                            t_cost, t_assembly = op.apply_to_assembly(assembly_state, cost_milling)
                            if t_cost>=0:
                                op.data["cost"] = t_cost
                                exploration_results.append([op, t_assembly, node_id])

                    if not found_non_inverted:
                        results.append(HierarchicalDisassemblySequencePlanner.plan_milling(assembly_state, gc, mill_diameter, cost_milling, False))
                    if not found_inverted:
                        results.append(HierarchicalDisassemblySequencePlanner.plan_milling(assembly_state, gc, mill_diameter, cost_milling, True))
            for res in results:
                if res is not None:
                    if res[0].data["cost"]>=0:
                        #planner.update_graph_with_operation(res[0], res[1], node_id)
                        exploration_results.append([res[0], res[1], node_id])
                    if res[0] not in calculated_milling_operations:
                        #planner.calculated_milling_operations.append(res[0])
                        new_milling_operations.append(res[0])
                        #scene = res[0].scene()
                        #scene.show()
        print(f"finished exploring node {node_id}")
        return [exploration_results, new_milling_operations]

def find_shortest_path(graph, assembly, mode="closest_com"):#TODO: Remove, not used and not working anymore!!!
    if len(graph.nodes) == 0:
        print("empty graph")
        return []

    # define a function to calculate edge weights based on the distance of the removed parts of the nodes
    def weight_func_com(start_hash, end_hash, dict_edge):
        pred = list(graph.predecessors(assembly.generate_hashID()))
        end_attr = graph.get_edge_data(start_hash, end_hash)
        if pred:
            min_pred = min(pred, key=lambda node: graph.nodes[node].get("cost", float("inf")))
            pred_attr = graph.get_edge_data(min_pred, start_hash)
            cost = np.linalg.norm(pred_attr["removed_parts_com"] -
                                  end_attr["removed_parts_com"])
        else:
            cost = np.linalg.norm([0.0, 0.0, 0.0] -
                                  end_attr["removed_parts_com"])
        return cost

    # find the last node in the graph
    last_node = None
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            last_node = node
    # calculate the shortest path with dijkstra_shortest_paths using the weight_func
    if mode == "closest_com":
        return nx.dijkstra_path(graph, assembly.generate_hashID(), last_node, weight=weight_func_com)
    else:
        print("mode not recognized")
        return []

def calculate_max_steps(G):
    max_lengths = {node: 0 for node in G.nodes()}

    # Function to perform BFS and update max_lengths
    def bfs_update_lengths(start_node):
        visited = set()
        queue = deque([(start_node, 0)])  # (node, distance)

        while queue:
            current_node, distance = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            # Update maximum length to this node
            if distance > max_lengths[current_node]:
                max_lengths[current_node] = distance

            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    # Perform BFS from each node
    for node in G.nodes():
        bfs_update_lengths(node)
        G.nodes[node]["layer"] = max_lengths[node]

    return max_lengths

def save_graph_with_timestamp(graph, relative_path=""):
    fig = plt.figure(figsize=(8, 8))
    calculate_max_steps(graph)
    pos = graph_layout(graph)
    #pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
    # flip y axis
    #for key, values in pos.items():
    #    pos[key] = [-1 * value for value in values]
    nx.draw_networkx_nodes(graph, pos, node_size=70, node_color="white", edgecolors='black')
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=8)

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

    plt.savefig(relative_path + f'{timestamp}.png')
    plt.close(fig)

if __name__ == '__main__':
    if __name__ == '__main__':
        file = resolve_path("EMO_SR_V2.xml")
        path_to_working_dir = resolve_path("crc_disassembly_sim")
        asm = Assembly.load_from_xml(file)
        planner = HierarchicalDisassemblySequencePlanner(asm, cost_milling=5, cost_screwing=-1,
            cost_gripping_and_removing=-1, mill_diameter=-1, path_to_working_dir=path_to_working_dir,
            available_screwdrivers=["H5", "T6"])

        for con in asm.connections:
            if con.connection_type == "geometric":
                op, asmres = planner.plan_milling(asm, con)
        op, asmres = planner.plan_milling(asm, asm.connections[2])
        op.scene(asm).show()
        fig = asmres.generate_surface_plot()
        plt.show(block=True)
        part = asmres.get_part_by_name("schraube_v_1_milled_endcap")
        DOF_part = asmres.get_part_dof(part)
        #asmres.save_part_meshes(r"C:\MaH\Data\Rubbish\anlasser")
        op_grip, asmgrip = planner.plan_gripping_and_removing(asmres,asmres.get_part_by_name("schraube_v_1_milled_endcap"), 1)
        #op, asmres = planner.plan_milling(asm, asm.connections[0], invert=True)
        #op.scene(asm).show()
        print("loaded")