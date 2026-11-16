import copy
import math
import time
import os
os.environ.setdefault('DISPLAY', ':0')

import pybullet as p
import pybullet_data
import pybullet_industrial as pi
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import sim_server_protocol
import cell_simulation
import straight_line_planner
import planner_worker_client
import pickle
import argparse

from PIL import Image
import io

import configparser

import open3d as o3d
from opcua import Client
from opcua import ua

import threading

from Classes.depth_camera import RGBDCamera
from methodes.g_code_processor import GCodeProcessor
from Classes.AssemblyPB import AssemblyPB
from helper_workarounds import apply_tool_path_rotation, invert_toolpath, get_all_active_constraint_ids

import tkinter as tk
from tkinter import simpledialog

from pathlib import Path
import re as _re
import contextlib
base_path = Path(__file__).parent.parent.resolve()
def resolve_path(relative_path):
    """Resolve a relative path to an absolute path."""
    return (base_path / relative_path).resolve()

def _win_to_linux_path(path: str) -> str:
    """Convert a path sent by the Windows-side HMI into one this WSL process can open.

      \\\\wsl.localhost\\Ubuntu\\home\\...  or  //wsl.localhost/Ubuntu/home/...  -> /home/...
      Z:\\home\\...   (the documented `net use Z: \\\\wsl.localhost\\Ubuntu` mapping)  -> /home/...
      C:\\Users\\...  (a real Windows drive)                                         -> /mnt/c/Users/...
    Paths that are already Linux-style pass through unchanged.
    """
    p = str(path).replace('\\', '/')
    m = _re.match(r'^//wsl(?:\.localhost|\$)?/[^/]+(/.*)', p)
    if m:
        return m.group(1)
    m = _re.match(r'^([A-Za-z]):(/.*)', p)
    if m:
        drive, rest = m.group(1).lower(), m.group(2)
        return rest if drive == 'z' else f'/mnt/{drive}{rest}'
    return p

gcode_path = resolve_path(r"submodule_crc_disassembly_sequence_generator\Data\gcodes")
print(f"gcode_path: {gcode_path}")

# Name of the screwdriver tool data on the Sinumerik.  Must exist in the tool
# table of EVERY channel that selects it — for a ghost run on channel 1 it is the
# tool DATA that matters (length offset), not the physical tool.
SCREWDRIVER_TOOL_NAME = "SCREWDRIVER"

# Tool yaw over the screw box.  G-code A/B/C are RADIANS throughout the sim
# (write_g_code converts with math.degrees); this used to be written as -90.0,
# i.e. -90 rad, which wrapped to -116.62 deg on export.
DROP_YAW_RAD = -math.pi / 2
# Exported joint-space lines (RA1..RA6, TRAFOOF): F is a rotary feed on the
# controller, so the simulation's Cartesian F (mm/min) is scaled down for them.
JOINT_FEED_EXPORT_SCALE = 0.1
# Tool data record on the controller for the arm WITHOUT a physical
# screwdriver (robot_C): same TCP geometry, nothing mounted.
GHOST_SCREWDRIVER_TOOL_NAME = "GHOST_SCREWDRIVER"
# Dress-pack cables around the wrist: axis 4 and axis 6 each within +-360 deg
# and their sum within +-360 deg.  Planned moves respect a tighter limit (see
# CuMotionWaypointPlanner.wrist_limit_rad, 3.25 rad); the exporter checks every line.
WRIST_CABLE_LIMIT_DEG = 360.0
# ... but the SUM |A4 + A6| is what visibly strains the dress pack: 250 deg is
# the absolute maximum the user is willing to run (2026-09-18; Run 4's C wait
# pose sat at A4 -101 / A6 -127 = -228 and already looked forced).  Enforced
# as a coupled limit in the planner (ik + paths), weighted in the roll picker
# and checked by the exporter.
WRIST_CABLE_SUM_LIMIT_DEG = 250.0
# Programmed stops (M0) in the exported programs.  The cell refused M0 while
# both channels run ("M0 cannot be used when running programs on two channels
# at the same time"); the stops were operator checkpoints from single-arm days
# and play no part in the coordination (WAITM does that).  Off = every M0,
# standalone or inside an M-translation, is written as a comment.
EXPORT_PROGRAM_STOPS = True
NL = chr(10)
# NJ290 parallelogram linkage between axes 2 and 3: the forearm angle is
# limited RELATIVE to the upper arm, i.e. a coupled limit  lo <= A3 - k*A2 <= hi
# (degrees).  Not modelled in the URDF/planner; fill in from the Comau
# working-area sheet (Comau_NJ29030_Workingareas).  None = no check.
# From the Comau drawing (verified by FK against its boundary-pose table):
#   * Comau Ax.3 is ABSOLUTE (forearm angle w.r.t. the base), URDF q3 is
#     relative to the upper arm:  Ax.3 = q3 - q2.
#   * Ax.3 range [-220, 0] deg -> coupled limit  -220 <= A3 - 1.0*A2 <= 0.
#   * The relative angle itself is limited to q3 >= -156 deg (parallelogram
#     bar), now in the URDFs.
PARALLELOGRAM_LIMIT = {'k': 1.0, 'lo': -220.0, 'hi': 0.0}
# Exported RA3 word uses Comau's absolute convention (RA3 = q3 - q2).
COMAU_AXIS3_ABSOLUTE = True

# Sim M-/T-codes -> Sinumerik program text.  Anything not listed is written
# raw ("T=12"), which on the controller SELECTS MAGAZINE TOOL 12 — so a channel
# must never be written without a table that covers every code it can emit.
M_TRANSLATION = {
    10: "RESET_GRIPPER() \nM0 \nGRIPPER_ABS_POS(75000,10000) ;program to close egu_gripper ",
    11: "RESET_GRIPPER() \nGRIPPER_ABS_POS(94000,10000) ;program to open egu_gripper",
    20: "M0 ;DEPRAG_UNSCREW(1,TRUE)\n ;DEPRAG_UNSCREW(2, TRUE)\n ;DEPRAG_UNSCREW(2, FALSE)",
    21: "M0 ; Part dropoff point",
    30: "M0 ;BFG_POS(-2000000)",
    31: "M0 ;'BFG_MATRIX_OPEN\nBFG_POS(-1250000)\nBFG_MATRIX_CLOSE'",
    40: "WAIT_FOR_GO",
    50: "M0 ; CALL MILL_ON.MPF",
    51: "M0 ; CALL MILL_OFF.MPF",
}
T_TRANSLATION = {
    1:  'CALL "STORE_SCREWDRIVER.MPF" \n STOPRE',
    2:  'CALL "GET_SCREWDRIVER" \n STOPRE \n;T="SCREWDRIVER" M6',
    3:  'CALL "STORE_EGU_GRIPPER" \n STOPRE',
    4:  'CALL "GET_EGU_GRIPPER" \n STOPRE',
    5:  "M0, 'CALL 013_STORE_BFG.MPF",
    6:  "M0; 'CALL 013_GET_BFG.MPF",
    7:  "M0;'CALL 011_STORE_MILL.MPF",
    8:  "M0 ;CALL 011_GET_MILL.MPF",
    11: 'CALL "SCREW_PROGRAM.MPF"',
    12: 'CALL "SEARCH_TURN_SCREWDRIVER.MPF"',
    13: 'CALL "GET_EGU_GRIPPER_TR" \n STOPRE',
    18: 'CALL "ZWISCHENSTELLUNG_WW.MPF" \n STOPRE',
}


def ghost_translation(table: dict, keep: set = frozenset()) -> dict:
    """Same keys, but every entry (except `keep`) becomes a comment: the motion
    stays, the physical tool interaction is skipped."""
    out = {}
    for k, v in table.items():
        out[k] = v if k in keep else f"; ghost: skipped [{v.splitlines()[0].strip()}]"
    return out

def limit_angle(angle):
    """
    Limit the angle to a range of [-179.999, 179.999] degrees.

    Parameters:
    - angle: The input angle in degrees.

    Returns:
    - The angle limited to the range [-179.999, 179.999] degrees.
    """
    # Normalize angle to be within [-360, 360]
    angle = angle % 360

    # Adjust angles greater than 180 to be within [-179.999, 179.999]
    if angle > 179.999:
        angle -= 360
    elif angle < -179.999:
        angle += 360
    if angle == 180.0:
        angle = 179.999
    elif angle == -180.0:
        angle = -179.999
    return angle


def coop_grasp_frame(pos, z_dir, roll):
    """4x4 flange pose at `pos` whose z axis points along `z_dir`, rolled by `roll` about it."""
    z = np.asarray(z_dir, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) < 1e-6 else x / np.linalg.norm(x)
    c, s_ = math.cos(roll), math.sin(roll)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z]) @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    T[:3, 3] = pos
    return T


def vector_to_quaternion(v):
    # Normalize the input vector
    v_normalized = v / np.linalg.norm(v)

    # Global z-axis
    z_global = np.array([0, 0, 1])

    # Rotation axis (cross product of v and z_global)
    axis = np.cross(v_normalized, z_global)

    # Calculate the cosine of the angle using dot product
    cos_theta = np.dot(v_normalized, z_global)

    # Calculate the angle
    angle = np.arccos(cos_theta)

    # Normalize the rotation axis
    axis_normalized = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else np.array([1,0,0])

    # Quaternion components
    w = np.cos(angle / 2)
    x = axis_normalized[0] * np.sin(angle / 2)
    y = axis_normalized[1] * np.sin(angle / 2)
    z = axis_normalized[2] * np.sin(angle / 2)

    return [x, y, z, w]


def quaternion_to_allign_z(v, current_orn):
    z = np.array([0, 0, 1])
    z, _ = p.multiplyTransforms([0,0,0], current_orn, z, [0,0,0,1])
    v = v / np.linalg.norm(v)  # Normalize v
    axis = np.cross(z, v)
    angle = np.arccos(max(min((np.dot(z, v), 1.0)), -1.0))
    print(f"z: {z}, v: {v}, angle: {np.degrees(angle)}")
    if np.linalg.norm(axis) <= 0.00000001:  # v is parallel or anti-parallel to z
        # Handle the special case
        if np.allclose(v, z):
            return np.array([0, 0, 0, 1])  # No rotation
        else:
            # 180 degrees rotation around any horizontal axis
            # Here we choose x-axis for simplicity
            return np.array([1, 0, 0, 0])

    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    q = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
    return q


def direction_to_rpy(direction):
    # Calculate yaw and pitch
    yaw = math.atan2(direction[1], direction[0])
    pitch = math.atan2(math.sqrt(direction[0]**2 + direction[1]**2),direction[2])
    roll = 0
    return [roll, pitch, yaw]
#{"M": {0: "Funktionxyz"}, "T": {1: 'T="greifer"'}}

def write_g_code(g_code: list, textfile: str, prefix: str = "", postfix: str = "",
                 m_translation: dict = None, t_translation : dict = None):
    mode = None            # 'cart' (TRAORI active) | 'joint' (TRAFOOF) | None
    cable_warnings = []
    with open(textfile, 'w') as file:
        file.write(prefix)
        for n_line, command in enumerate(g_code):
            is_joint = any(k.startswith('RA') for k in command)
            is_cart = any(k in command for k in ('X', 'Y', 'Z', 'A', 'B', 'C'))
            # Sinumerik: Cartesian words need the orientation transformation
            # (TRAORI); axis words need it OFF (TRAFOOF).  Switch on change.
            if is_joint and mode != 'joint':
                file.write('TRAFOOF\n')
                mode = 'joint'
            elif is_cart and not is_joint and mode != 'cart':
                file.write('TRAORI\n')
                mode = 'cart'
            if is_joint and PARALLELOGRAM_LIMIT:
                a2, a3 = math.degrees(command.get('RA2', 0.0)), math.degrees(command.get('RA3', 0.0))
                v = a3 - PARALLELOGRAM_LIMIT['k'] * a2
                if not (PARALLELOGRAM_LIMIT['lo'] <= v <= PARALLELOGRAM_LIMIT['hi']):
                    cable_warnings.append((n_line, f'A3-k*A2={v:+.0f}', 'parallelogram'))
            if is_joint:
                a4, a6 = math.degrees(command.get('RA4', 0.0)), math.degrees(command.get('RA6', 0.0))
                if abs(a4) > WRIST_CABLE_LIMIT_DEG or abs(a6) > WRIST_CABLE_LIMIT_DEG \
                        or abs(a4 + a6) > WRIST_CABLE_SUM_LIMIT_DEG:
                    cable_warnings.append((n_line, a4, a6))
            if 'comment' in command:
                file.write(f"; ---- {command['comment']}\n")
                continue
            if 'WAITM' in command:
                # Channel rendezvous: both channels block here until both have
                # reached this marker.  This is how the simulation's holds and
                # escapes (which are timing, not motion) reach the real cell.
                # The SYNC ordinal is UNIVERSAL: the same number appears at the
                # same rendezvous in both programs (the WAITM marker number
                # only cycles 1..sync_marker_count), so "SYNC 12" names the
                # same level of the task graph in either file.
                if 'seq' in command:
                    tot = f"/{command['total']}" if command.get('total') else ''
                    file.write(f"; ==== SYNC {command['seq']}{tot}: {command.get('why', '')}\n")
                file.write(f"WAITM({command['WAITM']},1,2) ; sync {command.get('why', '')}\n")
                continue
            order = ['G', 'X', 'Y', 'Z', 'A', 'B', 'C', 'R', 'RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6',
                     'M', 'F', 'T']
            for key in order:
                #print(f"command: {command}")
                value = command.get(key)
                if value is not None:
                    if key in {'G'}:
                        if value == 0:
                            file.write("G1 ")
                        else:
                            file.write(f'{key}{value} ')
                    elif key in {'X', 'Y', 'Z'}:
                        file.write(f'{key}{value*1000:.3f} ')
                    elif key in {'A', 'B', 'C', 'R'}:
                        file.write(f'{key}{limit_angle(math.degrees(value)):.3f} ')
                    elif key.startswith('RA'):
                        v = value
                        if key == 'RA3' and COMAU_AXIS3_ABSOLUTE:
                            v = value - command.get('RA2', 0.0)      # Comau Ax.3 is absolute
                        file.write(f'{key}={math.degrees(v):.3f} ')   # robot axis, degrees
                    elif key == 'F' and is_joint:
                        file.write(f'F{value * JOINT_FEED_EXPORT_SCALE:.1f} ')   # rotary feed, scaled
                    elif key in {'T'}:
                        if value == 0 or t_translation is None or value not in t_translation:
                            file.write(f'{key}={value}')
                        else:
                            file.write(t_translation[value])
                    elif key in {'M'}:
                        if value == 0 and not EXPORT_PROGRAM_STOPS:
                            file.write('; M0 (programmed stop suppressed: multi-channel run)')
                        elif m_translation is None or value not in m_translation:
                            file.write(f'{key}{value}')
                        else:
                            text = m_translation[value]
                            if not EXPORT_PROGRAM_STOPS:
                                # drop a leading/embedded M0 token, keep the rest of the macro
                                text = NL.join(
                                    (';M0 suppressed ' + ln.strip()[2:].strip())
                                    if (ln.strip() == 'M0' or ln.strip().startswith('M0 ') or ln.strip().startswith('M0,'))
                                    else ln
                                    for ln in text.split(NL))
                            if command.get('F') is not None:
                                file.write(f"{text}{command.get('F')})")
                                break
                            file.write(text)
                    else:
                        file.write(f'{key}{value} ')
            file.write('\n')
        if mode == 'joint':
            file.write('TRAORI\n')          # leave the channel in Cartesian mode
        file.write(postfix)
    if cable_warnings:
        first = cable_warnings[0]
        detail = (f'A4 {first[1]:+.0f} A6 {first[2]:+.0f}' if isinstance(first[1], float)
                  else f'{first[1]} ({first[2]})')
        print(f'[write_g_code] WARNING {os.path.basename(textfile)}: {len(cable_warnings)} joint lines exceed an '
              f'axis limit (wrist cables +-{WRIST_CABLE_LIMIT_DEG:.0f} deg on A4, A6; '
              f'+-{WRIST_CABLE_SUM_LIMIT_DEG:.0f} deg on A4+A6'
              f'{"; parallelogram A3-k*A2" if PARALLELOGRAM_LIMIT else ""}), e.g. line {first[0]}: {detail}',
              flush=True)

def transform_eulers_in_gcode(g_code: list):
    return_code = copy.deepcopy(g_code)
    for command in return_code:
        if 'A' in command and 'B' in command and 'C' in command:
            euler_pb = np.array([command['A'], command['B'], command['C']])
            rot = R.from_euler('xyz', euler_pb)
            euler_siemens = rot.as_euler('XYZ')
            command['A'] = euler_siemens[0]
            command['B'] = euler_siemens[1]
            command['C'] = euler_siemens[2]
    return return_code


'''def create_box_container(base_position, base_orientation=(0, 0, 0, 1), length=1.0, width=1.0, height=1.0, thickness=0.1,
                         color=(1, 0, 0, 1)):
    """
    Erstellt eine Kiste aus Box-Geometrien mit einer einstellbaren Farbe.

    :param base_position: Position der Mitte des Bodens (x, y, z)
    :param base_orientation: Orientierung als Quaternion (x, y, z, w)
    :param length: Länge der Kiste
    :param width: Breite der Kiste
    :param height: Höhe der Kiste
    :param thickness: Wandstärke
    :param color: Farbe der Kiste im RGBA-Format (z.B. (1, 0, 0, 1) für Rot)
    :return: Körper-ID des erstellten MultiBody-Objekts
    """
    # Setze physikalisches PyBullet-Modul
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Definition der Box-Größen (halbe Größen für PyBullet)
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2
    half_thickness = thickness / 2

    # Boxen für die Wände und den Boden
    # Jede Box wird relativ zur Basis positioniert
    box_geometries = [
        {"halfExtents": [half_length, half_thickness, half_height],
         "pos": [0, -half_width - half_thickness, half_height]},
        {"halfExtents": [half_length, half_thickness, half_height],
         "pos": [0, half_width + half_thickness, half_height]},
        {"halfExtents": [half_thickness, half_width, half_height],
         "pos": [-half_length - half_thickness, 0, half_height]},
        {"halfExtents": [half_thickness, half_width, half_height],
         "pos": [half_length + half_thickness, 0, half_height]},
        {"halfExtents": [half_length, half_width, half_thickness], "pos": [0, 0, half_thickness]},
    ]

    # IDs für visuelle und Kollisionsformen
    visual_shapes = []
    collision_shapes = []

    for box in box_geometries:
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=box["halfExtents"],
            rgbaColor=color
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=box["halfExtents"]
        )
        visual_shapes.append(visual_shape)
        collision_shapes.append(collision_shape)

    # MultiBody erstellen
    body_id = p.createMultiBody(
        baseMass=0,  # Masse 0 -> statisch
        baseCollisionShapeIndex=-1,  # Keine Basis-Kollisionsform
        baseVisualShapeIndex=-1,  # Keine Basis-Visualisierung
        basePosition=base_position,
        baseOrientation=base_orientation,
        linkMasses=[0] * len(box_geometries),
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=visual_shapes,
        linkPositions=[box["pos"] for box in box_geometries],
        linkOrientations=[[0, 0, 0, 1]] * len(box_geometries),
        linkInertialFramePositions=[[0, 0, 0]] * len(box_geometries),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(box_geometries),
        linkParentIndices=[0] * len(box_geometries),
        linkJointTypes=[p.JOINT_FIXED] * len(box_geometries),
        linkJointAxis=[[0, 0, 0]] * len(box_geometries)
    )

    return body_id'''
class WSKAgiloDrive:
    def __init__(self, server_active = False, pb_examples_path = "C:\MaH\pybullet_industrial\examples", activate_shadow=False):
        self.activate_shadow = activate_shadow
        self.storage_spaces = []
        self.assembly = AssemblyPB("Empty")
        self._assembly_loaded = None    # (file, mounting_mode) of the loaded assembly; duplicate-request guard
        self.fixture = None           # PyBullet body id of the loaded clamping fixture
        self.fixture_constraint = None  # constraint pinning assembly base to the fixture

        # Initialize the configparser
        self.config = configparser.ConfigParser()
        self.screwdriver_config = configparser.ConfigParser()

        # Read the config file
        self.config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))
        self.screwdriver_config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Screwdriver_config.ini'))

        #check wether spaen_point.npy is in the same directory as the file
        #if not use the hardcoded spawn_point
        _sim_dir = os.path.dirname(os.path.abspath(__file__))
        _sp_file = self.config['Settings']['spawn_point_file']
        _so_file = self.config['Settings']['spawn_orient_file']
        if os.path.exists(os.path.join(_sim_dir, _sp_file)):
            print(f'loading {_sp_file}')
            self.spawn_point = np.load(os.path.join(_sim_dir, _sp_file))
        else:
            self.spawn_point = np.array([-1, -1.3, 0.43032172322273254])

        if os.path.exists(os.path.join(_sim_dir, _so_file)):
            print(f'loading {_so_file}')
            _raw = np.load(os.path.join(_sim_dir, _so_file))
            # Files may be Euler (3-elem) or quaternion (4-elem).
            self.spawn_orient = tuple(p.getQuaternionFromEuler(_raw)) if len(_raw) == 3 else tuple(_raw)
        else:
            self.spawn_orient = p.getQuaternionFromEuler([np.pi, 0, 0])

        self.loading_point = [0.0, 0.5, 0.4]
        self.loading_orient = p.getQuaternionFromEuler(np.array([0, 0, 0]))


        self.work_point_C = [-1,-1.6,2] # neuer Position für Bearbeitung
        self.work_orient_C = p.getQuaternionFromEuler([-np.pi/2, np.pi, +np.pi/4]) # fuer Bearbeitung. ausgehend vom WKS
        self.save_point_C = [0,-1.5,2.0] # neuer Position für sicheren Zustand
        self.save_orient_C = p.getQuaternionFromEuler([0,0,0])
        #self.save_orient_C = p.getQuaternionFromEuler([0,0,0])
        self.save_point_D = [-1.99,-2,2.0]
        self.save_orient_D = p.getQuaternionFromEuler([0,0,0]) 

        # Define the camera's viewpoint
        camera_position = [0.5, -2.5, 1.5]#[1.25/2, -2.0/2, 1.8/2]  # Camera position in world coordinates
        camera_target_position = self.spawn_point#[0, 0, 0]  # Where the camera is looking
        camera_up_vector = [0, 0, 1]  # "Up" direction
        # Compute the view matrix
        self.camera_view_matrix = p.computeViewMatrix(camera_position, camera_target_position, camera_up_vector)
        # Define the camera's projection properties
        fov = 70  # Field of view (in degrees)
        aspect_ratio = 1.0  # Aspect ratio (width/height)
        near_plane = 0.1  # Near clipping plane
        far_plane = 100.0  # Far clipping plane
        self.camera_projection_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near_plane, far_plane)
        # define the camera's resolution
        self.camera_width, self.camera_height = 800, 800  # Desired resolution





        working_dir = os.path.dirname(__file__)

        # Initilizing urdf files
        urdf_comau= os.path.join(working_dir,
                                  'robot_descriptions', 'comau_nj290_robotNC.urdf')
        urdf_comau_shadow= os.path.join(working_dir,
                                  'robot_descriptions', 'comau_nj290_robotNC_Shadow.urdf')
        urdf_comau_sensor= os.path.join(working_dir,
                                  'robot_descriptions', 'comau_nj290_robotNC_sensor.urdf')
        urdf_comau_sensor_shadow= os.path.join(working_dir,
                                  'robot_descriptions', 'comau_nj290_robotNC_sensor_Shadow.urdf')
        urdf_screwdriver = os.path.join(working_dir,
                                  'robot_descriptions', 'screwDriver.urdf')
        urdf_millingtool = os.path.join(working_dir,
                                  'robot_descriptions', 'millingtool.urdf')
        urdf_DLG = os.path.join(working_dir,
                                 'robot_descriptions', 'DLG.urdf')
        urdf_store_box = os.path.join(working_dir,
                                      'Objects', 'Box', 'stor_box.urdf')
        urdf_tool_storage_robot1 = os.path.join(working_dir, 'transformer_cell', 'Objects', 'ToolStorage', 'ToolStorage.urdf')
        urdf_tool_storage_robot2 = os.path.join(working_dir, 'transformer_cell', 'Objects', 'ToolStorage', 'ToolStorage.urdf')
        urdf_zivid = os.path.join(working_dir,
                                 'robot_descriptions', 'zivid.urdf')
        urdf_faga = os.path.join(working_dir,'transformer_cell', 'Objects', 'InRePro', 'InRePro.urdf') # Umgebung
        dmg_path = os.path.join(working_dir, 'transformer_cell', 'Objects','DMG', 'DMG.urdf')
        ponticon_path = os.path.join(working_dir, 'transformer_cell', 'Objects','ponticon', 'ponticon.urdf')
        table_path = os.path.join(working_dir, 'transformer_cell', 'Objects', 'modules', 'table.urdf')
        clamping_desk_path = os.path.join(working_dir, 'fixtures', 'Spanntisch', 'spanntisch.urdf')
        urdf_robofixture = os.path.join(working_dir,'robot_descriptions', 'robofixture.urdf') #Assemblyhalterung am Roboter
        urdf_SRG = os.path.join(working_dir, 'robot_descriptions', 'SRG.urdf') 
        urdf_gripper = os.path.join(working_dir, 'robot_descriptions', 'gripper.urdf') 

                        
        # CRC_SIM_HEADLESS=1 runs the whole cell without a window (test benches,
        # CI): identical physics, no rendering.
        if os.environ.get('CRC_SIM_HEADLESS'):
            self.pysics_client = p.connect(p.DIRECT)
        else:
            self.pysics_client = p.connect(p.GUI, options='--background_color_red=0.95 ' +
                                                     '--background_color_green=0.95 ' +
                                                     '--background_color_blue=0.95 ' +
                                                     '--opengl2') #opengl2 is necessary to sidestep problem of maximal rendered triangles

        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=20.0, #Anpassung der Startansicht für FaGa
                                     cameraPitch=-30,
                                     cameraTargetPosition=np.array([-1, -1.3, 0.5]))
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(numSolverIterations=10000)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF(urdf_faga, [-2.59,-0.12,0], p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]), useFixedBase=True, globalScaling=0.001)
        p.loadURDF(dmg_path, [-2.59,-0.12,0], p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]), useFixedBase=True, globalScaling=0.001)
        p.loadURDF(ponticon_path, [-2.59,-0.12,0], p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]), useFixedBase=True, globalScaling=0.001)
        #p.loadURDF(mm_path, [2.139,-3,0.95], p.getQuaternionFromEuler([0,0,np.pi/2]),useFixedBase=True, globalScaling=0.001,flags= p.GEOM_FORCE_CONCAVE_TRIMESH)
        p.loadURDF(table_path, [2.139, -2.5, 0.95], p.getQuaternionFromEuler([0, 0, np.pi]), useFixedBase=True, globalScaling=0.001)

        # 120x120 clamping desk (Spanntisch) in the middle of the cell, yaw 0 = parallel to
        # robot_C's base.  The mesh origin is a corner, not the centre: the centre sits at
        # local (+0.505, +0.505), so XY is compensated and Z is set from the measured top height.
        self.clamping_desk_centre = np.array([0.21, -2.73])  # world XY of table centre (robot_C base frame)
        self.clamping_desk_top_z = 1.000                     # measured at the cell
        self.table_slab_margin_m = 0.10                      # see _scene_obstacles (horizontal)
        self.table_slab_top_margin_m = 0.05                  # planning slab top above the real top
        _desk_centre_offset = 0.505
        _desk_mesh_top_z = 0.300                             # top face in the mesh's own frame
        self._desk_args = (clamping_desk_path,
                           [self.clamping_desk_centre[0] - _desk_centre_offset,
                            self.clamping_desk_centre[1] - _desk_centre_offset,
                            self.clamping_desk_top_z - _desk_mesh_top_z])
        self._desk_id = p.loadURDF(self._desk_args[0], self._desk_args[1],
                                   p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self._desk_present = True
        tool_storage1_pos = [-2.5, -1, 0]
        tool_storage2_pos = [-0.5, -5, 0] #[-0.5, -4.7, 0]
        self._tool_storage_ids = [
            p.loadURDF(urdf_tool_storage_robot1, tool_storage1_pos, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, globalScaling=0.001),
            p.loadURDF(urdf_tool_storage_robot2, tool_storage2_pos, p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True, globalScaling=0.001)]
        # Opt-in: treat the tool storages as obstacles when the wait-pose wrist roll is chosen
        # (they are not in the planning scene; on the battery the default roll put the cable bar 13-20 mm into one).
        self.wait_avoid_storage = False
        # Opt-in benchmark switch: the tool storages stay visual, contacts with them are not faults.
        self.ignore_storage_contacts = False
        self.plan_storage_obstacles = False      # opt-in: planners see the tool storages as obstacles
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)

        # Robots and endeffectors
        start_pos_robot_C = np.array([0, 0, 0])
        start_orientation_robot_C = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_C = pi.RobotBase(urdf_comau, start_pos_robot_C, start_orientation_robot_C)
        self._urdf_robot_C = urdf_comau            # no sensor stack: planners must use THIS model for C
        self._urdf_robot_D = urdf_comau_sensor

        #start_pos_robot_D = np.array([-3.6353162, -0.6, 0])
        start_pos_robot_D = np.array([-1.99, -3.750, 0])  # measured at cell 02./03.09.2026; was -3.5153162 (235 mm error)
        start_orientation_robot_D = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.robot_D = pi.RobotBase(urdf_comau_sensor, start_pos_robot_D, start_orientation_robot_D)
        # Setting up base position of Robot
        self.robot_C.set_joint_position(
            {'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})#, 'q6': np.deg2rad(90.0)})
        self.robot_D.set_joint_position(
            {'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})#, 'q6': np.deg2rad(90.0)})
        for _ in range(100): # hoch setzten, damit roboter schneller in startposition kommen
            self.stepSimulation()

        self.client = Client("opc.tcp://192.168.0.13:4840")
        # Setting the username and password
        self.client.set_user("OpcUaClient")
        self.client.set_password("MaH_OpcUa")
        if activate_shadow:
            self.client = Client("opc.tcp://192.168.0.11:4840")
            # Setting the username and password
            self.client.set_user("OpcUaClient")
            self.client.set_password("Bau_OpcUa")

            self.robot_C_shadow = pi.RobotBase(urdf_comau_shadow, start_pos_robot_C, start_orientation_robot_C)
            self.robot_D_shadow = pi.RobotBase(urdf_comau_sensor_shadow, start_pos_robot_D, start_orientation_robot_D)
            self.robot_C_shadow.set_joint_position(
                {'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})#, 'q6': np.deg2rad(90.0)})
            self.robot_D_shadow.set_joint_position(
                {'q2': np.deg2rad(0.0), 'q3': np.deg2rad(-140.0), 'q5': np.deg2rad(45.0)})#, 'q6': np.deg2rad(90.0)})
            for _ in range(100):
                self.stepSimulation()
            # Start periodic task in a background thread
            self.shadow_thread = threading.Thread(target=self.shadow_loop, daemon=True)
            self.shadow_thread.start()

            #self.update_shadows()
        for _ in range(100):
            self.stepSimulation()

        self.start_pos_sc = np.array([tool_storage2_pos[0] - 0.2, tool_storage2_pos[1], tool_storage2_pos[2] + 0.9])
        self.start_orientation_sc = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.screwdriver = pi.SuctionGripper(
            urdf_screwdriver, self.start_pos_sc, self.start_orientation_sc,
            suction_links=["tcp"])
        self.sc_gripping_constraint = None

        # Second screwdriver for robot_C — stored at tool_storage1 (robot_C side).
        # Placed at a distinct XY offset from the DLG so they don't overlap in the shelf.
        self.start_pos_sc_C = np.array([tool_storage1_pos[0] + 0.3,
                                        tool_storage1_pos[1],
                                        tool_storage1_pos[2] + 0.9])
        self.start_orientation_sc_C = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self.screwdriver_C = pi.SuctionGripper(
            urdf_screwdriver, self.start_pos_sc_C, self.start_orientation_sc_C,
            suction_links=["tcp"])
        self.sc_C_gripping_constraint = None

        # Dual-arm live arbitration (see _interleave_streams).  The arm without
        # priority always holds inside yield_floor_m; inside yield_margin_m it holds
        # only while the priority arm's tool is moving towards it, so it can creep
        # up to the floor while the other arm is stationary or leaving.
        # Slower, more deliberate interaction: a wide awareness zone in which
        # BOTH arms move at a fraction of their pace, and a floor that keeps a
        # real reaction buffer — planned paths were computed against where the
        # other arm WAS, so a large gap is what keeps them valid.
        self.yield_margin_m = 0.25
        self.yield_floor_m = 0.10
        # "Approaching" = the closest distance between the arms (any link, any
        # tool) shrinks by more than the yielding arm's own step could explain.
        self.closing_eps_m = 0.002    # shrink beyond own step + this counts as closing
        self.own_step_m = 0.01        # one interpolation step of the processor
        # Offline pair safety (Pellegrinelli et al. 2017): when both arms are on
        # PLANNED segments, every configuration of one path is checked against
        # every configuration of the other (sphere model, timing-independent).
        # A pair whose closest approach stays above safe_pair_margin_m is SAFE
        # and runs at nominal pace with no hold / slow-down / escape logic;
        # only the hard-contact fail-safe stays on.  None = yield_margin_m.
        self.safe_pair_margin_m = None
        self._last_transit_joints = None   # joint path of the last _plan_transit_gcode
        # SWITCHES (2026-09-14): plan_tool_offset back ON after fixes 1+2 (per-arm URDFs,
        # free screw roll); pair_safety OFF (superseded by the TPG layer). Old note: both were OFF so the
        # planner behaves exactly as in the last run that completed ("working
        # but wonky").  plan_tool_offset=True makes cuMotion target the flange
        # at target@inv(T_off) (correct in principle, but the first standoff
        # move then found no path for either arm — cause not yet found).
        # pair_safety_enabled=True turns on the offline SAFE/UNSAFE pair
        # classification in _interleave_streams.
        self.plan_tool_offset = True
        self.pair_safety_enabled = False
        # Execution layer for dual-arm batches (see modular_planner/tpg.py):
        #   'reactive' = live yield arbitration (_interleave_streams)
        #   'tpg'      = offline Temporal Plan Graph: solo rehearsal of each
        #                arm, pairwise sphere safety + zone turn-taking as
        #                dependency edges, asynchronous execution, WAITM per edge.
        self.execution_mode = 'reactive'
        self.danger_zone = None            # DangerZone, set when an assembly is loaded
        self.danger_zone_half_m = 0.6      # 1.2 x 1.2 m box, the clamping table's footprint
        self.tpg_margin_m = 0.15           # sphere gap above which a segment pair is SAFE
        self.tpg_rule = 'segment'          # 'segment' (APEX-MR, cycle-ordered) | 'zone' (table turns)
        self.tpg_plan_b = True             # a waiting arm leaves the danger box (border wait pose)
        self.joint_space_transits = True   # planned transits run as RA1..RA6 joint lines (see _plan_transit_gcode)
        self.unscrew_dwell_s = 2.0         # sim-time dwell standing in for the controller's unscrew program
        self.tpg_rehearsal = 'dry'         # 'dry' = kinematic (no physics, seconds) | 'physics' = simulate every move
        # Execution pass: 'dry' replays the rehearsed segments kinematically
        # (joint states set directly, tool bodies moved along, sphere + mesh
        # contact check per tick) -> minutes for 32 screws; 'physics' steps the
        # engine for every op (Run 4: 1.4 h).  The export and the WAITMs are the
        # same: segments are fixed data after rehearsal, waits come from the graph.
        self.tpg_execution = os.environ.get('CRC_TPG_EXECUTION', 'dry')
        self.tpg_use_cache = True          # reuse a rehearsal for identical inputs (tpg_cache/<key>.pkl)
        # CAVEAT (found 2026-09-21): the cache key (_tpg_cache_key) now also hashes the desk/storage
        # scene flags below, but if you add a NEW flag that changes what the planner sees, add it to
        # the key too, or a stale rehearsal from a different scene will be replayed silently.
        self._rehearsing = None            # side being rehearsed (log labelling)
        self.tpg_sample_every = 4          # record spheres every N processor ops in rehearsal (~4-6 cm of motion)
        self.pace_divider = 2         # ALL motion: one 1 cm step every N ticks (2 = half speed)
        self.slow_factor = 3          # inside the margin the priority arm advances every 3rd tick
        # The G-code processor ignores F (one 1 cm step per call).  The loop
        # honours it instead: a line at feed_ref advances at full pace, a line at
        # F advances at F/feed_ref of it — so relative speeds match the cell.
        self.feed_ref = 10000.0
        # Chunked replanning: a planned transit executes only its first N
        # waypoints, then re-plans from where it is (the other arm has moved).
        self.transit_chunk_wps = 10
        self.transit_max_rounds = 12
        # Sync markers exported as WAITM(n,1,2) at every resume; Sinumerik
        # markers are reused cyclically.
        self.sync_marker_count = 9
        # Operation plans from the techstack (simulate / check_feasibility). Its motion planners are
        # registered here and chosen once by a configure_planner request; until one is configured, a
        # plan with paths is reported infeasible, never moved directly (see cell_simulation.py).
        # straight_line_checked (#149): straight lines, each sample placed by IK on the arms of this
        # scene and checked against its bodies; robot_1 is robot_C (NC channel 1), robot_2 is robot_D.
        self.cell_simulation = cell_simulation.CellSimulationService()
        straight_line_planner.register(
            self.cell_simulation,
            {"robot_1": self._straight_line_arm('C'), "robot_2": self._straight_line_arm('D')},
            self.pysics_client, ignore=self._straight_line_ignored)
        # Planner workers (#212): planners that need a GPU or Linux run as their own process and get a
        # scene description per operation (planner_worker_client.py). "planner_worker" takes the worker's
        # address as an option; each entry of [planner_workers] in config.ini registers a worker by name.
        planner_worker_client.register(self.cell_simulation)
        if self.config.has_section('planner_workers'):
            for name, address in self.config.items('planner_workers'):
                planner_worker_client.register(self.cell_simulation, name, address)
        # The yielding arm decelerates continuously from full pace at the margin
        # to a stop at the floor (a stage before any hold/escape is needed).
        # If the same two segments keep colliding despite escapes, the yielding
        # arm is frozen until the other has moved on two segments (serialise).
        self.max_pair_contacts = 4
        # Block resolution: if the priority arm is inside the floor of a HOLDING
        # arm and still closing for this many ticks it is blocked -> first swap
        # priority (let the holder finish), then make the holder escape.
        # An escape retraces the arm's own recent path (known collision-free)
        # before lifting; the other arm waits for it and then re-plans.
        self.block_ticks = 3
        self.escape_retrace_m = 0.30
        self.escape_dz_m = 0.40
        self.max_escapes = 3


        self.start_pos_mill = np.array([tool_storage2_pos[0] + 0.5, tool_storage2_pos[1], tool_storage2_pos[2] + 0.9])
        self.start_orientation_mill = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.millingtool = pi.EndeffectorTool(
            urdf_millingtool, self.start_pos_mill, self.start_orientation_mill)

        self.start_pos_grip = np.array([tool_storage2_pos[0] + 0.1, tool_storage2_pos[1], tool_storage2_pos[2] + 0.9])
        self.start_orientation_grip = p.getQuaternionFromEuler([0, 0, 0])
        #self.start_pos_grip, self.start_orientation_grip = self.robot_D.get_endeffector_pose()
        self.SRG = pi.Gripper(urdf_SRG, self.start_pos_grip, self.start_orientation_grip)
        self.SRG_gripping_constraint = None
         
        self.start_pos_DLG = np.array([tool_storage1_pos[0], tool_storage1_pos[1] + 0.2, tool_storage1_pos[2] + 1.1])
        self.start_orientation_DLG = p.getQuaternionFromEuler([0, 0, np.pi/2])
        #self.start_pos_grip, self.start_orientation_grip = self.robot_D.get_endeffector_pose()
        self.DLG = pi.Gripper(urdf_DLG, self.start_pos_DLG, self.start_orientation_DLG)
        self.DLG_gripping_constraint = None


        pos, orn = self.robot_D.get_endeffector_pose()
        posrf, ornrf = self.robot_C.get_endeffector_pose()
        
        self.start_pos_robofixture =np.array([0, -1.5, 1.95])# self.save_point_C # np.array([posrf[0],posrf[1],posrf[2]]) # np.array([-2, -3.5, 1])
        self.start_orientation_robofixture = p.getQuaternionFromEuler([0, 0, np.pi]) #ornrf
       
        self.robofixture = pi.Gripper(urdf_robofixture, self.start_pos_robofixture, self.start_orientation_robofixture)
        self.robofixture_gripping_constraint = None

        self.store_tool_map = {
            self.screwdriver: self.store_screwdriver,
            self.SRG: self.store_SRG,
            #self.DLG: self.store_DLG,
            #self.BFG: self.store_BFG,
            self.millingtool: self.store_millingtool
        }

        self.unstore_tool_map = {
            self.screwdriver: self.unstore_screwdriver,
            self.SRG: self.unstore_SRG,
            #self.DLG: self.unstore_DLG,
            #self.BFG: self.unstore_BFG,
            self.millingtool: self.unstore_millingtool
        }

        
        camera_parameters = {'width': int(1944), 'height': int(1200), 'fov': 35,
                             'aspect ratio': 754/449, 'near plane distance': 0.2, 'far plane distance': 100}
        self.zivid = RGBDCamera(urdf_zivid, pos, orn, camera_parameters, self.robot_D)
        #self.last_frame_zivid = {}
        #p.loadURDF(os.path.join(working_dir, 'Objects', "Spannplatte.urdf"), np.array([-2, -2.2, 0.0]), useFixedBase=True, globalScaling=0.001)
        # One screw box PER ARM, each on its own side of the table, so the arms
        # never converge on one drop point and never reach across the board to
        # drop.  storage_spaces[0] = robot_C (east box), [1] = robot_D (west box).
        # With a single shared box both arms flew to the identical point after
        # every pair and collided there indefinitely (live run 2026-09-11).
        # robot_D's box SOUTH of the table (0.6 m past the edge, table centre
        # x): 2.2 m east of D's base, an extended reach with free space around
        # it.  The old west position was 1.7 m from D's base, between base and
        # table, reachable only with a folded/flipped wrist (drop roll cost 9.5).
        # Headless comparison WITH the parallelogram limits (2026-09-17, from
        # both south corner screws): south-east 1.0 rad / 12 wps, south-centre
        # 1.15, south-west 2.8 (needed a forearm fold the real robot lacks),
        # SE corner 3.8.  See project_box_placement_study.
        posBox_D = np.array([0.7, -3.8, 0.261])
        # robot_C's box on robot_C's side (north of the table, between its base
        # and the table's NE corner) — the old east position sat against the
        # second side table and was reachable only with a folded wrist.
        posBox_C = np.array([0.9, -1.7, 0.261])
        p.loadURDF(urdf_store_box, posBox_D, useFixedBase=True)
        p.loadURDF(urdf_store_box, posBox_C, useFixedBase=True)
        self.storage_spaces.append({"droppoint": np.array([posBox_C[0]+0.1, posBox_C[1]+0.1, 0.6]),
                                    "approachpoint": np.array([posBox_C[0]+0.1, posBox_C[1]+0.1, 0.8])})
        # drop height 0.425 -> 0.6 (2026-09-18): at the SE box 0.425 needs Ax.2 > 75 deg (Run 8,
        # "RA2 Softwareendschalter" at the first drop); the drop point is also reachability-checked
        # in _screw_pullout_gcode and raised if needed.
        self.storage_spaces.append({"droppoint": np.array([posBox_D[0]+0.1, posBox_D[1]+0.1, 0.6]),
                                    "approachpoint": np.array([posBox_D[0]+0.1, posBox_D[1]+0.1, 0.8])})


        self.SRG.actuate(0)
        self.DLG.actuate(0)
        self.robofixture.actuate(1)

        self.robot_C.set_endeffector_pose(self.save_point_C, self.save_orient_C)
        self.robot_D.set_endeffector_pose(self.save_point_D, self.save_orient_D, endeffector_name="tcp_frame")
        for _ in range(100):
            self.stepSimulation()

        #self.pos_BFG, self.orn_BFG = self.BFG.get_tool_pose("tcp")
        self.pos_grip, self.orn_grip = self.SRG.get_tool_pose("tcp")
        self.pos_sc, self.orn_sc   = self.screwdriver.get_tool_pose("tcp")
        self.pos_sc_C, self.orn_sc_C = self.screwdriver_C.get_tool_pose("tcp")
        #self.draw_coordinate_frame(self.pos_sc, self.orn_sc)
        self.pos_mill, self.orn_mill = self.millingtool.get_tool_pose("tcp")
        self.pos_robofixture, self.orn_robofixture = self.robofixture.get_tool_pose("tcp")
        self.robofixture.couple(self.robot_C)
        
        """rob_posac, rob_orn = self.robofixture.get_tool_pose("tcp")
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=rob_posac)"""

        #next outcommented part is for real -> sim update
        """robot_C_tool_node_id = "ns=2;s=/Channel/Parameter/R[u1,1]"
        robot_D_tool_node_id = "ns=2;s=/Channel/Parameter/R[u2,1]"
        self.client.connect()
        node = self.client.get_node(robot_C_tool_node_id)
        tool_id_C = node.get_value()
        node = self.client.get_node(robot_D_tool_node_id)
        tool_id_D = node.get_value()

        print(f"Tool ID C: {tool_id_C}, Tool ID D: {tool_id_D}")
        self.client.disconnect()

        if tool_id_C == 6:
            self.BFG.couple(self.robot_C)
        if tool_id_D == 2:
            self.screwdriver.couple(self.robot_D)
        elif tool_id_D == 4:
            self.SRG.couple(self.robot_D)
        elif tool_id_D == 6:
            self.BFG.couple(self.robot_D)
        elif tool_id_D == 8:
            self.millingtool.couple(self.robot_D)
        elif tool_id_D == 10:
            self.DLG.couple(self.robot_D)


        self.update_real_robot_positions()"""
        for _ in range(100):
            self.stepSimulation()

        #gcode section
        # M-Commands have to be added in this convention
        m_commands = {
            "0":  [lambda: None],
            "10": [lambda: self.SRG.actuate(1)],
            "11": [lambda: self.SRG.actuate(0)],
            "12": [lambda: None],
            "20": [lambda: None],
            "201": [lambda: None],
            "21": [lambda: None],
            "22": [lambda: None],
            #"20": [lambda: self.screwdriver.activate()],
            #"21": [lambda: self.screwdriver.deactivate()],
            "30": [lambda: print("[WARNING] BFG not installed — M:30 no-op")],
            "31": [lambda: print("[WARNING] BFG not installed — M:31 no-op")],
            "32": [lambda: None],
            "33": [lambda: None],
            "40": [lambda: self.zivid.get_image()],
            "50": [lambda: None],
            "51": [lambda: None],
            "60": [lambda: self.DLG.actuate(1)],
            "61": [lambda: self.DLG.actuate(0)],
            "70": [lambda: None],
            "71": [lambda: None],
            "72": [lambda: None],
            "73": [lambda: None],
            "900": [lambda: None],
            "901": [lambda: None],
            "902": [lambda: None]
        }

        # T-Commands have to be added in this convention
        t_commands_C = {
            "6":  [lambda: print("[WARNING] BFG not installed — T:6 robot_C no-op")],
            "5":  [lambda: print("[WARNING] BFG not installed — T:5 robot_C no-op")],
            # Screwdriver_C tool-change commands (mirrors robot_D T:2/1/12/11)
            "2":  [lambda: self.screwdriver_C.couple(self.robot_C)],
            "1":  [lambda: self.screwdriver_C.decouple()],
            "12": [lambda: None],  # start screwdriver motor (no-op in sim)
            "11": [lambda: None],  # execute unscrew motion program (no-op in sim)
        }
        # T-Commands have to be added in this convention
        t_commands_D = {
            "18":[lambda: None],
            "13": [lambda: self.SRG.couple(self.robot_D)],
            "12": [lambda: None],
            "11": [lambda: None],
            "10": [lambda: self.DLG.couple(self.robot_D)],
            "9": [lambda: self.DLG.decouple()],
            "8": [lambda: self.millingtool.couple(self.robot_D)],
            "7": [lambda: self.millingtool.decouple()],
            "6": [lambda: print("[WARNING] BFG not installed — T:6 robot_D no-op")],
            "5": [lambda: print("[WARNING] BFG not installed — T:5 robot_D no-op")],
            "4": [lambda: self.SRG.couple(self.robot_D)],
            "3": [lambda: self.SRG.decouple()],
            "2": [lambda: self.screwdriver.couple(self.robot_D)],
            "1": [lambda: self.screwdriver.decouple()],
        }

        self.g_code_processor_C = GCodeProcessor(robot=self.robot_C, m_commands=m_commands,
                                                 t_commands=t_commands_C,
                                                 endeffector_list=[self.robofixture, self.screwdriver_C],
                                                 interpolation_precision=0.02)
        self.g_code_processor_D = GCodeProcessor(robot=self.robot_D, m_commands=m_commands, t_commands=t_commands_D,
                                                 endeffector_list=[self.SRG, self.screwdriver, self.millingtool, self.zivid], interpolation_precision=0.02) #zivid should be the last tool in the list
        self.g_code_processor_D.active_endeffector = -1
        
        self.g_code_C = []
        self.g_code_D = []
        self.safe_C()
        self.safe_D()
        self.g_code_C = []
        self.g_code_D = []

        # Motion planner selection. Change via set_planner().
        # Default: DIRECT — original single G1 move, unchanged behaviour.
        import sys as _sys
        _gnn_dir = os.path.dirname(os.path.abspath(__file__))
        if _gnn_dir not in _sys.path:
            _sys.path.insert(0, _gnn_dir)
        from modular_planner import PlannerType
        self._planner_type = PlannerType.DIRECT
        self._active_planner = None  # Holds the planner instance (e.g. GNNWaypointPlanner)

        # Mounting mode: 'floor' = fixed to world at FLOOR_SPAWN_POINT, robot_C free.
        #                'robofixture' = robot_C holds the assembly.
        self._mounting_mode = 'floor'
        # Neutral floor point midway between robot_C (0,0,0) and robot_D (-1.99,-3.515,0).
        self.FLOOR_SPAWN_POINT = np.array([-1.0, -1.76, 0.32])
        self.fixture_constraint = None
        self.robofixture_gripping_constraint = None

        if server_active:
            self.start_server()
    """def grip_SRG(self):
        self.SRG.actuate(1)
        for _ in range (50):
            p.stepSimulation(self.pysics_client)
        coupled_objects = self.SRG.activate()
        for c_o in coupled_objects:
            p.removeConstraint(c_o)"""
    def update_shadows(self):
        self.client.connect()
        nodesC = dict()
        nodesC['X'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,1]")
        nodesC['Y'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,2]")
        nodesC['Z'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,3]")
        nodesC['A'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,4]")
        nodesC['B'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,5]")
        nodesC['C'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,6]")
        nodesD = dict()
        nodesD['X'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,1]")
        nodesD['Y'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,2]")
        nodesD['Z'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,3]")
        nodesD['A'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,4]")
        nodesD['B'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,5]")
        nodesD['C'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,6]")
        valuesC = dict()
        valuesD = dict()
        for key in nodesC.keys():
            valuesC[key] = nodesC[key].get_value()
            valuesD[key] = nodesD[key].get_value()
        self.client.disconnect()
        print(valuesC)
        print(valuesD)
        euler_siemens_C = np.array([np.deg2rad(valuesC['A']), np.deg2rad(valuesC['B']), np.deg2rad(valuesC['C'])])
        rot = R.from_euler('XYZ', euler_siemens_C)
        euler_pb_C = rot.as_euler('xyz')
        euler_siemens_D = np.array([np.deg2rad(valuesD['A']), np.deg2rad(valuesD['B']), np.deg2rad(valuesD['C'])])
        rot = R.from_euler('XYZ', euler_siemens_D)
        euler_pb_D = rot.as_euler('xyz')
        self.robot_C_shadow.set_endeffector_pose(np.array([valuesC['X'], valuesC['Y'], valuesC['Z']]) / 1000,
                                                 p.getQuaternionFromEuler(euler_pb_C),
                                                 endeffector_name="link6")
        self.robot_D_shadow.set_endeffector_pose(np.array([valuesD['X'], valuesD['Y'], valuesD['Z']]) / 1000,
                                                 p.getQuaternionFromEuler(euler_pb_D),
                                                 endeffector_name="link6")
    def update_real_robot_positions(self):
        try:
            self.client.connect()
            nodesC = dict()
            nodesC['X'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,1]")
            nodesC['Y'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,2]")
            nodesC['Z'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,3]")
            nodesC['A'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,4]")
            nodesC['B'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,5]")
            nodesC['C'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u1,6]")
            nodesD = dict()
            nodesD['X'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,1]")
            nodesD['Y'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,2]")
            nodesD['Z'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,3]")
            nodesD['A'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,4]")
            nodesD['B'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,5]")
            nodesD['C'] = self.client.get_node("ns=2;s=/Channel/GeometricAxis/actToolBasePos[u2,6]")
            valuesC = dict()
            valuesD = dict()
            for key in nodesC.keys():
                valuesC[key] = nodesC[key].get_value()
                valuesD[key] = nodesD[key].get_value()
            self.client.disconnect()
            print(valuesC)
            print(valuesD)
            euler_siemens_C = np.array([np.deg2rad(valuesC['A']), np.deg2rad(valuesC['B']), np.deg2rad(valuesC['C'])])
            rot = R.from_euler('XYZ', euler_siemens_C)
            euler_pb_C = rot.as_euler('xyz')
            euler_siemens_D = np.array([np.deg2rad(valuesD['A']), np.deg2rad(valuesD['B']), np.deg2rad(valuesD['C'])])
            rot = R.from_euler('XYZ', euler_siemens_D)
            euler_pb_D = rot.as_euler('xyz')
            """self.robot_C.set_endeffector_pose(np.array([valuesC['X'], valuesC['Y'], valuesC['Z']]) / 1000,
                                                     p.getQuaternionFromEuler(euler_pb_C))"""
            self.zivid.set_tool_pose(np.array([valuesC['X'], valuesC['Y'], valuesC['Z']]) / 1000,
                                                     p.getQuaternionFromEuler(euler_pb_C))
            self.robot_D.set_endeffector_pose(np.array([valuesD['X'], valuesD['Y'], valuesD['Z']]) / 1000,
                                                     p.getQuaternionFromEuler(euler_pb_D))
        except:
            print("Error while updating real robot positions. Is the OPC UA Server running?")
        for _ in range(100):
            self.stepSimulation()
    def shadow_loop(self):
        while True:
            self.update_shadows()
            for _ in range(5):
                p.stepSimulation()
            time.sleep(0.1)
    def stick_robofixture(self, position_g, orientation_g,  part_name = "base"):
        #position_g, orientation_g = self.robofixture.get_tool_pose() # kann evtl raus.
        
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(inv_pos, inv_orn, position_o, orientation_o)
       
        self.robofixture_gripping_constraint= p.createConstraint(self.robofixture.urdf, self.robofixture._tcp_id,
                                                         self.assembly.part_ids[part_name], -1,
                                                         p.JOINT_FIXED, [0, 0, 0],
                                                         parentFramePosition=restraint_pos,
                                                         childFramePosition=[0, 0, 0],
                                                         parentFrameOrientation=restraint_orn,
                                                         childFrameOrientation=None)
        print("parent Frame Orientation")
        print(restraint_orn)
        print("child Frame Orientation")
        print(orientation_o)
        p.removeConstraint(self.assembly.constraint_ids[part_name])
    
    def stick_SRG(self, part_name):
        position_g, orientation_g = self.SRG.get_tool_pose()
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(
            inv_pos, inv_orn, position_o, orientation_o)
        self.SRG_gripping_constraint= p.createConstraint(self.SRG.urdf, self.SRG._tcp_id,
                                                         self.assembly.part_ids[part_name], -1,
                                                         p.JOINT_FIXED, [0, 0, 0],
                                                         parentFramePosition=restraint_pos,
                                                         childFramePosition=[0, 0, 0],
                                                         parentFrameOrientation=restraint_orn,
                                                         childFrameOrientation=None)
        p.removeConstraint(self.assembly.constraint_ids[part_name])
    def stick_DLG(self, part_name):
        position_g, orientation_g = self.DLG.get_tool_pose()
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(
            inv_pos, inv_orn, position_o, orientation_o)
        p.removeConstraint(self.assembly.constraint_ids[part_name])
        self.DLG_gripping_constraint= p.createConstraint(self.DLG.urdf, self.DLG._tcp_id,
                                                         self.assembly.part_ids[part_name], -1,
                                                         p.JOINT_FIXED, [0, 0, 0],
                                                         parentFramePosition=restraint_pos,
                                                         childFramePosition=[0, 0, 0],
                                                         parentFrameOrientation=restraint_orn,
                                                         childFrameOrientation=None)
        print(f"removed constraint {self.assembly.constraint_ids[part_name]} added constraint {self.DLG_gripping_constraint} for part {part_name} with id {self.assembly.part_ids[part_name]}")
        return self.DLG_gripping_constraint

    def stick_BFG(self, part_name):
        position_g, orientation_g = self.BFG.get_tool_pose()
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(
            inv_pos, inv_orn, position_o, orientation_o)
        self.BFG_gripping_constraint= p.createConstraint(self.BFG.urdf, self.BFG._tcp_id,
                                                         self.assembly.part_ids[part_name], -1,
                                                         p.JOINT_FIXED, [0, 0, 0],
                                                         parentFramePosition=restraint_pos,
                                                         childFramePosition=[0, 0, 0],
                                                         parentFrameOrientation=restraint_orn,
                                                         childFrameOrientation=None)
        p.removeConstraint(self.assembly.constraint_ids[part_name])
    def stick_sc(self, part_name):
        print(f"tcp_id: {self.screwdriver._tcp_id}")
        position_g, orientation_g = self.screwdriver.get_tool_pose()
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(
            inv_pos, inv_orn, position_o, orientation_o)
        self.sc_gripping_constraint= p.createConstraint(self.screwdriver.urdf, self.screwdriver._tcp_id,
                                                         self.assembly.part_ids[part_name], -1,
                                                         p.JOINT_FIXED, [0, 0, 0],
                                                         parentFramePosition=restraint_pos,
                                                         childFramePosition=[0, 0, 0],
                                                         parentFrameOrientation=restraint_orn,
                                                         childFrameOrientation=None)
        p.removeConstraint(self.assembly.constraint_ids[part_name])

    def stick_sc_C(self, part_name: str):
        """Attach screwdriver_C (robot_C's tool) to part_name via a fixed PyBullet constraint.

        Mirrors stick_sc() but for the second screwdriver on robot_C.  Called
        after the G-code engagement loop has positioned the screwdriver over the
        screw head and the assembly constraint on that screw must be released so
        the screw can move with the tool.
        """
        print(f"[stick_sc_C] tcp_id: {self.screwdriver_C._tcp_id}", flush=True)
        position_g, orientation_g = self.screwdriver_C.get_tool_pose()
        inv_pos, inv_orn = p.invertTransform(position_g, orientation_g)
        position_o, orientation_o = p.getBasePositionAndOrientation(
            self.assembly.part_ids[part_name])
        restraint_pos, restraint_orn = p.multiplyTransforms(
            inv_pos, inv_orn, position_o, orientation_o)
        self.sc_C_gripping_constraint = p.createConstraint(
            self.screwdriver_C.urdf, self.screwdriver_C._tcp_id,
            self.assembly.part_ids[part_name], -1,
            p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=restraint_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=restraint_orn,
            childFrameOrientation=None)
        p.removeConstraint(self.assembly.constraint_ids[part_name])

    def store_screwdriver_C(self):
        """Return screwdriver_C to its rest position at tool_storage1 and decouple it from robot_C."""
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_sc_C)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_sc_C)
        g_code_temp = [
            {'G': 1, 'X': self.pos_sc_C[0], 'Y': self.pos_sc_C[1],
             'Z': self.pos_sc_C[2] + 0.5,
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 10000.0},
            {'G': 1, 'X': self.pos_sc_C[0], 'Y': self.pos_sc_C[1], 'Z': self.pos_sc_C[2],
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 500.0},
            {'T': 1},  # decouple screwdriver_C from robot_C
        ]
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        self.g_code_processor_C.active_endeffector = -1
        # Re-couple the robofixture so robot_C is not left tool-less.
        self.robofixture.couple(self.robot_C)

    def unstore_screwdriver_C(self):
        """Pick up screwdriver_C from tool_storage1 and couple it to robot_C.

        Always decouples the robofixture first: it is directly coupled in __init__
        without going through a T-command, so active_endeffector may still read -1
        even when the robofixture is physically attached.  Decoupling a tool that is
        not coupled is a safe no-op in pybullet_industrial.
        """
        self.robofixture.decouple()
        self.g_code_processor_C.active_endeffector = -1
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_sc_C)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_sc_C)
        # Use the planner (GNN or DIRECT) for the long-travel approach to the storage
        # position.  This avoids the IK elbow-flip that occurs when commanding a raw
        # Cartesian G1 straight from the spawn configuration.
        _pos_D, _orn_D = self.robot_D.get_endeffector_pose()
        above_storage_C = np.array([self.start_pos_sc_C[0],
                                    self.start_pos_sc_C[1],
                                    self.start_pos_sc_C[2] + 0.5])
        self._execute_move_both(
            target_pos_C=above_storage_C,
            target_orn_q_C=np.array(self.start_orientation_sc_C),
            target_pos_D=np.array(_pos_D),
            target_orn_q_D=np.array(_orn_D),
            feed_rate_C=10000.0)
        # Fine descent and coupling handled by G-code (short straight-down move only).
        g_code_temp = [
            {'G': 1, 'X': self.start_pos_sc_C[0], 'Y': self.start_pos_sc_C[1],
             'Z': self.start_pos_sc_C[2],
             'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0},
            {'T': 2},  # couple screwdriver_C to robot_C
            {'G': 1, 'X': self.pos_sc_C[0], 'Y': self.pos_sc_C[1],
             'Z': self.pos_sc_C[2] + 0.4,
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0},
        ]
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(100):
            self.stepSimulation()

    def _interleave_gcode(self, g_code_C: list, g_code_D: list,
                          step_sims: int = 5,
                          check_robot_collision: bool = False) -> bool:
        """Execute two G-code sequences simultaneously by interleaving their steps.

        Each iteration of the main loop advances robot_C's processor by one step,
        then robot_D's processor by one step, then ticks PyBullet `step_sims` times.
        PyBullet's physics engine sees both robots moving at the same time.

        Both processors are exhausted before returning — the slower robot's remaining
        steps are drained after the faster one finishes.

        Parameters
        ----------
        g_code_C / g_code_D : list
            G-code waypoint dicts for robot_C and robot_D respectively.
        step_sims : int
            Physics steps per G-code step.
        check_robot_collision : bool
            When True, query PyBullet contact points between the two robot bodies
            after every physics tick.  If contact is detected the loop stops
            immediately and returns False.  Use this for the screwing phase where
            the pre-planned paths have no inter-robot awareness.

        Returns
        -------
        bool
            True  — completed without inter-robot contact.
            False — stopped early because a robot–robot collision was detected.
        """
        self.g_code_processor_C.g_code = g_code_C
        self.g_code_processor_D.g_code = g_code_D
        gen_C = iter(self.g_code_processor_C)
        gen_D = iter(self.g_code_processor_D)
        c_done = d_done = False
        while not (c_done and d_done):
            if not c_done:
                try:
                    next(gen_C)
                except StopIteration:
                    c_done = True
            if not d_done:
                try:
                    next(gen_D)
                except StopIteration:
                    d_done = True
            for _ in range(step_sims):
                self.stepSimulation()
            if check_robot_collision:
                contacts = p.getContactPoints(
                    bodyA=self.robot_C.urdf, bodyB=self.robot_D.urdf)
                if contacts:
                    print(
                        f'[_interleave_gcode] COLLISION between robot_C and robot_D '
                        f'detected after physics step.  Aborting interleaved execution.',
                        flush=True,
                    )
                    return False
        return True

    # ------------------------------------------------------------------
    # Dual-arm live arbitration
    # ------------------------------------------------------------------

    def _arm_bodies(self, side: str) -> list:
        """PyBullet body ids that make up one arm: the robot and its screwdriver."""
        if side == 'C':
            return [self.robot_C.urdf, self.screwdriver_C.urdf]
        return [self.robot_D.urdf, self.screwdriver.urdf]

    def _arm_link_aabbs(self, side: str) -> list:
        """World AABB of EVERY link of the arm and its tool.

        p.getAABB(body) alone is the base link only — the pedestal — which is
        why an arm fed to the planner that way was invisible to it.
        """
        out = []
        for bid in self._arm_bodies(side):
            for link in range(-1, p.getNumJoints(bid)):
                lo, hi = p.getAABB(bid, link)
                if all(h > l for l, h in zip(lo, hi)):
                    out.append((np.array(lo), np.array(hi)))
        return out

    def _arm_obstacle_aabbs(self, side: str) -> list:
        """The arm as a list of (aabb_min, aabb_max) boxes for the OTHER arm's planner.

        With cuMotion active: one small cube per collision sphere of the robot
        (cuMotion's own model, tight) plus the tool body's AABB.  Otherwise the
        per-link AABBs.  Link AABBs of oblique links are far too fat: they made
        cuMotion reject start states 20 cm from the other arm as "in collision".
        """
        from modular_planner import PlannerType
        robot, screwdriver, _, _, _ = self._arm(side)
        if self._planner_type != PlannerType.CUMOTION or self._active_planner is None \
                or not hasattr(self._active_planner, 'robot_spheres'):
            return self._arm_link_aabbs(side)
        try:
            rid = robot.urdf
            joints = [j for j in range(p.getNumJoints(rid)) if p.getJointInfo(rid, j)[2] != p.JOINT_FIXED]
            conf = np.array([p.getJointState(rid, j)[0] for j in joints], dtype=np.float64)
            base_pos, base_q = p.getBasePositionAndOrientation(rid)
            out = []
            for centre_local, r in self._active_planner.robot_spheres(conf, robot_name=f'robot_{side}'):
                c, _ = p.multiplyTransforms(base_pos, base_q, centre_local.tolist(), [0, 0, 0, 1])
                c = np.array(c)
                out.append((c - r, c + r))
            lo, hi = self._body_aabb(screwdriver.urdf)
            out.append((lo, hi))
            return out
        except Exception as e:
            print(f'[obstacles] sphere model for robot_{side} failed ({e}); link AABBs.', flush=True)
            return self._arm_link_aabbs(side)

    def _closest_link_pair(self, margin: float):
        """(distance, bodyA, linkA, bodyB, linkB) of the closest C/D approach within margin."""
        best = None
        for a in self._arm_bodies('C'):
            for b in self._arm_bodies('D'):
                for cp in p.getClosestPoints(bodyA=a, bodyB=b, distance=margin):
                    if best is None or cp[8] < best[0]:
                        best = (cp[8], a, cp[3], b, cp[4])
        return best

    def _describe_link(self, bid: int, link: int) -> str:
        base_name, body_name = (s.decode() for s in p.getBodyInfo(bid))
        link_name = base_name if link == -1 else p.getJointInfo(bid, link)[12].decode()
        return f'{body_name}/{link_name}'

    def _arms_within(self, margin: float) -> bool:
        """True if any body of arm C is within `margin` metres of any body of arm D."""
        return self._arms_closest(margin) is not None

    def _arms_closest(self, margin: float):
        """Closest approach between the two arms if it is within `margin`.

        Returns (distance, point_on_C, point_on_D) in world coordinates, or None.
        """
        best = None
        for a in self._arm_bodies('C'):
            for b in self._arm_bodies('D'):
                for cp in p.getClosestPoints(bodyA=a, bodyB=b, distance=margin):
                    d = cp[8]
                    if best is None or d < best[0]:
                        best = (d, np.array(cp[5]), np.array(cp[6]))
        return best

    def _tcp_position(self, side: str) -> np.ndarray:
        robot = self.robot_C if side == 'C' else self.robot_D
        return np.array(robot.get_endeffector_pose()[0], dtype=float)

    def _tcp_pose(self, side: str):
        """The pose every Cartesian G-code line commands: the TCP of the tool the
        arm holds (flange pose only when nothing is coupled).  Used for the
        WAITM split lines, escape lifts, return targets and the start-pose
        header, so those must be in the same frame as the planned lines.
        (Was the flange pose: every split line then hopped the tool 22 cm to
        the flange with its orientation -> axis-5 limit alarm at the first
        WAITM of Async Run 3.)"""
        robot, tool, _, _, _ = self._arm(side)
        if self.plan_tool_offset and tool.is_coupled():
            pos, q = tool.get_tool_pose()
        else:
            pos, q = robot.get_endeffector_pose()
        return np.array(pos, dtype=float), np.array(q, dtype=float)

    def _retrace_is_clear(self, side: str, points) -> tuple:
        """Would retracing along `points` run into the other arm as it is NOW?

        The arm is already close (or touching), so an absolute clearance test
        can never pass.  Instead: sampled every 2 cm, the path is rejected only
        if it brings the TCP CLOSER to the other arm than it is now AND inside
        the floor distance — getting somewhat nearer while still well clear is
        fine.  Returns (clear, min_distance_along_path).
        """
        other = 'D' if side == 'C' else 'C'
        boxes = self._arm_obstacle_aabbs(other)
        if not boxes or not points:
            return True, float('inf')

        def dist_to_boxes(pt):
            best = float('inf')
            for lo, hi in boxes:
                gap = np.maximum(np.maximum(lo - pt, 0.0), pt - hi)
                best = min(best, float(np.linalg.norm(gap)))
            return best

        here = self._tcp_position(side)
        d_now = dist_to_boxes(here)
        samples = []
        prev = here
        for pt in points:
            pt = np.asarray(pt, dtype=float)
            n = max(1, int(np.ceil(np.linalg.norm(pt - prev) / 0.02)))
            samples += [prev + (pt - prev) * k / n for k in range(1, n + 1)]
            prev = pt
        d_min = min(dist_to_boxes(s) for s in samples)
        closer = d_min < d_now - 0.01
        return not (closer and d_min < self.yield_floor_m), d_min

    def _escape_segments(self, side: str, current_seg: dict, trail=None) -> list:
        """Segments that move this arm out of the way and later bring it back.

        escape: first retrace the arm's own recent path (it came in along it,
                so it is known to be clear) for up to escape_retrace_m, then
                lift escape_dz_m.  Never held — it is leaving.
        return: a freshly planned transit back to where the interrupted segment
                starts, so the segment can simply run again from its beginning.
                A transit segment needs no return: re-running it re-plans.
        """
        gc = []
        travelled = 0.0
        # A LEAVING arm (pullout / drop / retreat) never retraces: its trail
        # leads back into the workspace it is trying to get out of.  Lift only.
        if trail and current_seg.get('yields'):
            prev = np.array(trail[-1][0])
            for pos_t, q_t in reversed(list(trail)[:-1]):
                step = float(np.linalg.norm(np.array(pos_t) - prev))
                if step < 0.02:
                    continue
                if travelled + step > self.escape_retrace_m:
                    break
                travelled += step
                e_t = p.getEulerFromQuaternion(q_t)
                gc.append({'G': 1, 'X': pos_t[0], 'Y': pos_t[1], 'Z': pos_t[2],
                           'A': e_t[0], 'B': e_t[1], 'C': e_t[2], 'F': 3000.0})
                prev = np.array(pos_t)
        # The trail was clear when the arm came in, but the other arm may have
        # moved into it since: if retracing would bring us closer to it than we
        # already are, do not retrace — lift directly.
        retrace_blocked = False
        if gc:
            clear, d_min = self._retrace_is_clear(side, [(g['X'], g['Y'], g['Z']) for g in gc])
            if not clear:
                print(f'[arbitration] robot_{side}: retrace path would run into the other arm '
                      f'(min {d_min*100:.1f} cm) — lifting directly', flush=True)
                gc, travelled, retrace_blocked = [], 0.0, True

        # Retrace and lift are separate segments: after the retrace the loop
        # checks whether the arms are clear and drops the lift if so.  With
        # nothing to retrace, the lift runs directly.
        segs = []
        if gc:
            segs.append({'name': f'escape-retrace ({current_seg["name"]})', 'gcode': gc,
                         'retraced_m': round(travelled, 2)})
            pos = np.array([gc[-1]['X'], gc[-1]['Y'], gc[-1]['Z']])
            e = (gc[-1]['A'], gc[-1]['B'], gc[-1]['C'])
        else:
            pos, q = self._tcp_pose(side)
            e = p.getEulerFromQuaternion(q)
        segs.append({'name': f'escape-lift ({current_seg["name"]})',
                     'gcode': [{'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2] + self.escape_dz_m,
                                'A': e[0], 'B': e[1], 'C': e[2], 'F': 3000.0}],
                     'retraced_m': round(travelled, 2)})
        if 'make_gcode' not in current_seg and 'start_pose' in current_seg:
            spos, sq = current_seg['start_pose']
            se = p.getEulerFromQuaternion(sq)

            def make():
                return self._plan_transit_gcode(side, spos, sq) + [
                    {'G': 1, 'X': spos[0], 'Y': spos[1], 'Z': spos[2],
                     'A': se[0], 'B': se[1], 'C': se[2], 'F': 1000.0}]
            segs.append({'name': f'return ({current_seg["name"]})', 'make_gcode': make, 'yields': True})
        return segs

    def _path_spheres(self, side: str, joint_path, wps):
        """Every collision sphere of the arm at every configuration of a planned
        path, in world coordinates: an (N, 4) array of x, y, z, r.

        Rows are cuMotion's own world-collision spheres (the model the planner
        checked) at each joint configuration, plus one sphere bounding the tool
        body at each waypoint's TCP (the same tool AABB the per-tick obstacle
        model uses).  None if the arm has no sphere model.
        """
        if not joint_path or self._active_planner is None \
                or not hasattr(self._active_planner, 'robot_spheres'):
            return None
        robot, screwdriver, _, _, _ = self._arm(side)
        base_pos, base_q = p.getBasePositionAndOrientation(robot.urdf)
        Rb = np.array(p.getMatrixFromQuaternion(base_q)).reshape(3, 3)
        bp = np.asarray(base_pos, dtype=float)
        lo, hi = p.getAABB(screwdriver.urdf)
        tool_r = 0.5 * float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
        rows = []
        for k, conf in enumerate(joint_path):
            for c, r in self._active_planner.robot_spheres(np.asarray(conf, dtype=np.float64),
                                                            robot_name=f'robot_{side}'):
                w = Rb @ np.asarray(c, dtype=float) + bp
                rows.append((w[0], w[1], w[2], r))
            if k < len(wps) and 'X' in wps[k]:
                rows.append((float(wps[k]['X']), float(wps[k]['Y']), float(wps[k]['Z']), tool_r))
        return np.asarray(rows, dtype=float)

    def _classify_pair_safety(self, seg_C: dict, seg_D: dict, margin: float = None):
        """Offline SAFE/UNSAFE classification of two planned segments.

        Timing-independent: the minimum sphere-to-sphere distance over the full
        Cartesian product of every configuration robot_C's path visits and every
        configuration robot_D's path visits.  If that minimum exceeds `margin`
        the two moves cannot interfere however the arms are paced against each
        other, and the reactive arbitration can be skipped for the pair.

        Returns (safe: bool, min_distance or None).  Segments without a joint
        path (fixed G-code, DIRECT fallback) are not classifiable -> UNSAFE.
        """
        if margin is None:
            margin = self.safe_pair_margin_m if self.safe_pair_margin_m is not None else self.yield_margin_m
        if not seg_C.get('joint_path') or not seg_D.get('joint_path'):
            return False, None
        sph_C = self._path_spheres('C', seg_C['joint_path'], seg_C.get('gcode', []))
        sph_D = self._path_spheres('D', seg_D['joint_path'], seg_D.get('gcode', []))
        if sph_C is None or sph_D is None or len(sph_C) == 0 or len(sph_D) == 0:
            return False, None
        d_min = float('inf')
        for i in range(0, len(sph_C), 512):        # blocked: paths can hold thousands of spheres
            blk = sph_C[i:i + 512]
            centres = np.linalg.norm(blk[:, None, :3] - sph_D[None, :, :3], axis=2)
            gap = centres - blk[:, None, 3] - sph_D[None, :, 3]
            d_min = min(d_min, float(gap.min()))
        return d_min > margin, d_min

    def _world_contacts(self, depth: float = -0.01) -> list:
        """Arm or tool bodies penetrating a static body (table, board, storage)
        deeper than `depth` (negative = penetration), as 'body/link' strings."""
        arms = set(self._arm_bodies('C')) | set(self._arm_bodies('D'))
        # Only what must never be touched: the clamping desk, the tool storages
        # and the assembly plate.  Screws (engaged on purpose), the screw boxes
        # (the drop goes inside them) and the floor are not faults.
        hard = set()
        for bid in range(p.getNumBodies()):
            name = p.getBodyInfo(bid)[1].decode().lower()
            if 'spanntisch' in name or ('toolstorage' in name and not getattr(self, 'ignore_storage_contacts', False)):
                hard.add(bid)
        if self.assembly is not None and 'base' in getattr(self.assembly, 'part_ids', {}):
            hard.add(self.assembly.part_ids['base'])
        # robot_C carries NO tool in the cell (ghost screwdriver = tool data only),
        # and its planning model has no tool spheres for the same reason; the
        # sim's ghost tool mesh must not raise a fault either (it did: the
        # Run-8 abort at "transit screw_13", ghost tool 13 mm into the table).
        arms.discard(self.screwdriver_C.urdf)
        out = []
        for a in arms:
            for c in p.getContactPoints(bodyA=a):
                if c[2] not in hard or c[8] > depth:
                    continue
                out.append(f'{self._describe_link(a, c[3])} -> {self._describe_link(c[2], c[4])} ({c[8] * 1000:.0f} mm)')
        return out

    def _arms_in_contact(self) -> bool:
        for a in self._arm_bodies('C'):
            for b in self._arm_bodies('D'):
                if p.getContactPoints(bodyA=a, bodyB=b):
                    return True
        return False

    @staticmethod
    def _priority_holder(out, done, default, cycles=None):
        """Which arm may keep moving when the two are too close.

        1. An arm that has finished everything never holds priority.
        2. Within a cycle, the arm that has NOT yet got its screw out has
           priority, so the one done unscrewing leaves while the other engages.
        3. Across cycles, the arm that has completed FEWER screws has priority —
           so the arms alternate and neither can starve the other.
        4. Only with everything equal does `default` (robot 1 = C) decide.
        """
        if done['C'] and not done['D']:
            return 'D'
        if done['D'] and not done['C']:
            return 'C'
        if out['C'] and not out['D']:
            return 'D'
        if out['D'] and not out['C']:
            return 'C'
        if cycles is not None:
            if cycles['C'] < cycles['D']:
                return 'C'
            if cycles['D'] < cycles['C']:
                return 'D'
        return default

    def _interleave_streams(self, stream_C: list, stream_D: list,
                            priority_default: str = 'C',
                            step_sims: int = 5):
        """Run two per-arm segment streams tick by tick with live yield arbitration.

        A stream is a list of segments ``{'name', 'gcode', 'on_done', 'screw_out'}``.
        ``on_done`` runs when that segment's G-code is exhausted; ``screw_out``
        marks the segment after which the arm has its screw out of the assembly,
        which hands priority to the other arm.

        Every tick the arm without priority decides whether to advance its next
        1 cm interpolation step (the processor's granularity):
          * inside ``yield_floor_m`` it always holds;
          * inside ``yield_margin_m`` it holds only while the priority arm's tool
            is moving towards it — if that arm is stationary (unscrewing) or
            receding (pulling out, going to drop) the other keeps creeping in;
          * otherwise it advances.
        So the deprioritised arm ends up parked as close as the floor allows and
        moves the very tick the priority arm starts to leave.

        Real contact is still detected: before any screw has been attached to a
        tool it aborts so the caller can fall back to serial execution;
        afterwards it warns and continues, since the state cannot be replayed.

        Returns (completed: bool, any_screw_attached: bool).
        """
        procs = {'C': self.g_code_processor_C, 'D': self.g_code_processor_D}
        streams = {'C': stream_C, 'D': stream_D}
        idx = {'C': 0, 'D': 0}
        gens = {}
        done = {'C': not stream_C, 'D': not stream_D}
        out = {'C': False, 'D': False}
        cycles = {'C': 0, 'D': 0}      # screws completed per arm, for fair alternation
        holding = {'C': False, 'D': False}
        any_attached = False

        def load(side):
            seg = streams[side][idx[side]]
            if seg.get('yields'):
                out[side] = False     # entering for the next screw: priority state restarts
            if 'make_gcode' in seg:
                # Planned only now, from the arm's real pose and the other arm's
                # current position (transit segments).
                seg['gcode'] = seg['make_gcode']()
            # The export is built AS EXECUTED: each line is appended when the
            # processor reads it (see the advance step), so a sync marker can be
            # spliced exactly where a hold happened.
            procs[side].g_code = list(seg['gcode'])
            gens[side] = iter(procs[side])

        def export_of(side):
            return self.g_code_C if side == 'C' else self.g_code_D

        sync_no = [0]

        def sync(waiter, mover, why):
            """A rendezvous: `waiter` may not continue past here before `mover`
            has reached its own current point.  Exported as WAITM in BOTH
            programs; the in-flight motion line of each arm is split at its
            current pose so the marker lands where the wait really was."""
            sync_no[0] += 1
            n = 1 + (sync_no[0] - 1) % self.sync_marker_count
            for s in (waiter, mover):
                gc = export_of(s)
                pos, q = self._tcp_pose(s)
                e = p.getEulerFromQuaternion(q)
                marker = {'WAITM': n, 'why': why, 'seq': sync_no[0]}
                j = len(gc) - 1
                if j >= 0 and 'X' in gc[j] and not gc[j].get('_done'):
                    line = gc[j]
                    here = dict(line)
                    here.update({'X': float(pos[0]), 'Y': float(pos[1]), 'Z': float(pos[2])})
                    if 'A' in line:
                        here.update({'A': e[0], 'B': e[1], 'C': e[2]})
                    gc[j] = here
                    gc.append(marker)
                    gc.append(line)
                else:
                    gc.append(marker)
            print(f'[sync] WAITM({n}) — robot_{waiter} resumes after robot_{mover}: {why}', flush=True)

        def truncate_export(side):
            """The arm abandons its in-flight line (its segment is re-planned):
            end that line where the arm actually is, and mark it finished."""
            gc = export_of(side)
            if gc and 'X' in gc[-1] and not gc[-1].get('_done'):
                pos, q = self._tcp_pose(side)
                e = p.getEulerFromQuaternion(q)
                here = dict(gc[-1])
                here.update({'X': float(pos[0]), 'Y': float(pos[1]), 'Z': float(pos[2]), '_done': True})
                if 'A' in here:
                    here.update({'A': e[0], 'B': e[1], 'C': e[2]})
                gc[-1] = here

        for side in ('C', 'D'):
            if not done[side]:
                load(side)

        contact_warned = False
        prev_dist = None                        # closest gap last tick (None outside the zone)
        advanced_last = {'C': False, 'D': False}  # which arms took a step last tick
        # Block resolution state for the current encounter (reset when the arms part).
        override = None          # forced priority holder while resolving a block
        enc = None               # {'prio': arm that had priority, 'held': arm that was holding}
        blocked_ticks = 0
        n_escapes = 0
        # Where each arm's TCP has actually been: an escape retraces this.
        from collections import deque
        trail = {'C': deque(maxlen=400), 'D': deque(maxlen=400)}
        tick_no = 0
        credit = {'C': 0.0, 'D': 0.0}        # pacing accumulator (1 cm steps)
        feed = {'C': self.feed_ref, 'D': self.feed_ref}   # F of the line each arm is on
        waited_for = {'C': None, 'D': None}   # arm this one is waiting on (escape) -> sync on release
        freeze_until = {'C': None, 'D': None}  # serialisation: wait until other idx >= this
        pair_contacts = {}                    # (seg_C, seg_D) -> contacts seen

        def in_escape(s):
            return not done[s] and streams[s][idx[s]]['name'].startswith('escape')

        # Offline pair safety.  Whenever the pair of current segments changes,
        # classify it: two PLANNED moves whose sphere models never come within
        # the margin — over every combination of configurations along the two
        # paths, i.e. whatever the timing — are SAFE and run at nominal pace
        # with the reactive logic (hold, closing-rate, deceleration, block /
        # escape, serialisation) switched off.  Only the hard-contact check
        # below stays on for them.  Anything else (an unplanned segment, a
        # DIRECT fallback, an escape) is UNSAFE = full arbitration as before.
        pair_safe = {'safe': False, 'key': None}

        def classify():
            key = tuple((idx[s] if not done[s] else None) for s in ('C', 'D'))
            if key == pair_safe['key']:
                return
            pair_safe['key'] = key
            if done['C'] or done['D']:
                pair_safe['safe'] = False
                return
            sC, sD = streams['C'][idx['C']], streams['D'][idx['D']]
            if not getattr(self, 'pair_safety_enabled', True):
                pair_safe['safe'] = False
                return
            safe, d = self._classify_pair_safety(sC, sD)
            pair_safe['safe'] = bool(safe)
            if d is not None:
                verdict = 'SAFE — nominal pace, no arbitration' if safe else 'UNSAFE — reactive arbitration'
                print(f'[pair-safety] "{sC["name"]}" x "{sD["name"]}": closest approach '
                      f'{d*100:.1f} cm over all configurations -> {verdict}', flush=True)
            elif 'joint_path' in sC or 'joint_path' in sD:
                print(f'[pair-safety] "{sC["name"]}" x "{sD["name"]}": not classifiable '
                      f'(a side has no planned joint path) -> UNSAFE', flush=True)

        while not (done['C'] and done['D']):
            tick_no += 1
            classify()
            holder = override or self._priority_holder(out, done, priority_default, cycles)
            other = 'D' if holder == 'C' else 'C'

            # Does the non-priority arm have to hold this tick?  And is the
            # priority arm itself being blocked by an arm that is holding?
            must_hold = False
            approaching = False
            why = ''
            info = None if pair_safe['safe'] else self._arms_closest(self.yield_margin_m)
            if info is not None:
                dist, pt_C, pt_D = info
                # Closing-rate: the closest gap (any link, any tool) shrank by
                # more than the yielding arm's own last step could explain, so
                # the priority arm — some part of it — is coming in.
                if prev_dist is not None:
                    own = self.own_step_m if advanced_last[other] else 0.0
                    closing = prev_dist - dist
                    approaching = closing > own + self.closing_eps_m
                if dist <= self.yield_floor_m:
                    must_hold, why = True, f'{dist*100:.1f} cm <= floor'
                elif approaching:
                    must_hold, why = True, (f'{dist*100:.1f} cm, closing '
                                            f'{closing*1000:.0f} mm/tick by robot_{holder}')
                prev_dist = dist
            else:
                override, enc, blocked_ticks, n_escapes = None, None, 0, 0   # encounter over
                prev_dist = None
            advanced_last = {'C': False, 'D': False}

            # Priority arm inside the floor of a HOLDING arm and still closing:
            # it cannot get where it needs to go.  1) swap: let the holding arm
            # finish instead.  2) still blocked: the original holder escapes
            # upward and the original priority arm proceeds.
            blocked = (info is not None and dist <= self.yield_floor_m
                       and approaching and holding[other])
            blocked_ticks = blocked_ticks + 1 if blocked else 0
            if blocked_ticks >= self.block_ticks:
                blocked_ticks = 0
                if enc is None:
                    enc = {'prio': holder, 'held': other}
                    override = other
                    print(f'[arbitration] robot_{holder} is BLOCKED by holding robot_{other} '
                          f'-> swapping priority: robot_{other} proceeds, robot_{holder} holds',
                          flush=True)
                elif n_escapes < self.max_escapes:
                    n_escapes += 1
                    esc_side, go_side = enc['held'], enc['prio']
                    override = go_side
                    cur = streams[esc_side][idx[esc_side]]
                    new = self._escape_segments(esc_side, cur, trail[esc_side])
                    streams[esc_side][idx[esc_side]:idx[esc_side]] = new
                    holding[esc_side] = False
                    load(esc_side)
                    print(f'[arbitration] mutual block -> robot_{esc_side} ESCAPES '
                          f'(retrace {new[0]["retraced_m"]} m, then lift +{self.escape_dz_m:.2f} m only if still close, '
                          f'then returns to "{cur["name"]}"); robot_{go_side} waits '
                          f'(escape {n_escapes}/{self.max_escapes})', flush=True)
                    holder, other = go_side, esc_side
                    must_hold = False
                else:
                    print('[arbitration] block could not be resolved after '
                          f'{self.max_escapes} escapes — aborting.', flush=True)
                    return False, any_attached

            for side in ('C', 'D'):
                if done[side]:
                    continue
                other_side = 'D' if side == 'C' else 'C'
                # While the other arm is escaping, wait for it: its old plan is
                # what got us here, and this arm re-plans once the escape is done.
                if in_escape(other_side):
                    waited_for[side] = other_side
                    continue
                # Serialised after repeated contacts: wait until the other arm
                # has moved on before doing anything at all.
                if freeze_until[side] is not None and not in_escape(side):
                    if done[other_side] or idx[other_side] >= freeze_until[side]:
                        print(f'[arbitration] robot_{side} released from serialisation', flush=True)
                        freeze_until[side] = None
                        sync(side, other_side, 'serialisation released')
                    else:
                        continue
                # Only an arm ENTERING the shared space (transit/engage) can be
                # held.  An arm pulling out or leaving for the box always proceeds:
                # holding it would keep it in the way of the arm it yields to.
                entering = streams[side][idx[side]].get('yields', False)

                # One pacing number per arm per tick, in 1 cm steps:
                #   feed:   this line's F relative to feed_ref (the cell's real ratio)
                #   global: /pace_divider (everything slower)
                #   zone:   priority arm at 1/slow_factor inside the awareness zone
                #   yield:  yielding entering arm decelerates to 0 at the floor
                allowance = (feed[side] / self.feed_ref) / self.pace_divider
                if in_escape(side):
                    allowance = 1.0 / self.pace_divider
                elif info is not None:
                    if side == holder:
                        allowance /= self.slow_factor
                    elif entering and not must_hold:
                        frac = (dist - self.yield_floor_m) / max(self.yield_margin_m - self.yield_floor_m, 1e-6)
                        allowance *= min(max(frac, 0.0), 1.0)
                credit[side] += allowance
                if credit[side] < 1.0:
                    continue
                credit[side] -= 1.0
                if must_hold and side == other and entering:
                    if not holding[side]:
                        seg = streams[side][idx[side]]['name']
                        pair = self._closest_link_pair(self.yield_margin_m)
                        where = (f' [C:{self._describe_link(pair[1], pair[2])} <-> '
                                 f'D:{self._describe_link(pair[3], pair[4])}]') if pair else ''
                        print(f'[arbitration] robot_{side} holds at "{seg}" '
                              f'({why}; robot_{holder} has priority){where}', flush=True)
                        holding[side] = True
                    continue
                if holding[side]:
                    print(f'[arbitration] robot_{side} resumes', flush=True)
                    holding[side] = False
                    sync(side, other_side, 'hold released')
                if waited_for[side] is not None:
                    # The escape we waited for is over: rendezvous before moving on.
                    sync(side, waited_for[side], 'escape finished')
                    waited_for[side] = None
                try:
                    line = next(gens[side])
                    advanced_last[side] = True
                    if isinstance(line, dict):
                        export_of(side).append(line)          # export as executed
                        if line.get('F') is not None:
                            feed[side] = float(line['F'])
                    trail[side].append(self._tcp_pose(side))
                except StopIteration:
                    seg = streams[side][idx[side]]
                    if seg.get('replan'):
                        # Chunked transit: only the first waypoints were executed;
                        # plan the rest from where the arm is now.
                        load(side)
                        continue
                    if seg['name'].startswith('escape-retrace'):
                        # Was retracing enough?  If the arms are clear of the
                        # floor, the lift that follows is not needed.
                        nxt = streams[side][idx[side] + 1] if idx[side] + 1 < len(streams[side]) else None
                        clear = not self._arms_within(self.yield_floor_m + 0.05)
                        if clear and nxt is not None and nxt['name'].startswith('escape-lift'):
                            streams[side].pop(idx[side] + 1)
                            print(f'[arbitration] robot_{side} retrace ({seg["retraced_m"]} m) was '
                                  f'enough — no lift', flush=True)
                        elif nxt is not None:
                            print(f'[arbitration] robot_{side} retraced {seg["retraced_m"]} m, '
                                  f'still close — lifting', flush=True)
                    nxt = streams[side][idx[side] + 1] if idx[side] + 1 < len(streams[side]) else None
                    if seg['name'].startswith('escape') and not (nxt and nxt['name'].startswith('escape')):
                        # Escape complete.  The other arm waited; whatever transit
                        # it was on was planned against the pre-escape world.
                        o = other_side
                        if not done[o] and 'make_gcode' in streams[o][idx[o]]:
                            print(f'[arbitration] robot_{side} escape done -> robot_{o} '
                                  f're-plans "{streams[o][idx[o]]["name"]}"', flush=True)
                            truncate_export(o)
                            load(o)
                    if seg.get('on_done'):
                        seg['on_done']()
                        if seg.get('attaches_screw'):
                            any_attached = True
                    if seg.get('screw_out'):
                        out[side] = True
                        cycles[side] += 1
                        if override == side:
                            override, enc = None, None    # the favoured arm is done: normal rules again
                        print(f'[arbitration] robot_{side} has its screw out '
                              f'(C:{cycles["C"]} D:{cycles["D"]} done) -> '
                              f'priority to robot_{"D" if side == "C" else "C"}', flush=True)
                    idx[side] += 1
                    if idx[side] >= len(streams[side]):
                        done[side] = True
                    else:
                        load(side)

            for _ in range(step_sims):
                self.stepSimulation()

            if self._arms_in_contact():
                # An escape in progress is still touching while it lifts — give it time.
                if any(not done[s] and streams[s][idx[s]]['name'].startswith('escape')
                       for s in ('C', 'D')):
                    continue
                if pair_safe['safe']:
                    # The fail-safe: the offline classification said these two
                    # moves could not meet, yet they touch (model / margin
                    # mismatch).  Drop the pair to UNSAFE and hand it to the
                    # existing contact handling — escape, then abort to the
                    # serial (Option A) fallback if that is exhausted.
                    cur = {s: (streams[s][idx[s]]['name'] if not done[s] else 'done') for s in ('C', 'D')}
                    print(f'[pair-safety] CONTACT on a pair classified SAFE ("{cur["C"]}" x '
                          f'"{cur["D"]}") — classification overridden, reactive arbitration on.',
                          flush=True)
                    pair_safe['safe'] = False
                pair = self._closest_link_pair(0.0)
                where = (f' [C:{self._describe_link(pair[1], pair[2])} <-> '
                         f'D:{self._describe_link(pair[3], pair[4])}]') if pair else ''
                cur = {s: (streams[s][idx[s]]['name'] if not done[s] else 'done') for s in ('C', 'D')}
                key = (cur['C'], cur['D'])
                pair_contacts[key] = pair_contacts.get(key, 0) + 1
                if pair_contacts[key] >= self.max_pair_contacts and not done[other] and not done[holder]:
                    # The same two moves keep colliding no matter how often the
                    # yielding arm escapes and both re-plan: the conflict is
                    # structural.  Serialise — freeze the yielding arm until the
                    # priority arm has moved on two segments.
                    pair_contacts[key] = 0
                    freeze_until[other] = idx[holder] + 2
                    new = self._escape_segments(other, streams[other][idx[other]], None)
                    streams[other][idx[other]:idx[other]] = [s for s in new if s['name'].startswith('escape')]
                    holding[other] = False
                    load(other)
                    print(f'[arbitration] {self.max_pair_contacts} contacts between "{cur["C"]}" and '
                          f'"{cur["D"]}" — SERIALISING: robot_{other} lifts and waits until '
                          f'robot_{holder} has moved on.', flush=True)
                    continue
                if n_escapes < self.max_escapes and not done[other]:
                    # Already touching: no point swapping — the arm without
                    # priority gets out of the way (carrying its screw if it has
                    # one), then returns and retries.
                    n_escapes += 1
                    esc = other
                    new = self._escape_segments(esc, streams[esc][idx[esc]], trail[esc])
                    streams[esc][idx[esc]:idx[esc]] = new
                    holding[esc] = False
                    load(esc)
                    override, enc, blocked_ticks = holder, {'prio': holder, 'held': esc}, 0
                    print(f'[arbitration] CONTACT{where} — C at "{cur["C"]}", D at "{cur["D"]}". '
                          f'robot_{esc} ESCAPES (retrace {new[0]["retraced_m"]} m, then lift only if still close '
                          f'+{self.escape_dz_m:.2f} m), robot_{holder} waits '
                          f'(escape {n_escapes}/{self.max_escapes}).', flush=True)
                    continue
                if not any_attached:
                    print(f'[arbitration] CONTACT between arms before any screw was attached'
                          f'{where} — C at "{cur["C"]}", D at "{cur["D"]}". '
                          f'Escapes exhausted; aborting for serial fallback.', flush=True)
                    return False, False
                if not contact_warned:
                    print(f'[arbitration] WARNING: contact{where} with escapes exhausted after '
                          'a screw was attached; continuing (state cannot be replayed).', flush=True)
                    contact_warned = True
        return True, any_attached

    def unscrew_C(self, part_name: str, path: list, direction: np.array):
        """Unscrew one part using robot_C with screwdriver_C.

        Mirrors unscrew() but targets robot_C / screwdriver_C / g_code_processor_C
        and uses storage_spaces[0] as the drop location so screws from both arms
        land in the same bin.  Robot_C uses the DIRECT (plain G-code) planner;
        GNN/cuMotion planning for robot_C can be wired in later.
        """
        if part_name not in self.assembly.part_ids:
            print(f'[unscrew_C] ERROR: part "{part_name}" not in assembly.part_ids.', flush=True)
            return
        tool_path = pi.ToolPath(
            np.transpose([pnt * self.assembly.scaleFactor for pnt in path]))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        apply_tool_path_rotation(tool_path, orn)
        tool_path.translate(pos)
        direction_r, _ = p.multiplyTransforms([0, 0, 0], orn, direction, [0, 0, 0, 1])
        _, current_dir_sc_C = self.screwdriver_C.get_tool_pose()
        _, grip_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction_r, current_dir_sc_C),
                                              [0, 0, 0],
                                              current_dir_sc_C)
        tool_path.orientations = np.array([grip_orient for _ in path]).transpose()
        tool_path.draw(False, [0, 0, 1])  # draw in blue to distinguish from robot_D (red)

        # ── Phase 1: approach to start of screwing path via modular planner ──
        pos0 = tool_path.positions[:, 0]
        self._execute_move(pos0, tool_path.orientations[:, 0],
                           robot=self.robot_C, g_code_processor=self.g_code_processor_C,
                           feed_rate=10000.0, robot_name='robot_C', g_code_log=self.g_code_C)

        # ── Phase 2: engage and unscrew (mirrors robot_D's G-code sequence) ─────
        g_code_temp = []
        pos1 = tool_path.positions[:, 1]
        orn1 = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos1[0], 'Y': pos1[1], 'Z': pos1[2],
             'A': orn1[0], 'B': orn1[1], 'C': orn1[2], 'F': 2000.0})
        g_code_temp.append({'T': 12})  # start screwdriver motor slowly
        for i in range(2, len(path) - 1):
            pos_i = tool_path.positions[:, i]
            orn_i = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append(
                {'G': 1, 'X': pos_i[0], 'Y': pos_i[1], 'Z': pos_i[2],
                 'A': orn_i[0], 'B': orn_i[1], 'C': orn_i[2], 'F': 1000.0})
        g_code_temp.append({'M': 0})   # pause (matches robot_D sequence)
        g_code_temp.append({'T': 11})  # execute unscrew program
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()

        # ── Phase 3: attach screw to tool (stick_sc_C releases assembly constraint internally) ──
        self.stick_sc_C(part_name)

        # ── Phase 4: pull screw out and drop into storage bin ────────────────────
        g_code_temp = []
        pos_last = tool_path.positions[:, -1]
        orn_last = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos_last[0], 'Y': pos_last[1], 'Z': pos_last[2],
             'A': orn_last[0], 'B': orn_last[1], 'C': orn_last[2], 'F': 1500.0})
        drop_delta = tool_path.positions[:, -1] - tool_path.positions[:, -2]
        approach = self.storage_spaces[0]["approachpoint"] + drop_delta
        drop     = self.storage_spaces[0]["droppoint"]    + drop_delta * 0.25
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2],
                             'A': 0.0, 'B': 0.0, 'C': DROP_YAW_RAD, 'F': 10000.0})
        g_code_temp.append({'G': 1, 'X': drop[0], 'Y': drop[1], 'Z': drop[2]})
        g_code_temp.append({'M': 21})  # decouple / release screw
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        p.removeConstraint(self.sc_C_gripping_constraint)
        self.sc_C_gripping_constraint = None

        g_code_temp = [{'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2],
                         'F': 5000.0}]
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()

    # ------------------------------------------------------------------
    # Screw-cycle building blocks shared by unscrew_both and unscrew_pipeline
    # ------------------------------------------------------------------

    def _arm(self, side: str):
        """(robot, screwdriver, gripping-constraint attribute, stick fn, storage idx) for one arm."""
        if side == 'C':
            return self.robot_C, self.screwdriver_C, 'sc_C_gripping_constraint', self.stick_sc_C, 0
        return self.robot_D, self.screwdriver, 'sc_gripping_constraint', self.stick_sc, 1

    def _cell_tools(self):
        """Every tool body of the scene, the camera included (created after the planners are registered, so
        looked up late)."""
        names = ('screwdriver', 'screwdriver_C', 'SRG', 'DLG', 'BFG', 'millingtool', 'robofixture', 'zivid')
        return [getattr(self, n) for n in names if hasattr(self, n)]

    @staticmethod
    def _in_connector_frame(tool, pose):
        """`pose` (world) in the frame of the tool's connector link, which `couple` puts on the arm's end-effector
        link. Taken from the tool's own kinematics, so it holds before physics has moved the tool there."""
        if tool._connector_id == -1:
            connector = p.getBasePositionAndOrientation(tool.urdf)
        else:
            connector = p.getLinkState(tool.urdf, tool._connector_id, computeForwardKinematics=True)[4:6]
        return p.multiplyTransforms(*p.invertTransform(*connector), *pose)

    def _straight_line_arm(self, side: str):
        """The arm of `side` for straight_line_checked: IK on its end-effector link; it carries every body
        coupled to it (tools, the camera), and the TCP is that of its coupled tool (as `_tcp_pose`). robot_C's
        screwdriver is a ghost (tool data only, no tool on the cell): it gives the TCP but moves no body and is
        never checked, as in `_world_contacts`."""
        robot = self.robot_C if side == 'C' else self.robot_D

        def coupled():
            return [t for t in self._cell_tools() if getattr(t, '_coupled_robot', None) is robot]

        def tools():
            return [(t.urdf, self._in_connector_frame(t, p.getBasePositionAndOrientation(t.urdf)))
                    for t in coupled() if t is not getattr(self, 'screwdriver_C', None)]

        def tcp():
            tool = next((t for t in coupled() if t is not getattr(self, 'zivid', None)), None)
            if tool is None or not self.plan_tool_offset:
                return None
            return self._in_connector_frame(tool, tool.get_tool_pose())

        return straight_line_planner.Arm(robot.urdf, robot._default_endeffector_id, tools, tcp)

    def _straight_line_ignored(self):
        """Bodies that are never obstacles for straight_line_checked: the product (an operation touches it on
        purpose), robot_C's ghost screwdriver and the shadow robots."""
        ignored = {self.screwdriver_C.urdf} if hasattr(self, 'screwdriver_C') else set()
        if self.assembly is not None:
            ignored |= set(getattr(self.assembly, 'part_ids', {}).values())
        for name in ('robot_C_shadow', 'robot_D_shadow'):
            if hasattr(self, name):
                ignored.add(getattr(self, name).urdf)
        return ignored

    def _screw_tool_path(self, part_name, path, direction, screwdriver):
        tool_path = pi.ToolPath(
            np.transpose([pnt * self.assembly.scaleFactor for pnt in path]))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        apply_tool_path_rotation(tool_path, orn)
        tool_path.translate(pos)
        direction_r, _ = p.multiplyTransforms([0, 0, 0], orn, direction, [0, 0, 0, 1])
        _, current_dir = screwdriver.get_tool_pose()
        _, grip_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction_r, current_dir),
                                              [0, 0, 0],
                                              current_dir)
        # The bit spins: the roll about the screw axis is free.  Inheriting the
        # current roll fixes the wrist arbitrarily (some rolls have no cuMotion
        # IK at all); pick one the planner can actually reach at the standoff.
        side = 'C' if screwdriver is self.screwdriver_C else 'D'
        grip_orient = self._pick_screw_roll(side, tool_path.positions[:, 0], grip_orient, part_name)
        tool_path.orientations = np.array([grip_orient for _ in path]).transpose()
        return tool_path

    def _pick_screw_roll(self, side: str, tip_pos, tool_q, part_name: str = '',
                         steps_deg: int = 30, seed=None, avoid_storage: bool = False):
        """Roll `tool_q` about its own z (the screw axis) until cuMotion's IK
        finds a valid, limit-clear, self-collision-free configuration for the
        FLANGE at the standoff.  Candidates are tried nearest-to-current first
        (0, +-30, +-60, ...) so the wrist turns as little as possible; the
        current roll is kept if the planner is not cuMotion or nothing works.
        """
        planner = self._active_planner
        if planner is None or not hasattr(planner, 'ik') or not self.plan_tool_offset:
            return tool_q
        robot, _, _, _, _ = self._arm(side)
        try:
            T_off = self._tool_offset(side)
            bp, bq = p.getBasePositionAndOrientation(robot.urdf)
            T_base = self._pose_to_4x4(bp, bq)
            if seed is None:
                seed = self._planner_start_conf(side)
            seed = np.asarray(seed, dtype=float)
            static = self._static_obstacles()      # a roll that parks the wrist in the board is no good
            if avoid_storage:
                from modular_planner import ObstacleBox
                for sid in self._tool_storage_ids:
                    for lk in range(-1, p.getNumJoints(sid)):
                        lo, hi = p.getAABB(sid, lk)
                        static.append(ObstacleBox.from_pybullet_aabb(np.array(lo), np.array(hi)))
            rolls = [0] + [r for k in range(1, 180 // steps_deg + 1) for r in (k * steps_deg, -k * steps_deg)]
            rolls = [r for r in rolls if -180 < r <= 180]
            # Score every reachable roll by how far the arm has to move from
            # where it is.  q5 (the wrist pitch) decides whether the angle head
            # hangs below the flange or is flipped over it: a sign change there
            # is a whole wrist flip, so it costs extra even when the joint
            # distance looks small.  Multi-turn q4/q6 are already wrapped to
            # the nearest turn by planner.ik().
            w = np.array([1.0, 1.0, 1.0, 0.5, 2.0, 0.3])
            best = None
            for roll in rolls:
                _, q_roll = p.multiplyTransforms([0, 0, 0], tool_q, [0, 0, 0],
                                                 p.getQuaternionFromEuler([0, 0, math.radians(roll)]))
                T_goal = self._pose_to_4x4(tip_pos, q_roll) @ np.linalg.inv(T_off)
                q_ik = planner.ik(T_goal, robot_name=f'robot_{side}', seed=seed, robot_base_pose=T_base,
                                  obstacles=static)
                if q_ik is None:
                    continue
                cost = float(np.linalg.norm(w * (q_ik - seed)))
                if abs(seed[4]) > 0.3 and np.sign(q_ik[4]) != np.sign(seed[4]):
                    cost += 3.0                      # wrist flip
                # dress-pack twist: prefer rolls that keep |q4 + q6| small (the
                # 12 candidate rolls differ mostly in q6, so this is the lever)
                cost += 0.4 * abs(float(q_ik[3] + q_ik[5]))
                if best is None or cost < best[0]:
                    best = (cost, roll, np.asarray(q_roll), q_ik)
            if best is not None:
                cost, roll, q_roll, q_ik = best
                if roll != 0:
                    print(f'[roll robot_{side}] {part_name}: roll {roll:+d} deg '
                          f'(joint cost {cost:.2f}, q5 {q_ik[4]:+.2f})', flush=True)
                return q_roll
            print(f'[roll robot_{side}] {part_name}: no roll with a valid IK — keeping current.', flush=True)
        except Exception as e:
            print(f'[roll robot_{side}] {part_name}: roll selection failed ({e}); keeping current.', flush=True)
        return tool_q

    @staticmethod
    def _screw_engage_gcode(tp, path):
        gc = []
        pos1 = tp.positions[:, 1]
        orn1 = p.getEulerFromQuaternion(tp.orientations[:, 1])
        gc.append({'G': 1, 'X': pos1[0], 'Y': pos1[1], 'Z': pos1[2],
                   'A': orn1[0], 'B': orn1[1], 'C': orn1[2], 'F': 2000.0})
        gc.append({'T': 12})
        for i in range(2, len(path) - 1):
            pos_i = tp.positions[:, i]
            orn_i = p.getEulerFromQuaternion(tp.orientations[:, i])
            gc.append({'G': 1, 'X': pos_i[0], 'Y': pos_i[1], 'Z': pos_i[2],
                       'A': orn_i[0], 'B': orn_i[1], 'C': orn_i[2], 'F': 1000.0})
        gc.append({'M': 0})
        gc.append({'T': 11})
        return gc

    def _reachable_drop(self, side, approach, approach_q, drop, step: float = 0.025):
        """The drop descent is a Cartesian line the controller solves itself: make
        sure a joint solution within ALL limits exists at the drop point, else
        raise it towards the approach point until one does.  (Run 8: robot_D's
        0.425 m drop at the SE box needed Ax.2 > 75 deg -> software limit alarm.)"""
        planner = self._active_planner
        if side is None or planner is None or not hasattr(planner, 'ik'):
            return drop
        robot, _, _, _, _ = self._arm(side)
        T_off = self._tool_offset(side)
        bp, bq = p.getBasePositionAndOrientation(robot.urdf)
        T_base = self._pose_to_4x4(bp, bq)
        seed = planner.ik(self._pose_to_4x4(np.asarray(approach, float), np.asarray(approach_q, float)) @ np.linalg.inv(T_off),
                          robot_name=f'robot_{side}', seed=self._planner_start_conf(side), robot_base_pose=T_base)
        z0 = float(drop[2])
        d = np.array(drop, dtype=float)
        while d[2] < approach[2] - 1e-6:
            sol = planner.ik(self._pose_to_4x4(d, np.asarray(approach_q, float)) @ np.linalg.inv(T_off),
                             robot_name=f'robot_{side}', seed=seed if seed is not None else self._planner_start_conf(side),
                             robot_base_pose=T_base)
            if sol is not None:
                break
            d[2] += step
        if d[2] != z0:
            print(f'[drop@box_{side}] drop point RAISED {z0:.3f} -> {d[2]:.3f} m: lower is not reachable '
                  f'within the joint limits at this box', flush=True)
        return d

    def _screw_pullout_gcode(self, tp, storage_idx, side: str = None):
        gc = []
        pos_last = tp.positions[:, -1]
        orn_last = p.getEulerFromQuaternion(tp.orientations[:, 2])
        gc.append({'G': 1, 'X': pos_last[0], 'Y': pos_last[1], 'Z': pos_last[2],
                   'A': orn_last[0], 'B': orn_last[1], 'C': orn_last[2], 'F': 1500.0})
        drop_delta = tp.positions[:, -1] - tp.positions[:, -2]
        approach = self.storage_spaces[storage_idx]["approachpoint"] + drop_delta
        drop     = self.storage_spaces[storage_idx]["droppoint"]    + drop_delta * 0.25
        # The drop yaw is as free as the screw roll: keep DROP_YAW_RAD when the
        # planner can reach it (the exported C-90.000), otherwise the nearest
        # yaw it can, so the flight to the box has a reachable goal.
        approach_q = p.getQuaternionFromEuler([0.0, 0.0, DROP_YAW_RAD])
        if side is not None:
            approach_q = self._pick_screw_roll(side, approach, approach_q, f'drop@box_{side}')
        e = p.getEulerFromQuaternion(approach_q)
        gc.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2],
                   'A': e[0], 'B': e[1], 'C': e[2], 'F': 10000.0})
        drop = self._reachable_drop(side, approach, approach_q, drop)
        gc.append({'G': 1, 'X': drop[0], 'Y': drop[1], 'Z': drop[2]})
        gc.append({'M': 21})
        return gc, approach, approach_q

    def _screw_segments(self, side: str, part_name: str, tp, path):
        """engage -> attach -> pull out -> fly to box -> drop -> release -> retreat.

        Only 'engage' carries ``yields``: an arm entering the shared space may be
        held, an arm leaving it never is.  The flight to the box is the longest
        move of the cycle and crosses the shared zone, so it is PLANNED (lazily,
        like a transit) rather than a straight line; only the short descent
        into the box and the retreat back up stay as fixed G-code.
        """
        _, _, constraint_attr, stick_fn, storage_idx = self._arm(side)
        gc_engage = self._screw_engage_gcode(tp, path)
        gc_pull, approach, approach_q = self._screw_pullout_gcode(tp, storage_idx, side)
        gc_retreat = [{'G': 1, 'X': approach[0], 'Y': approach[1],
                       'Z': approach[2], 'F': 5000.0}]
        to_box = self._planned_segment(f'to-box {part_name}', side, approach, approach_q,
                                       tail=[gc_pull[1]], yields=False)

        def stick_then_settle():
            # The real unscrewing happens in the controller's SCREW_PROGRAM
            # (exported as T11); the sim only attaches the screw.  Dwell for the
            # time the program takes so timing/throughput in the sim is realistic.
            for _ in range(10 + int(self.unscrew_dwell_s * 240)):
                self.stepSimulation()
            stick_fn(part_name)

        def release():
            cid = getattr(self, constraint_attr)
            if cid is not None:
                p.removeConstraint(cid)
                setattr(self, constraint_attr, None)

        return [
            {'name': f'engage {part_name}',  'gcode': gc_engage, 'yields': True,
             'on_done': stick_then_settle, 'attaches_screw': True,
             'start_pose': (np.array(tp.positions[:, 0]), np.array(tp.orientations[:, 0]))},
            {'name': f'pullout {part_name}', 'gcode': gc_pull[:1], 'screw_out': True},
            to_box,
            {'name': f'drop {part_name}',    'gcode': gc_pull[2:], 'on_done': release},
            {'name': f'retreat {part_name}', 'gcode': gc_retreat},
        ]

    def _unwind_continuous_joints(self, side: str):
        """Bring multi-turn joints (wrist rolls q4/q6) back into [-pi, pi].

        Every unscrew spins the wrist ~3 rad and nothing ever unwinds it, so q6
        drifts to -8, -11 rad... The tool pose is identical modulo 2*pi, so a
        reset is invisible physically but gives the planner (and IK) a sane seed.
        """
        robot, _, _, _, _ = self._arm(side)
        rid = robot.urdf
        for j in range(p.getNumJoints(rid)):
            info = p.getJointInfo(rid, j)
            if info[2] != p.JOINT_REVOLUTE or info[9] - info[8] < 2.5 * math.pi:
                continue
            q, dq = p.getJointState(rid, j)[:2]
            wrapped = q - 2.0 * math.pi * round(q / (2.0 * math.pi))
            if abs(wrapped - q) > 1e-9:
                p.resetJointState(rid, j, wrapped, dq)

    def _set_desk(self, present: bool):
        """The clamping desk is part of the cell only for table / robofixture work.  With the
        assembly on the floor it is not there at all (body and planning slab)."""
        if present == self._desk_present:
            return
        if present:
            self._desk_id = p.loadURDF(self._desk_args[0], self._desk_args[1],
                                       p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        else:
            p.removeBody(self._desk_id)
            self._desk_id = None
        self._desk_present = present
        print(f'[cell] clamping desk {"present" if present else "REMOVED (floor mode)"}', flush=True)

    def _scene_obstacles(self) -> list:
        """Static cell furniture the planner must respect — currently the clamping
        desk as one solid block from the floor to its top face, when present
        (`self._desk_present`, off in floor mode — see `set_mounting_mode`; returns
        an empty list in that case).

        The planner otherwise only sees assembly parts, so it could route an
        elbow or wrist through the tabletop.  As a block it also enforces the
        working rule: over the table every part of the arm stays above the
        surface.  The margin is small on purpose — the screw box is 0.31 m
        beside the table with drop points below table height, and a transit may
        start there; a slab that swallowed it would make that start invalid.

        The tool storages are NOT included here by default (they overlap some
        planned paths in ways that were only worked around, not modelled properly);
        pass `plan_storage_obstacles=True` to add them as real obstacles instead.
        """
        from modular_planner import ObstacleBox
        c = self.clamping_desk_centre
        half = 0.6 + self.table_slab_margin_m
        lo = np.array([c[0] - half, c[1] - half, 0.0])
        hi = np.array([c[0] + half, c[1] + half, self.clamping_desk_top_z + self.table_slab_top_margin_m])
        boxes = [ObstacleBox.from_pybullet_aabb(lo, hi)] if self._desk_present else []
        if getattr(self, 'plan_storage_obstacles', False):      # opt-in: the tool storages are real obstacles
            for sid in getattr(self, '_tool_storage_ids', []):
                slo, shi = p.getAABB(sid)
                boxes.append(ObstacleBox.from_pybullet_aabb(np.array(slo), np.array(shi)))
        return boxes

    def _static_obstacles(self) -> list:
        """Everything that does not move: the table block + one AABB per assembly part."""
        from modular_planner import ObstacleBox
        obstacles = self._scene_obstacles()
        if self.assembly is not None and self.assembly.part_ids:
            for pid in self._planning_part_ids():
                lo, hi = p.getAABB(pid)
                obstacles.append(ObstacleBox.from_pybullet_aabb(np.array(lo), np.array(hi)))
        return obstacles

    _coop_mode = False

    def _planning_part_ids(self) -> list:
        """Assembly parts that count as planning obstacles.  During the cooperative task none do:
        one AABB around the platform legs and the carrier would swallow the grip poses."""
        if self._coop_mode or self.assembly is None:
            return []
        return list(self.assembly.part_ids.values())

    def _planner_start_conf(self, side: str, margin: float = 0.02) -> np.ndarray:
        """Joint state for cuMotion, pulled `margin` rad inside the URDF limits.

        IK parks the wrist ON its limits when the tool points straight down at
        full reach (q5 = ±2.182, q2 = ±1.309 in the logs) and cuMotion rejects a
        start state on a limit.  Planning from a hair inside is physically the
        same pose; the first waypoint draws the arm in.
        """
        robot, _, _, _, _ = self._arm(side)
        rid = robot.urdf
        conf = []
        for j in range(p.getNumJoints(rid)):
            info = p.getJointInfo(rid, j)
            if info[2] == p.JOINT_FIXED:
                continue
            q = p.getJointState(rid, j)[0]
            lo, hi = info[8], info[9]
            if hi - lo < 2.5 * math.pi:          # a real limit, not a multi-turn joint
                q = min(max(q, lo + margin), hi - margin)
            conf.append(q)
        return np.array(conf, dtype=np.float64)

    def _tool_offset(self, side: str) -> np.ndarray:
        """T_flange->tool: the fixed transform from the robot's own tcp_frame (the
        frame cuMotion plans, at the tool coupling) to the TCP of the tool it
        holds (the point every G-code line commands).  For the screwdriver this
        is ~0.22 m.  Identity if no tool is coupled.
        """
        robot, tool, _, _, _ = self._arm(side)
        if not self.plan_tool_offset or not tool.is_coupled():
            return np.eye(4)
        tp, tq = tool.get_tool_pose()
        return np.linalg.inv(self._get_eef_pose_4x4(robot)) @ self._pose_to_4x4(tp, tq)

    def _flange_wps_to_tool(self, wps: list, T_off: np.ndarray) -> list:
        """Planner waypoints are flange poses; the processor and the export want
        tool-TCP poses: rewrite X..C of every motion line as (T_flange @ T_off)."""
        for wp in wps:
            if 'X' not in wp:
                continue
            T = np.eye(4)
            T[:3, :3] = R.from_euler('xyz', [wp['A'], wp['B'], wp['C']]).as_matrix()
            T[:3, 3] = [wp['X'], wp['Y'], wp['Z']]
            T = T @ T_off
            e = R.from_matrix(T[:3, :3]).as_euler('xyz')
            wp.update({'X': float(T[0, 3]), 'Y': float(T[1, 3]), 'Z': float(T[2, 3]),
                       'A': float(e[0]), 'B': float(e[1]), 'C': float(e[2])})
        return wps

    gnn_pipeline_transits = True     # False = GNN runs get straight-line transits again (old behaviour)

    def _plan_transit_gnn(self, side: str, target_pos, target_orn_q, feed: float, direct: list) -> list:
        """GNN transit inside the per-arm pipeline.  Both CRC robots go through the policy in one
        pass (robot_C = slot 0, robot_D = slot 1, as everywhere else); the other arm is asked to
        stay where it is, so the policy itself keeps the two apart.  Returns Cartesian G1 lines
        (tool poses, no joint path) ending exactly on the target; DIRECT if the GNN fails."""
        try:
            # The virtual robot's tip is the FLANGE (same z axis as the URDF tcp_frame), the pipeline
            # works with TOOL poses: convert with the tool offset, and back for the waypoints.
            T_off = {s_: self._tool_offset(s_) for s_ in ('C', 'D')}
            cur = {s_: self._pose_to_4x4(*self._tcp_pose(s_)) @ np.linalg.inv(T_off[s_]) for s_ in ('C', 'D')}
            tgt = {s_: (self._pose_to_4x4(target_pos, target_orn_q) @ np.linalg.inv(T_off[s_]) if s_ == side else cur[s_])
                   for s_ in ('C', 'D')}
            res = self._active_planner.plan_multi(
                current_eef_poses=[cur['C'], cur['D']], target_eef_poses=[tgt['C'], tgt['D']],
                obstacles=self._static_obstacles(), constraints=[], robot_names=['robot_C', 'robot_D'])
            wps = res[0 if side == 'C' else 1]
            if not wps:
                raise RuntimeError('GNN returned no waypoints')
            self._flange_wps_to_tool(wps, T_off[side])
            for wp in wps:
                wp['F'] = feed
            last = wps[-1]
            if np.linalg.norm([last['X'] - target_pos[0], last['Y'] - target_pos[1], last['Z'] - target_pos[2]]) > 1e-3:
                wps.append(direct[0])
            print(f'[transit robot_{side}] GNN path, {len(wps)} wps.', flush=True)
            return wps
        except Exception as e:
            print(f'[transit robot_{side}] GNN failed ({e}); DIRECT straight-line move (no collision avoidance).',
                  flush=True)
            return direct

    def _plan_transit_gcode(self, side: str, target_pos, target_orn_q, feed: float = 10000.0,
                            _retry: bool = True, start_conf=None):
        """Single-arm collision-free transit to a pose, planned NOW from the arm's
        actual state with the assembly and the other arm's links as obstacles.

        If cuMotion rejects the START state (it is inside the other arm's boxes),
        the arm first lifts straight up out of the clash and planning is retried
        from there.  A blind straight-line G1 is the very last resort only.
        """
        from modular_planner import PlannerType
        robot, _, _, _, _ = self._arm(side)
        orn_e = p.getEulerFromQuaternion(target_orn_q)
        direct = [{'G': 1, 'X': target_pos[0], 'Y': target_pos[1], 'Z': target_pos[2],
                   'A': orn_e[0], 'B': orn_e[1], 'C': orn_e[2], 'F': feed}]

        self._last_transit_joints = None
        if self._planner_type == PlannerType.GNN and self._active_planner is not None and self.gnn_pipeline_transits:
            return self._plan_transit_gnn(side, target_pos, target_orn_q, feed, direct)
        if self._planner_type != PlannerType.CUMOTION or self._active_planner is None:
            return direct
        if start_conf is None:
            self._unwind_continuous_joints(side)
        try:
            from modular_planner import ObstacleBox
            from scipy.spatial.transform import Rotation as R
            other = 'D' if side == 'C' else 'C'
            frozen = getattr(self, '_dry_obstacles', None)
            if start_conf is not None and frozen is not None:
                obstacles = list(frozen)          # dry rehearsal: scene + parked arm, built once
            else:
                obstacles = self._static_obstacles()
                for lo, hi in self._arm_obstacle_aabbs(other):
                    obstacles.append(ObstacleBox.from_pybullet_aabb(lo, hi))

            rid = robot.urdf
            if start_conf is None:
                start_conf = self._planner_start_conf(side)
            pos, q = p.getBasePositionAndOrientation(rid)
            T_base = np.eye(4)
            T_base[:3, :3] = R.from_quat(q).as_matrix()
            T_base[:3, 3] = pos

            # cuMotion moves the robot's tcp_frame (the tool coupling), but the
            # target is where the TOOL's tip must go — ~22 cm further out for
            # the screwdriver.  Plan the flange to target @ inv(T_off) and give
            # the processor the waypoints as tool poses again.
            T_off = self._tool_offset(side)
            wps = self._active_planner.plan(
                current_eef_pose=self._get_eef_pose_4x4(robot),
                target_eef_pose=self._pose_to_4x4(target_pos, target_orn_q) @ np.linalg.inv(T_off),
                obstacles=obstacles,
                constraints=[],
                robot_name=f'robot_{side}',
                start_conf=start_conf,
                robot_base_pose=T_base,
            )
            self._flange_wps_to_tool(wps, T_off)
            for wp in wps:
                wp['F'] = feed
            jp = getattr(self._active_planner, 'last_joint_path', None)
            if wps and self.joint_space_transits and jp is not None and len(jp) == len(wps):
                # Execute (and export) the planner's configuration path itself:
                # the arm then sweeps exactly what cuMotion collision-checked,
                # instead of PyBullet re-solving IK per Cartesian sample and
                # possibly landing on another wrist branch.
                wps = [{'G': 1, **{f'RA{i + 1}': float(v) for i, v in enumerate(q)}, 'F': feed} for q in jp]
            if wps:
                self._last_transit_joints = list(jp) if jp is not None and len(jp) == len(wps) else None
                extra = ''
                if jp is not None and len(jp) > 1:
                    steps = np.abs(np.diff(np.asarray(jp), axis=0))
                    extra = (f', joint path {steps.max(axis=1).sum():.2f} rad '
                             f'(largest single step {steps.max():.2f} rad, '
                             f'per joint {np.round(steps.sum(axis=0), 1).tolist()})')
                print(f'[transit robot_{side}] cuMotion path, {len(wps)} wps, '
                      f'{len(obstacles)} obstacle boxes{extra}.', flush=True)
                return wps
            reason = 'returned no path'
        except Exception as e:
            reason = str(e)

        # Planning failed.  If we are pressed against the other arm the start
        # state is inside its boxes: lift out of the clash first, then retry.
        if _retry and start_conf is None and self._arms_within(self.yield_margin_m):
            pos, q = self._tcp_pose(side)
            e = p.getEulerFromQuaternion(q)
            lift = [{'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2] + self.escape_dz_m,
                     'A': e[0], 'B': e[1], 'C': e[2], 'F': 5000.0}]
            print(f'[transit robot_{side}] cuMotion failed ({reason}); too close to the other '
                  f'arm — lifting +{self.escape_dz_m:.2f} m and re-planning.', flush=True)
            proc = self.g_code_processor_C if side == 'C' else self.g_code_processor_D
            self._run_single(proc, {'gcode': lift})
            if side == 'C':
                self.g_code_C += lift
            else:
                self.g_code_D += lift
            return self._plan_transit_gcode(side, target_pos, target_orn_q, feed, _retry=False)

        print(f'[transit robot_{side}] cuMotion failed ({reason}); DIRECT straight-line move '
              f'(no collision avoidance).', flush=True)
        return direct

    def _snap_orientation(self, side: str, pos, orn_q):
        """Short G-code move at the current XYZ to the exact planned orientation
        (the transit planner reaches the position; this makes the tool match)."""
        orn_e = p.getEulerFromQuaternion(orn_q)
        proc = self.g_code_processor_C if side == 'C' else self.g_code_processor_D
        gc = [{'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2],
               'A': orn_e[0], 'B': orn_e[1], 'C': orn_e[2], 'F': 1000.0}]
        if side == 'C':
            self.g_code_C += gc
        else:
            self.g_code_D += gc
        self._run_single(proc, {'gcode': gc})

    def _run_single(self, proc, seg: dict, step_sims: int = 5, on_step=None):
        if 'make_gcode' in seg:
            seg['gcode'] = seg['make_gcode']()
        proc.g_code = list(seg['gcode'])
        for _ in proc:
            for _ in range(step_sims):
                self.stepSimulation()
            if on_step is not None:
                on_step()
        for _ in range(20):
            self.stepSimulation()

    def _planned_segment(self, name: str, side: str, target_pos, target_q,
                         tail: list, yields: bool, feed: float = 10000.0) -> dict:
        """A lazily planned move that executes in CHUNKS.

        Each time the segment loads it plans from the arm's real pose against
        the other arm's current spheres, but executes only the first
        transit_chunk_wps waypoints; the loop then loads it again (re-plans)
        until the remaining path fits in one chunk.  So the path is re-validated
        every ~10 cm as the other arm moves — the same idea as the joint move's
        replanning loop.  `tail` (reorient / approach lines) runs once, at the end.
        """
        seg = {'name': name, 'yields': yields, 'rounds': 0,
               'target': (np.asarray(target_pos, dtype=float), np.asarray(target_q, dtype=float),
                          list(tail), feed)}

        def make():
            seg['rounds'] += 1
            gc = self._plan_transit_gcode(side, np.asarray(target_pos, dtype=float), target_q, feed=feed)
            jp = getattr(self, '_last_transit_joints', None)   # None when DIRECT
            if len(gc) > self.transit_chunk_wps and seg['rounds'] < self.transit_max_rounds:
                seg['replan'] = True
                seg['joint_path'] = jp[:self.transit_chunk_wps] if jp else None
                return gc[:self.transit_chunk_wps]
            seg['replan'] = False
            seg['joint_path'] = jp if jp else None
            return gc + list(tail)

        seg['make_gcode'] = make
        return seg

    def _transit_segment(self, side: str, tp, part_name: str):
        """Chunked, re-planned transit to the screw's standoff, then a short
        reorient so the tool matches the planned approach orientation exactly."""
        target_pos = tp.positions[:, 0]
        target_q = tp.orientations[:, 0]
        orn_e = p.getEulerFromQuaternion(target_q)
        # Reorient only.  (Earlier versions bracketed this with M0/T12/T11,
        # which export as programmed stops + CALL SEARCH_TURN / SCREW_PROGRAM
        # at every standoff — the unscrew program belongs in the engage segment.)
        tail = [{'G': 1, 'X': target_pos[0], 'Y': target_pos[1], 'Z': target_pos[2],
                 'A': orn_e[0], 'B': orn_e[1], 'C': orn_e[2], 'F': 1000.0}]
        return self._planned_segment(f'transit {part_name}', side, target_pos, target_q, tail, yields=True)

    @contextlib.contextmanager
    def _not_exported(self):
        """Run sim motions without them entering the G-code export.

        Tool pick-up / hand-over is done by hand in the cell, so the exported
        program must start at the first task motion.  Both streams are restored
        because _execute_move_both appends hold-still waypoints to the idle arm too.
        """
        n_C, n_D = len(self.g_code_C), len(self.g_code_D)
        try:
            yield
        finally:
            del self.g_code_C[n_C:]
            del self.g_code_D[n_D:]

    def _station_anchors_in_path_frame(self):
        """[base_C, base_D] as xyz in the assembly's path frame.

        Path points are part-local, unscaled (mm for the benchmarks); the sim
        places them with  world = spawn_point + R(spawn_orient) @ (p * scale),
        so  p = R^T @ (world - spawn_point) / scale.
        """
        if self.assembly is None:
            return None
        inv_pos, inv_orn = p.invertTransform(self.assembly.spawn_point, self.assembly.spawn_orient)
        out = []
        for robot in (self.robot_C, self.robot_D):
            base, _ = p.getBasePositionAndOrientation(robot.urdf)
            local, _ = p.multiplyTransforms(inv_pos, inv_orn, base, [0, 0, 0, 1])
            out.append([float(v) / self.assembly.scaleFactor for v in local])
        return out

    def _equip_both_screwdrivers(self, active_eef_list_index: int):
        """Make sure both arms hold their screwdrivers, swapping tools if needed.
        Not exported: in the cell tools are fetched by hand before the program runs."""
        with self._not_exported():
            self._equip_both_screwdrivers_sim(active_eef_list_index)

    def _equip_both_screwdrivers_sim(self, active_eef_list_index: int):
        if active_eef_list_index < 0:
            self.unstore_screwdriver()
        elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.screwdriver:
            if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.SRG:
                self.midway_point_to_storage()
                self.store_SRG()
            elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                self.midway_point_to_storage()
                self.store_millingtool()
            self.g_code_processor_D.active_endeffector = -1
            self.unstore_screwdriver()
            self.midway_point()

        c_eef_idx = self.g_code_processor_C.active_endeffector
        if c_eef_idx < 0:
            self.unstore_screwdriver_C()
        elif self.g_code_processor_C.endeffector_list[c_eef_idx] != self.screwdriver_C:
            self.g_code_processor_C.active_endeffector = -1
            self.unstore_screwdriver_C()

    def unscrew_pipeline(self, ops_C: list, ops_D: list, priority_default: str = 'C'):
        """Run each arm through its whole list of screws as one continuous chain.

        ops_X: list of {'part_name', 'path', 'direction'} for that arm, in order.
        Each arm runs transit -> engage -> pull out -> drop -> retreat for every
        screw back-to-back, never waiting for the other arm at any boundary.  The
        only coupling is the live yield arbitration in _interleave_streams.
        Transits are planned just-in-time from the arm's real pose with the other
        arm's current AABB as an obstacle.
        """
        for op in ops_C + ops_D:
            if op['part_name'] not in self.assembly.part_ids:
                print(f'[unscrew_pipeline] ERROR: "{op["part_name"]}" not in assembly.', flush=True)
                return False

        def chain(side, ops, colour):
            _, screwdriver, _, _, _ = self._arm(side)
            segs, first_tp = [], None
            for k, op in enumerate(ops):
                tp = self._screw_tool_path(op['part_name'], op['path'], op['direction'], screwdriver)
                tp.draw(False, colour)
                if k == 0:
                    first_tp = tp      # first standoff is reached by the joint move below
                else:
                    segs.append(self._transit_segment(side, tp, op['part_name']))
                segs += self._screw_segments(side, op['part_name'], tp, op['path'])
            return segs, first_tp

        stream_C, tp0_C = chain('C', ops_C, [0, 0, 1])
        stream_D, tp0_D = chain('D', ops_D, [1, 0, 0])

        # Both arms come out of the tool fetch in arbitrary poses, possibly close
        # together — the hard case.  Use the joint replanning move (both arms
        # planned against each other every cycle) to reach the two first
        # standoffs; the independent chains then start from known-safe places.
        if tp0_C is not None and tp0_D is not None:
            self._execute_move_both(
                target_pos_C=tp0_C.positions[:, 0], target_orn_q_C=tp0_C.orientations[:, 0],
                target_pos_D=tp0_D.positions[:, 0], target_orn_q_D=tp0_D.orientations[:, 0],
                feed_rate_C=10000.0, feed_rate_D=10000.0, tool_targets=True)
            for side, tp0 in (('C', tp0_C), ('D', tp0_D)):
                self._snap_orientation(side, tp0.positions[:, 0], tp0.orientations[:, 0])
        elif tp0_C is not None:
            self._run_single(self.g_code_processor_C, self._transit_segment('C', tp0_C, ops_C[0]['part_name']))
        elif tp0_D is not None:
            self._run_single(self.g_code_processor_D, self._transit_segment('D', tp0_D, ops_D[0]['part_name']))

        print(f'[unscrew_pipeline] robot_C: {len(ops_C)} screws, robot_D: {len(ops_D)} screws '
              f'— running as independent chains.', flush=True)
        ok, _ = self._interleave_streams(stream_C, stream_D, priority_default=priority_default)
        if not ok:
            print('[unscrew_pipeline] aborted on contact before any screw was attached.', flush=True)
        return ok

    # ------------------------------------------------------------------ export gate
    def _joint_clearance_report(self, side: str, min_ok_m: float = 0.03, steps: int = 6) -> str:
        """EXPORT GATE for one arm.  Walks the exported program as the controller
        will execute it: joint lines (RA1..RA6) directly, Cartesian lines through
        the planner's IK (joint limits, parallelogram and cable-sum limits, self
        collision) from the previous configuration.  Every configuration, with
        joint-linear interpolation between consecutive lines, is checked with
        the planner's spheres (arm, flange bars, tool) against the REAL table
        (no planning margin) and the assembly parts.  A Cartesian line without
        a solution = a software-limit alarm on the cell (Run 8, Z445 drop).
        Prints the offending segments, returns header comment lines."""
        planner = self._active_planner
        if planner is None or not hasattr(planner, 'robot_spheres'):
            return ''
        gc = self.g_code_C if side == 'C' else self.g_code_D
        robot, _, _, _, _ = self._arm(side)
        bp, bq = p.getBasePositionAndOrientation(robot.urdf)
        Rb = np.array(p.getMatrixFromQuaternion(bq)).reshape(3, 3); pb = np.asarray(bp)
        T_base = self._pose_to_4x4(bp, bq)
        T_off = self._tool_offset(side)
        c = self.clamping_desk_centre
        boxes = [(np.array([c[0] - 0.6, c[1] - 0.6, 0.0]), np.array([c[0] + 0.6, c[1] + 0.6, self.clamping_desk_top_z]))] if self._desk_present else []
        # cooperative task: the arms hold the carrier and reach between the platform legs, one AABB per
        # part would flag every grip, so only the desk counts there
        if self.assembly is not None and self.assembly.part_ids and not getattr(self.assembly, 'cooperative', None):
            for pid in self.assembly.part_ids.values():
                lo, hi = p.getAABB(pid)
                boxes.append((np.asarray(lo), np.asarray(hi)))
        seg, prev, worst, no_ik = 'start', None, {}, {}
        pos, qq = (np.asarray(x, dtype=float) for x in self._tcp_pose(side))
        q_last = self._planner_start_conf(side)
        # the program starts at the header pose = the arm's first joint line, if any
        for line in gc:
            if all(f'RA{i}' in line for i in range(1, 7)):
                q_last = np.array([line[f'RA{i}'] for i in range(1, 7)], dtype=float)
                break
        n_joint = n_cart = 0
        for line in gc:
            if 'comment' in line:
                seg = line['comment']
                continue
            q1 = None
            if all(f'RA{i}' in line for i in range(1, 7)):
                q1 = np.array([line[f'RA{i}'] for i in range(1, 7)], dtype=float)
                n_joint += 1
            elif 'X' in line or 'A' in line:
                e = p.getEulerFromQuaternion(qq)
                npos = np.array([line.get('X', pos[0]), line.get('Y', pos[1]), line.get('Z', pos[2])], dtype=float)
                nq = np.asarray(p.getQuaternionFromEuler([line.get('A', e[0]), line.get('B', e[1]), line.get('C', e[2])]), dtype=float)
                pos, qq = npos, nq
                n_cart += 1
                sol = planner.ik(self._pose_to_4x4(npos, nq) @ np.linalg.inv(T_off), robot_name=f'robot_{side}',
                                 seed=q_last, robot_base_pose=T_base)
                if sol is None:
                    no_ik.setdefault(seg, []).append(f'X{npos[0] * 1000:.0f} Y{npos[1] * 1000:.0f} Z{npos[2] * 1000:.0f}')
                    continue
                q1 = np.asarray(sol, dtype=float)
            if q1 is None:
                continue
            seq = [q1] if prev is None else [prev + (q1 - prev) * t for t in np.linspace(0.0, 1.0, steps + 1)[1:]]
            prev = q_last = q1
            for q in seq:
                for cl, r in planner.robot_spheres(q, robot_name=f'robot_{side}'):
                    w = Rb @ np.asarray(cl, dtype=float) + pb
                    for lo, hi in boxes:
                        d = np.linalg.norm(np.maximum(0.0, np.maximum(lo - w, w - hi))) - r
                        if d < 0 and np.all((w > lo) & (w < hi)):
                            d = -(np.min(np.minimum(w - lo, hi - w)) + r)
                        if seg not in worst or d < worst[seg]:
                            worst[seg] = d
        bad = {s_: d for s_, d in worst.items() if d < min_ok_m}
        hdr = ''
        if no_ik:
            n = sum(len(v) for v in no_ik.values())
            print(f'[EXPORT GATE robot_{side}] {n} Cartesian line(s) have NO joint solution within the limits '
                  f'(software-limit alarm on the cell):', flush=True)
            for s_, pts in no_ik.items():
                print(f'    {s_}: {", ".join(pts[:3])}{" ..." if len(pts) > 3 else ""}', flush=True)
            hdr += (f';EXPORT GATE: {n} Cartesian line(s) UNREACHABLE within joint limits '
                    f'(first: {next(iter(no_ik))}) — DO NOT RUN\n')
        if bad:
            print(f'[EXPORT GATE robot_{side}] {len(bad)} segment(s) with less than {min_ok_m * 100:.0f} cm clearance '
                  f'to the table/assembly:', flush=True)
            for s_, d in sorted(bad.items(), key=lambda kv: kv[1])[:20]:
                print(f'    {d * 1000:+6.0f} mm  {s_}', flush=True)
            hdr += (f';EXPORT GATE: {len(bad)} segment(s) below {min_ok_m * 100:.0f} cm clearance to table/assembly '
                    f'(worst {min(bad.values()) * 1000:+.0f} mm) — DO NOT RUN without checking the console list\n')
        if not no_ik and not bad:
            m = min(worst.values()) if worst else float('nan')
            print(f'[EXPORT GATE robot_{side}] OK: {n_joint} joint + {n_cart} Cartesian lines, all reachable, '
                  f'minimum clearance to the table/assembly {m * 1000:+.0f} mm', flush=True)
            hdr = f';EXPORT GATE: {n_joint} joint + {n_cart} Cartesian lines reachable, clear of table/assembly by >= {m * 1000:.0f} mm\n'
        k = getattr(self, '_dry_ik_failures', 0)
        if k:
            hdr += f';EXPORT GATE: the kinematic execution could not solve {k} Cartesian line(s) (see console) — DO NOT RUN\n'
        return hdr

    # ------------------------------------------------------------------ TPG
    def _live_spheres(self, side: str) -> np.ndarray:
        """The arm right now as (N, 4) world spheres: cuMotion's collision
        spheres at the current joint state plus one sphere bounding the tool."""
        robot, screwdriver, _, _, _ = self._arm(side)
        rows = []
        planner = self._active_planner
        if planner is not None and hasattr(planner, 'robot_spheres'):
            rid = robot.urdf
            joints = [j for j in range(p.getNumJoints(rid)) if p.getJointInfo(rid, j)[2] != p.JOINT_FIXED]
            conf = np.array([p.getJointState(rid, j)[0] for j in joints], dtype=np.float64)
            base_pos, base_q = p.getBasePositionAndOrientation(rid)
            Rb = np.array(p.getMatrixFromQuaternion(base_q)).reshape(3, 3)
            for c, r in planner.robot_spheres(conf, robot_name=f'robot_{side}'):
                w = Rb @ np.asarray(c, dtype=float) + np.asarray(base_pos)
                rows.append((w[0], w[1], w[2], r))
        else:
            for lo, hi in self._arm_link_aabbs(side):
                c = 0.5 * (np.asarray(lo) + np.asarray(hi))
                rows.append((c[0], c[1], c[2], 0.5 * float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))))
        lo, hi = self._body_aabb(screwdriver.urdf)
        c = 0.5 * (lo + hi)
        rows.append((c[0], c[1], c[2], 0.5 * float(np.linalg.norm(hi - lo))))
        return np.asarray(rows, dtype=float)

    def _wait_pose(self, side: str, toward=None):
        """Where a deprioritised arm waits (Plan B): 0.5 m above its OWN screw
        box's approach point, tool down.  The box is on the arm's own side by
        construction, outside the danger box, and it is where the arm already
        is after every drop — so a wait costs no extra travel.  Robot_C: NE of
        the table; robot_D: south of it."""
        _, _, _, _, idx = self._arm(side)
        pos = np.array(self.storage_spaces[idx]['approachpoint'], dtype=float)
        pos[2] += 0.5
        q = p.getQuaternionFromEuler([0.0, 0.0, DROP_YAW_RAD])
        # Wrist heading at the wait pose: the roll closest (in joint space) to
        # the configuration the arm will need at its first screw, so the wait
        # pose is already "facing the work" and the first transit is short.
        seed = None
        if toward is not None:
            planner = self._active_planner
            if planner is not None and hasattr(planner, 'ik') and self.plan_tool_offset:
                robot, _, _, _, _ = self._arm(side)
                bp, bq = p.getBasePositionAndOrientation(robot.urdf)
                seed = planner.ik(self._pose_to_4x4(toward[0], toward[1]) @ np.linalg.inv(self._tool_offset(side)),
                                  robot_name=f'robot_{side}', seed=self._planner_start_conf(side),
                                  robot_base_pose=self._pose_to_4x4(bp, bq), obstacles=self._static_obstacles())
        q = self._pick_screw_roll(side, pos, q, f'wait@box_{side}', seed=seed, avoid_storage=self.wait_avoid_storage)
        return pos, q

    def _arm_in_zone(self, side: str) -> bool:
        return self.danger_zone is not None and self.danger_zone.contains_spheres(self._live_spheres(side))

    def _live_move_segment(self, name: str, side: str, pos, q, exact: bool = True) -> dict:
        """A planned move made NOW (not rehearsed) for the Plan-B detours."""
        gc = self._plan_transit_gcode(side, np.asarray(pos, dtype=float), q)
        if exact:
            e = p.getEulerFromQuaternion(q)
            gc = gc + [{'G': 1, 'X': float(pos[0]), 'Y': float(pos[1]), 'Z': float(pos[2]),
                        'A': e[0], 'B': e[1], 'C': e[2], 'F': 1000.0}]
        return {'name': name, 'gcode': gc, 'live': True}

    def _spheres_at(self, side: str, q, T_off, T_base) -> np.ndarray:
        """(N, 4) world spheres of the arm at joint vector q, from kinematics only:
        cuMotion's collision spheres + one sphere bounding the tool at the TCP."""
        planner = self._active_planner
        rows = []
        Rb, bp = T_base[:3, :3], T_base[:3, 3]
        for c, r in planner.robot_spheres(np.asarray(q, dtype=np.float64), robot_name=f'robot_{side}'):
            w = Rb @ np.asarray(c, dtype=float) + bp
            rows.append((w[0], w[1], w[2], r))
        planner._select(f'robot_{side}')
        T_fl = T_base @ np.array(planner._kinematics.pose(
            np.asarray(q, dtype=np.float64), planner._tool_frame, planner._kinematics.base_frame_name()).matrix())
        tip = (T_fl @ T_off)[:3, 3]
        rows.append((tip[0], tip[1], tip[2], self._tool_bound_radius(side)))
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _body_aabb(bid: int):
        """AABB over the base AND all links (p.getAABB(body) alone is only the base
        link, which for the screwdriver is an empty frame: a 2 mm box that made
        the tool invisible to every sphere/AABB check until 2026-09-18)."""
        lo, hi = (np.asarray(x, dtype=float) for x in p.getAABB(bid))
        for j in range(p.getNumJoints(bid)):
            l2, h2 = p.getAABB(bid, j)
            lo, hi = np.minimum(lo, l2), np.maximum(hi, h2)
        return lo, hi

    def _tool_bound_radius(self, side: str) -> float:
        _, screwdriver, _, _, _ = self._arm(side)
        lo, hi = self._body_aabb(screwdriver.urdf)
        return 0.5 * float(np.linalg.norm(hi - lo))

    def _dry_rehearse_stream(self, side: str, stream: list):
        """PLANNING-ONLY rehearsal: no physics.  Walks the chain in joint space
        from the arm's current configuration: planned moves are planned whole
        (from the previous segment's end configuration, other arm as parked),
        fixed G-code lines are resolved with cuMotion's IK, and every
        configuration is turned into swept spheres by forward kinematics.
        The result is the same fixed segment data the physical rehearsal gives
        (gcode, spheres, start_pose), in seconds instead of minutes.
        """
        planner = self._active_planner
        robot, screwdriver, _, _, _ = self._arm(side)
        T_off = self._tool_offset(side)
        bp, bq = p.getBasePositionAndOrientation(robot.urdf)
        T_base = self._pose_to_4x4(bp, bq)
        q = self._planner_start_conf(side)
        pos, qq = (np.asarray(x, dtype=float) for x in self._tcp_pose(side))
        static = self._static_obstacles()
        from modular_planner import ObstacleBox
        other = 'D' if side == 'C' else 'C'
        self._dry_obstacles = list(static) + [ObstacleBox.from_pybullet_aabb(lo, hi)
                                              for lo, hi in self._arm_obstacle_aabbs(other)]
        self._rehearsing = side
        import time as _time
        t_start = _time.perf_counter()
        print(f'[REHEARSAL robot_{side}] DRY (kinematic) rehearsal — {len(stream)} segments', flush=True)

        def ik_line(line, q_seed):
            """Configuration for a Cartesian line (missing words keep the last pose)."""
            nonlocal pos, qq
            e = p.getEulerFromQuaternion(qq)
            npos = np.array([line.get('X', pos[0]), line.get('Y', pos[1]), line.get('Z', pos[2])], dtype=float)
            ne = [line.get('A', e[0]), line.get('B', e[1]), line.get('C', e[2])]
            nq = np.asarray(p.getQuaternionFromEuler(ne), dtype=float)
            sol = planner.ik(self._pose_to_4x4(npos, nq) @ np.linalg.inv(T_off), robot_name=f'robot_{side}',
                             seed=q_seed, robot_base_pose=T_base)
            pos, qq = npos, nq
            return sol

        for k, seg in enumerate(stream):
            seg['start_pose'] = (pos.copy(), qq.copy())
            samples = [self._spheres_at(side, q, T_off, T_base)]
            if 'make_gcode' in seg:
                tpos, tq, tail, feed = seg['target']
                gc = self._plan_transit_gcode(side, tpos, tq, feed=feed, start_conf=q)
                jp = self._last_transit_joints
                if jp:
                    for cfg in jp:
                        samples.append(self._spheres_at(side, cfg, T_off, T_base))
                    q = np.asarray(jp[-1], dtype=float)
                    pos, qq = tpos.copy(), tq.copy()
                else:
                    # DIRECT fallback: straight line in Cartesian space, IK along it
                    for t in np.linspace(0.0, 1.0, 8)[1:]:
                        sol = ik_line({'X': pos[0] + t * (tpos[0] - pos[0]), 'Y': pos[1] + t * (tpos[1] - pos[1]),
                                       'Z': pos[2] + t * (tpos[2] - pos[2])}, q)
                        if sol is not None:
                            q = sol
                            samples.append(self._spheres_at(side, q, T_off, T_base))
                    pos, qq = tpos.copy(), tq.copy()
                for line in tail:
                    if 'X' in line or 'A' in line:
                        sol = ik_line(line, q)
                        if sol is not None:
                            q = sol
                            samples.append(self._spheres_at(side, q, T_off, T_base))
                seg['gcode'] = list(gc) + list(tail)
                del seg['make_gcode']
                seg.pop('replan', None)
            else:
                for line in seg['gcode']:
                    if 'X' in line or 'A' in line:
                        sol = ik_line(line, q)
                        if sol is None:
                            print(f'[REHEARSAL robot_{side}] "{seg["name"]}": no IK for a line, '
                                  f'keeping the previous configuration', flush=True)
                            continue
                        q = sol
                        samples.append(self._spheres_at(side, q, T_off, T_base))
            seg['spheres'] = np.vstack(samples)
            print(f'[REHEARSAL robot_{side}] {k + 1}/{len(stream)} "{seg["name"]}": '
                  f'{len(seg["gcode"])} lines, {len(samples)} configurations '
                  f'({_time.perf_counter() - t_start:.1f} s so far)', flush=True)
        self._dry_obstacles = None
        self._rehearsing = None

    # -- rehearsal cache ------------------------------------------------------
    def _tpg_cache_key(self, ops_C, ops_D) -> str:
        import hashlib, json
        planner = self._active_planner
        key = {
            'assembly': getattr(self, '_loaded_assembly_file', None), 'mode': self._mounting_mode,
            'ops_C': [(o['part_name'], np.round(np.asarray(o['path']), 3).tolist()) for o in ops_C],
            'ops_D': [(o['part_name'], np.round(np.asarray(o['path']), 3).tolist()) for o in ops_D],
            'boxes': [np.round(np.asarray(sp['approachpoint']), 3).tolist() for sp in self.storage_spaces],
            'planner': (getattr(planner, '_urdf_overrides', None), getattr(planner, '_waypoint_step', None)),
            'flags': (self.plan_tool_offset, self.joint_space_transits, self.tpg_rehearsal, DROP_YAW_RAD),
            # scene flags that change what _scene_obstacles/_static_obstacles return without changing
            # 'mode' or 'boxes' above (found 2026-09-21: their absence let a stale rehearsal from a
            # different desk state get replayed silently — see PROGRESS_LOG.md 2026-09-21):
            'scene_flags': (getattr(self, '_desk_present', True),
                             getattr(self, 'plan_storage_obstacles', False),
                             getattr(self, 'ignore_storage_contacts', False)),
            'version': 13,  # 10: q6 sign; 11: tool spheres, slab margins; 12: D tool at +x (controller-verified);
                             # 13: desk-present / storage-obstacle scene flags added to the key
        }
        return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _tpg_cache_path(self, key: str) -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpg_cache')
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{key}.pkl')

    def _tpg_cache_save(self, key, stream_C, stream_D):
        import pickle
        data = {}
        for side, stream in (('C', stream_C), ('D', stream_D)):
            data[side] = [{'name': sg['name'], 'gcode': sg['gcode'], 'spheres': sg['spheres'],
                           'cycle': sg.get('cycle'), 'start_pose': sg.get('start_pose')} for sg in stream]
        with open(self._tpg_cache_path(key), 'wb') as fh:
            pickle.dump(data, fh)
        print(f'[tpg] rehearsal cached: {self._tpg_cache_path(key)}', flush=True)

    def _tpg_cache_load(self, key, stream_C, stream_D) -> bool:
        import pickle
        path = self._tpg_cache_path(key)
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        for side, stream in (('C', stream_C), ('D', stream_D)):
            if [sg['name'] for sg in stream] != [d['name'] for d in data[side]]:
                print(f'[tpg] cache {key} does not match the current chain for robot_{side} — ignored', flush=True)
                return False
        for side, stream in (('C', stream_C), ('D', stream_D)):
            for sg, d in zip(stream, data[side]):
                sg['gcode'], sg['spheres'], sg['start_pose'] = d['gcode'], d['spheres'], d['start_pose']
                sg.pop('make_gcode', None)
                sg.pop('replan', None)
        print(f'[tpg] rehearsal loaded from cache {path} (delete the file to force a new rehearsal)', flush=True)
        return True

    def _rehearse_stream(self, side: str, stream: list):
        """Run one arm's whole segment chain ALONE (the other arm parked where
        it is) and turn it into fixed data: every lazily planned segment is
        resolved to its executed G-code (all chunks), and every segment gets
        the spheres its arm swept, sampled each processor step.  on_done hooks
        run as in a real execution (screws come out); the caller restores the
        world afterwards."""
        proc = self.g_code_processor_C if side == 'C' else self.g_code_processor_D
        self._rehearsing = side
        print(f'[REHEARSAL robot_{side}] start — {len(stream)} segments, solo, not exported', flush=True)
        for k, seg in enumerate(stream):
            print(f'[REHEARSAL robot_{side}] {k + 1}/{len(stream)} "{seg["name"]}"', flush=True)
            seg.setdefault('start_pose', tuple(np.asarray(x) for x in self._tcp_pose(side)))
            samples = [self._live_spheres(side)]
            count = [0]

            warned = [False]

            def sample():
                count[0] += 1
                if count[0] % self.tpg_sample_every == 0:
                    samples.append(self._live_spheres(side))
                if not warned[0] and count[0] % 10 == 0 and self._arms_in_contact():
                    pair = self._closest_link_pair(0.0)
                    where = (f' [C:{self._describe_link(pair[1], pair[2])} <-> '
                             f'D:{self._describe_link(pair[3], pair[4])}]') if pair else ''
                    print(f'[REHEARSAL robot_{side}] WARNING: contact with the parked arm during '
                          f'"{seg["name"]}"{where}', flush=True)
                    warned[0] = True

            if 'make_gcode' in seg:
                gcode = []
                while True:
                    chunk = seg['make_gcode']()
                    gcode += chunk
                    self._run_single(proc, {'gcode': chunk}, on_step=sample)
                    if not seg.get('replan'):
                        break
                seg['gcode'] = gcode
                del seg['make_gcode']
                seg.pop('replan', None)
            else:
                self._run_single(proc, seg, on_step=sample)
            seg['spheres'] = np.vstack(samples)
            if seg.get('on_done'):
                seg['on_done']()
        print(f'[REHEARSAL robot_{side}] done', flush=True)
        self._rehearsing = None

    def _sync_marker(self, waiter: str, mover: str, why: str):
        """Insert a WAITM rendezvous into both exported programs at the arms'
        current poses (splitting an in-flight motion line)."""
        self._sync_no = getattr(self, '_sync_no', 0) + 1
        n = 1 + (self._sync_no - 1) % self.sync_marker_count
        for s in (waiter, mover):
            gc = self.g_code_C if s == 'C' else self.g_code_D
            pos, q = self._tcp_pose(s)
            e = p.getEulerFromQuaternion(q)
            marker = {'WAITM': n, 'why': why, 'seq': self._sync_no,
                      'total': getattr(self, '_sync_total', 0)}
            j = len(gc) - 1
            if j >= 0 and 'X' in gc[j] and not gc[j].get('_done'):
                line = gc[j]
                here = dict(line)
                here.update({'X': float(pos[0]), 'Y': float(pos[1]), 'Z': float(pos[2])})
                if 'A' in line:
                    here.update({'A': e[0], 'B': e[1], 'C': e[2]})
                gc[j] = here
                gc.append(marker)
                gc.append(line)
            else:
                gc.append(marker)
        print(f'[sync] WAITM({n}) — robot_{waiter} resumes after robot_{mover}: {why}', flush=True)

    # -- kinematic execution helpers ------------------------------------------
    def _dry_set_conf(self, side: str, q) -> None:
        """Put the arm at joint vector q without physics, and carry its tool
        body along (the coupling constraint only acts when the engine steps)."""
        robot, tool, _, _, _ = self._arm(side)
        rid = robot.urdf
        joints = [j for j in range(p.getNumJoints(rid)) if p.getJointInfo(rid, j)[2] != p.JOINT_FIXED][:6]
        for j, v in zip(joints, q):
            p.resetJointState(rid, j, float(v))
        if tool.is_coupled():
            # tool connector COM frame == robot tcp COM frame (identity fixed constraint)
            tcp_pos, tcp_q = p.getLinkState(rid, robot._convert_endeffector(tool._coupling_link)
                                            if tool._coupling_link else robot._default_endeffector_id,
                                            computeForwardKinematics=True)[:2]
            conn = tool._connector_id
            if conn == -1:
                p.resetBasePositionAndOrientation(tool.urdf, tcp_pos, tcp_q)
            else:
                b_pos, b_q = p.getBasePositionAndOrientation(tool.urdf)
                c_pos, c_q = p.getLinkState(tool.urdf, conn)[:2]
                inv_c = p.invertTransform(c_pos, c_q)
                rel_pos, rel_q = p.multiplyTransforms(*inv_c, b_pos, b_q)     # base in connector frame
                n_pos, n_q = p.multiplyTransforms(tcp_pos, tcp_q, rel_pos, rel_q)
                p.resetBasePositionAndOrientation(tool.urdf, n_pos, n_q)

    def _dry_resolve_segment(self, side: str, seg: dict, q0, step_rad: float = 0.04) -> list:
        """Turn a segment's G-code into a replay list of ('line', dict) and
        ('q', conf) entries: joint lines give their configuration directly,
        Cartesian lines are solved with the planner's IK from the previous
        configuration, and consecutive configurations are interpolated in
        joint space in steps of <= step_rad (so the tick-wise sphere/contact
        check sees the whole sweep)."""
        planner = self._active_planner
        robot, _, _, _, _ = self._arm(side)
        T_off = self._tool_offset(side)
        bp, bq = p.getBasePositionAndOrientation(robot.urdf)
        T_base = self._pose_to_4x4(bp, bq)
        pos, qq = (np.asarray(x, dtype=float) for x in self._tcp_pose(side))
        q = np.asarray(q0, dtype=float)
        out = []
        if 'make_gcode' in seg:
            seg['gcode'] = seg['make_gcode']()
            del seg['make_gcode']
        for line in seg['gcode']:
            out.append(('line', line))
            target = None
            if all(f'RA{i}' in line for i in range(1, 7)):
                target = np.array([line[f'RA{i}'] for i in range(1, 7)], dtype=float)
            elif 'X' in line or 'A' in line:
                e = p.getEulerFromQuaternion(qq)
                npos = np.array([line.get('X', pos[0]), line.get('Y', pos[1]), line.get('Z', pos[2])], dtype=float)
                nq = np.asarray(p.getQuaternionFromEuler([line.get('A', e[0]), line.get('B', e[1]),
                                                          line.get('C', e[2])]), dtype=float)
                sol = planner.ik(self._pose_to_4x4(npos, nq) @ np.linalg.inv(T_off), robot_name=f'robot_{side}',
                                 seed=q, robot_base_pose=T_base)
                pos, qq = npos, nq
                if sol is None:
                    self._dry_ik_failures = getattr(self, '_dry_ik_failures', 0) + 1
                    print(f'[tpg-dry robot_{side}] "{seg["name"]}": NO IK within limits for a Cartesian line '
                          f'{ {k: round(float(v), 3) for k, v in line.items() if k in "XYZABC"} } — the controller '
                          f'would hit a software limit here; holding in the replay', flush=True)
                else:
                    target = np.asarray(sol, dtype=float)
            if target is not None:
                d = float(np.max(np.abs(target - q)))
                n = max(1, int(np.ceil(d / step_rad)))
                for k in range(1, n + 1):
                    out.append(('q', q + (target - q) * (k / n)))
                q = target
        return out

    def _execute_tpg(self, tpg, step_sims: int = 5, dry: bool = False) -> bool:
        """Asynchronous execution of a TemporalPlanGraph in the sim, exporting
        as executed with one WAITM per satisfied edge.  Contact = hard
        fail-safe (abort; caller falls back).

        dry=True: kinematic replay (no physics).  Each tick moves each arm one
        interpolation step (<= 0.04 rad) along its rehearsed segment, the tool
        bodies are carried along, and the fail-safe is a sphere-gap test plus
        Bullet's collision detection on the meshes (performCollisionDetection,
        no integration).  Same export, same WAITMs; minutes instead of hours."""
        sim = self
        self._sync_no = 0
        self._sync_total = len(tpg.edges)      # one WAITM per satisfied edge
        procs = {'C': self.g_code_processor_C, 'D': self.g_code_processor_D}
        gens = {}
        credit = {'C': 0.0, 'D': 0.0}
        feed = {'C': self.feed_ref, 'D': self.feed_ref}
        replay = {}                            # dry: side -> list of ('line', dict) / ('q', conf)
        from modular_planner.tpg import sphere_gap
        if dry:
            self._dry_ik_failures = 0
            print('[tpg] execution: DRY (kinematic replay, no physics)', flush=True)

        class Driver:
            def start(self_, side, idx):
                seg = tpg.segs[side][idx]
                (sim.g_code_C if side == 'C' else sim.g_code_D).append(
                    {'comment': f'robot_{side} segment {idx + 1}/{len(tpg.segs[side])} '
                                f'(after SYNC {getattr(sim, "_sync_no", 0)}): {seg["name"]}'})
                if dry:
                    replay[side] = sim._dry_resolve_segment(side, seg, sim._planner_start_conf(side))
                    return
                procs[side].g_code = list(seg['gcode'])
                gens[side] = iter(procs[side])

            def step(self_, side):
                if dry:
                    lst = replay[side]
                    while lst:
                        kind, val = lst.pop(0)
                        if kind == 'line':
                            (sim.g_code_C if side == 'C' else sim.g_code_D).append(dict(val))
                            continue
                        sim._dry_set_conf(side, val)
                        return False
                    return True
                credit[side] += (feed[side] / sim.feed_ref) / sim.pace_divider
                if credit[side] < 1.0:
                    return False
                credit[side] -= 1.0
                try:
                    line = next(gens[side])
                except StopIteration:
                    return True
                if isinstance(line, dict):
                    (sim.g_code_C if side == 'C' else sim.g_code_D).append(line)
                    if line.get('F') is not None:
                        feed[side] = float(line['F'])
                return False

            def finished(self_, side, idx):
                seg = tpg.segs[side][idx]
                if seg.get('on_done'):
                    if dry:
                        # the hooks attach/release screws and dwell by stepping
                        # the engine; keep the bookkeeping, skip the stepping
                        real = sim.stepSimulation
                        sim.stepSimulation = lambda: None
                        try:
                            seg['on_done']()
                        finally:
                            sim.stepSimulation = real
                        return
                    seg['on_done']()

            def wait(self_, edge):
                side = edge.wait_side
                print(f'[tpg] robot_{side} waits: {edge.why}', flush=True)
                # Plan B: a waiting arm never sits inside the danger box.  If it
                # is in there, move it to its wait pose on the border first
                # (planned now, against the other arm's current spheres) and
                # come back to the segment's start pose when released.
                if sim.tpg_plan_b and sim.danger_zone is not None and sim._arm_in_zone(side):
                    pos, q = sim._wait_pose(side)
                    here, _ = sim._tcp_pose(side)
                    if np.linalg.norm(np.asarray(here) - pos) < 0.05:
                        return False            # already at the wait pose: nothing to do
                    out = sim._live_move_segment(f'wait-out {side}', side, pos, q, exact=False)
                    out['return_to'] = tpg.segs[side][tpg.idx[side]].get('start_pose')
                    tpg.insert_segment(side, tpg.idx[side], out)
                    print(f'[tpg] robot_{side} is inside the box while waiting -> Plan B: '
                          f'moving out to the border {np.round(pos, 2)}', flush=True)
                    return True
                return False

            def release(self_, edge):
                side = edge.wait_side
                prev = tpg.segs[side][tpg.idx[side] - 1] if tpg.idx[side] > 0 else {}
                if prev.get('live') and prev.get('return_to') is not None:
                    pos, q = prev['return_to']
                    back = sim._live_move_segment(f'return {side}', side, pos, q, exact=True)
                    tpg.insert_segment(side, tpg.idx[side], back)
                    print(f'[tpg] robot_{side} released -> returning to {np.round(pos, 2)} '
                          f'before "{tpg.segs[side][tpg.idx[side] + 1]["name"]}"', flush=True)
                sim._sync_marker(edge.wait_side, edge.until_side, edge.why)

            def tick(self_):
                if dry:
                    gap = sphere_gap(sim._live_spheres('C'), sim._live_spheres('D'))
                    p.performCollisionDetection()          # meshes, no integration
                    hit = sim._world_contacts(depth=-0.01)
                    if hit:
                        print(f'[tpg] CONTACT with the world (dry): {hit} — aborting (fail-safe).', flush=True)
                        return False
                    if gap < 0.0 or sim._arms_in_contact():
                        pair = sim._closest_link_pair(0.0)
                        where = (f' [C:{sim._describe_link(pair[1], pair[2])} <-> '
                                 f'D:{sim._describe_link(pair[3], pair[4])}]') if pair else ''
                        print(f'[tpg] CONTACT (dry, sphere gap {gap * 100:+.0f} cm){where} — the plan graph '
                              f'was wrong for this pair; aborting (fail-safe).', flush=True)
                        return False
                    return True
                for _ in range(step_sims):
                    sim.stepSimulation()
                if sim._arms_in_contact():
                    pair = sim._closest_link_pair(0.0)
                    where = (f' [C:{sim._describe_link(pair[1], pair[2])} <-> '
                             f'D:{sim._describe_link(pair[3], pair[4])}]') if pair else ''
                    print(f'[tpg] CONTACT{where} — the plan graph was wrong for this pair; '
                          f'aborting (fail-safe).', flush=True)
                    return False
                return True

            def deadlock(self_, waiting):
                print(f'[tpg] DEADLOCK: {waiting} — graph should be acyclic, aborting.', flush=True)

        return tpg.run(Driver())

    def unscrew_pipeline_tpg(self, ops_C: list, ops_D: list, priority_default: str = 'C'):
        """Offline, time-aware dual-arm batch (Temporal Plan Graph).

        1. joint move to the first standoffs (as the reactive pipeline);
        2. REHEARSAL: each arm runs its whole chain alone from a saved world
           state (the other arm parked at its standoff), fixing all G-code and
           recording swept spheres; the world is restored after each;
        3. GRAPH: zone turn-taking + pairwise sphere safety -> dependency edges;
        4. EXECUTION: both arms asynchronously, waiting only at edges, one
           WAITM per edge in the export.
        """
        from modular_planner import PlannerType as _PT
        if self._planner_type == _PT.GNN:
            print('[unscrew_pipeline_tpg] the GNN planner supports the reactive layer only '
                  '(no joint paths / collision spheres for the offline graph).', flush=True)
            return False

        from modular_planner.tpg import TemporalPlanGraph
        if self.danger_zone is None and self.tpg_rule == 'zone':
            print('[tpg] no danger zone (assembly not loaded in floor/table mode) '
                  '-> reactive pipeline.', flush=True)
            return self.unscrew_pipeline(ops_C, ops_D, priority_default)
        for op in ops_C + ops_D:
            if op['part_name'] not in self.assembly.part_ids:
                print(f'[tpg] ERROR: "{op["part_name"]}" not in assembly.', flush=True)
                return False

        def chain(side, ops, colour):
            """Every screw, the first included, starts with a rehearsed transit:
            both arms begin the batch at their wait poses on the box border."""
            _, screwdriver, _, _, _ = self._arm(side)
            segs = []
            for k, op in enumerate(ops):
                tp = self._screw_tool_path(op['part_name'], op['path'], op['direction'], screwdriver)
                tp.draw(False, colour)
                n0 = len(segs)
                segs.append(self._transit_segment(side, tp, op['part_name']))
                segs += self._screw_segments(side, op['part_name'], tp, op['path'])
                for sg in segs[n0:]:
                    sg['cycle'] = k            # k-th screw of this arm (schedule order)
            return segs

        print(f'[tpg] === ASYNCHRONOUS (TPG) execution: robot_C {len(ops_C)} screws, '
              f'robot_D {len(ops_D)} screws — rehearsing each arm alone first ===', flush=True)
        stream_C = chain('C', ops_C, [0, 0, 1])
        stream_D = chain('D', ops_D, [1, 0, 0])

        # -- initial state = both arms at their WAIT POSES (above their own
        # boxes), wrist already turned towards the first screw.  Rehearsal and
        # execution start from this same state.
        first = {'C': stream_C[0]['target'][:2] if stream_C else None,
                 'D': stream_D[0]['target'][:2] if stream_D else None}
        wait = {s_: self._wait_pose(s_, toward=first[s_]) for s_ in ('C', 'D')}
        print(f'[tpg] both arms to their wait poses: C {np.round(wait["C"][0], 2)}, '
              f'D {np.round(wait["D"][0], 2)}', flush=True)
        # In the cell the arms are placed at their wait poses by hand (the
        # tool pick-up area is not modelled), so this move is not exported:
        # the programs START at the wait poses, recorded in the file header.
        with self._not_exported():
            self._execute_move_both(
                target_pos_C=wait['C'][0], target_orn_q_C=wait['C'][1],
                target_pos_D=wait['D'][0], target_orn_q_D=wait['D'][1],
                feed_rate_C=10000.0, feed_rate_D=10000.0, tool_targets=True)
            for s_ in ('C', 'D'):
                self._snap_orientation(s_, wait[s_][0], wait[s_][1])
        self._start_pose_header = {}
        for s_ in ('C', 'D'):
            pos, qq = self._tcp_pose(s_)
            # header angles in the controller's convention (same conversion as
            # transform_eulers_in_gcode), so they compare with the WCS display
            e = R.from_quat(qq).as_euler('XYZ')
            q = self._planner_start_conf(s_)
            self._start_pose_header[s_] = (
                f';START POSE robot_{s_} (place the arm here before starting):\n'
                f';  tool  X{pos[0]*1000:.1f} Y{pos[1]*1000:.1f} Z{pos[2]*1000:.1f} '
                f'A{math.degrees(e[0]):.1f} B{math.degrees(e[1]):.1f} C{math.degrees(e[2]):.1f}\n'
                f';  axes  ' + ' '.join(
                    f'RA{i + 1}={math.degrees(v - (q[1] if (i == 2 and COMAU_AXIS3_ABSOLUTE) else 0.0)):.1f}'
                    for i, v in enumerate(q)) + '  (RA3 absolute, Comau convention)\n')

        # -- rehearsal, one arm at a time, from that state; the other arm stays
        # parked at its wait pose (where it waits in Plan B: if the rehearsed
        # paths are not safe against that, the wait pose is no good in
        # general).  The world is restored after each arm.
        import time as _time
        key = self._tpg_cache_key(ops_C, ops_D)
        if not (self.tpg_use_cache and self._tpg_cache_load(key, stream_C, stream_D)):
            state = p.saveState()
            for side, stream in (('C', stream_C), ('D', stream_D)):
                other = 'D' if side == 'C' else 'C'
                t0 = _time.perf_counter()
                print(f"[REHEARSAL robot_{side}] robot_{other} parked at its wait pose "
                      f"{np.round(wait[other][0], 2)}", flush=True)
                with self._not_exported():
                    if self.tpg_rehearsal == 'dry':
                        self._dry_rehearse_stream(side, stream)
                    else:
                        self._rehearse_stream(side, stream)
                        p.restoreState(state)
                for attr in ('sc_C_gripping_constraint', 'sc_gripping_constraint'):
                    cid = getattr(self, attr, None)
                    if cid is not None:
                        try:
                            p.removeConstraint(cid)
                        except Exception:
                            pass
                        setattr(self, attr, None)
                n_sph = sum(len(sg['spheres']) for sg in stream)
                print(f'[tpg] rehearsed robot_{side}: {len(stream)} segments, {n_sph} sphere samples, '
                      f'{_time.perf_counter() - t0:.1f} s', flush=True)
            p.removeState(state)
            if self.tpg_use_cache:
                self._tpg_cache_save(key, stream_C, stream_D)

        # -- graph
        tpg = TemporalPlanGraph(stream_C, stream_D)
        if self.tpg_rule == 'segment':
            tpg.add_segment_rule(first=priority_default, margin=self.tpg_margin_m)
        else:
            tpg.add_zone_rule(self.danger_zone, first=priority_default, margin=self.tpg_margin_m)
        print(f'[tpg] rule = {self.tpg_rule}, margin = {self.tpg_margin_m} m', flush=True)
        print('[tpg] ' + tpg.describe().replace('\n', '\n[tpg] '), flush=True)
        if not tpg.check_acyclic():
            print('[tpg] graph has a cycle — refusing to execute.', flush=True)
            return False

        # -- execution
        print(f'[unscrew_pipeline_tpg] robot_C: {len(ops_C)} screws, robot_D: {len(ops_D)} screws '
              f'— asynchronous execution, {len(tpg.edges)} rendezvous.', flush=True)
        ok = self._execute_tpg(tpg, dry=(self.tpg_execution == 'dry'))
        if not ok:
            print('[unscrew_pipeline_tpg] aborted.', flush=True)
        return ok

    def unscrew_both(self,
                     part_name_C: str, path_C: list, direction_C: np.array,
                     part_name_D: str, path_D: list, direction_D: np.array):
        """Unscrew two independent screws simultaneously — one arm each.

        Both robots approach, engage, and pull out their respective screws at the
        same time.  The G-code steps for robot_C and robot_D are interleaved by
        _interleave_gcode() so PyBullet advances both arms each physics tick,
        giving a physically accurate dual-arm simulation.

        Phases
        ------
        1. Compute tool paths for both screws.
        2. Move both robots to their respective approach positions using
           _execute_move_both() — planned transit (cuMotion/GNN) that treats the
           other arm as an obstacle.
        3-7. Each arm then runs its own stream — engage → attach → pull out →
           drop → release → retreat — interleaved tick by tick by
           _interleave_streams() with live yield arbitration: when the arms come
           within yield_margin_m the arm without priority holds until the gap
           reopens.  Priority defaults to robot_C and passes to the other arm
           the moment this one has its screw out, so one arm unscrews while the
           other is away dropping its screw.  Real contact before any screw is
           attached falls back to fully serial execution (Option A).
        """
        # ── Phase 1: build tool paths ─────────────────────────────────────────
        if part_name_C not in self.assembly.part_ids:
            print(f'[unscrew_both] ERROR: "{part_name_C}" not in assembly.', flush=True)
            return
        if part_name_D not in self.assembly.part_ids:
            print(f'[unscrew_both] ERROR: "{part_name_D}" not in assembly.', flush=True)
            return

        tp_C = self._screw_tool_path(part_name_C, path_C, direction_C, self.screwdriver_C)
        tp_D = self._screw_tool_path(part_name_D, path_D, direction_D, self.screwdriver)
        tp_C.draw(False, [0, 0, 1])   # blue path for robot_C
        tp_D.draw(False, [1, 0, 0])   # red path  for robot_D

        # ── Phase 2: move both robots to their approach positions ─────────────
        # _execute_move_both() uses GNN planning when available so approach paths
        # are collision-free between the two arms.
        self._execute_move_both(
            target_pos_C=tp_C.positions[:, 0],
            target_orn_q_C=tp_C.orientations[:, 0],
            target_pos_D=tp_D.positions[:, 0],
            target_orn_q_D=tp_D.orientations[:, 0],
            feed_rate_C=10000.0,
            feed_rate_D=10000.0,
            tool_targets=True,
        )

        # ── Phase 2b: reorient robot_C to match the planned approach orientation ─
        # The GNN transit move reaches the correct position but may leave robot_C
        # with a wrong EEF orientation (trained in MuJoCo, orientation convention
        # differs slightly).  A short G-code move at the current XYZ snaps the
        # tool to the expected orientation before engagement begins.
        _pos_C_now, _ = self.robot_C.get_endeffector_pose()
        _orn_C_target = p.getEulerFromQuaternion(tp_C.orientations[:, 0])
        gc_reorient_C = [
            {'M': 0}, {'T': 12},
            {'G': 1,
             'X': _pos_C_now[0], 'Y': _pos_C_now[1], 'Z': _pos_C_now[2],
             'A': _orn_C_target[0], 'B': _orn_C_target[1], 'C': _orn_C_target[2],
             'F': 1000.0},
            {'M': 0}, {'T': 11},
        ]
        self.g_code_processor_C.g_code = gc_reorient_C
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()

        # ── Phases 3-7 as two independent per-arm streams, arbitrated live ─────
        stream_C = self._screw_segments('C', part_name_C, tp_C, path_C)
        stream_D = self._screw_segments('D', part_name_D, tp_D, path_D)
        # G-code is exported as executed by _interleave_streams (incl. sync markers).
        ok, _ = self._interleave_streams(stream_C, stream_D, priority_default='C')
        if not ok:
            print(
                f'[unscrew_both] Contact during engagement of '
                f'"{part_name_C}" / "{part_name_D}" despite arbitration. '
                f'Retreating both arms and serialising (Option A fallback).',
                flush=True,
            )
            # Retreat both arms to their standoff positions (path[0]) and deactivate motors.
            pos0_C = tp_C.positions[:, 0]
            pos0_D = tp_D.positions[:, 0]
            orn0_C = p.getEulerFromQuaternion(tp_C.orientations[:, 0])
            orn0_D = p.getEulerFromQuaternion(tp_D.orientations[:, 0])
            gc_retreat_C = [
                {'M': 0}, {'T': 11},
                {'G': 1, 'X': pos0_C[0], 'Y': pos0_C[1], 'Z': pos0_C[2],
                 'A': orn0_C[0], 'B': orn0_C[1], 'C': orn0_C[2], 'F': 5000.0},
            ]
            gc_retreat_D = [
                {'M': 0}, {'T': 11},
                {'G': 1, 'X': pos0_D[0], 'Y': pos0_D[1], 'Z': pos0_D[2],
                 'A': orn0_D[0], 'B': orn0_D[1], 'C': orn0_D[2], 'F': 5000.0},
            ]
            # Run retreat without collision check (arms moving apart — safe).
            self._interleave_gcode(gc_retreat_C, gc_retreat_D, check_robot_collision=False)
            # Now execute the two screws sequentially: D first, then C.
            # Each call handles its own transit, engagement, pullout, and drop.
            print(f'[unscrew_both] Sequential fallback: unscrew {part_name_D} (robot_D)…', flush=True)
            self.unscrew(part_name_D, path_D, direction_D)
            print(f'[unscrew_both] Sequential fallback: unscrew {part_name_C} (robot_C)…', flush=True)
            self.unscrew_C(part_name_C, path_C, direction_C)
            # unscrew / unscrew_C handle everything internally — return here.
            return

    # ------------------------------------------------------------------
    # Cooperative rigid carry: two arms, ghost grippers, the carrier part of the
    # loaded assembly (dual grip benchmark declares it in its XML root)
    # ------------------------------------------------------------------
    COOP_DEFAULTS = dict(ghost_len=0.10, approach_back=0.15, path_step=0.02, exec_stride=2)
    _coop = None

    def _coop_setup(self):
        """Cooperative task from the LOADED assembly (XML root attributes cooperative_*)."""
        asm = self.assembly
        spec = getattr(asm, 'cooperative', None) if asm is not None else None
        if not spec:
            raise ValueError('the loaded assembly declares no cooperative task (load the dual grip benchmark)')
        if self._coop is not None and self._coop.get('assembly') is asm:
            return self._coop
        carrier = spec['carrier']
        if carrier not in asm.part_ids:
            raise ValueError(f'carrier part "{carrier}" is not in the loaded assembly')
        part = next(pt for pt in asm.parts if pt.name == carrier)
        bounds = np.asarray(part.mesh.bounds, dtype=float) * asm.scaleFactor
        dims = bounds[1] - bounds[0]
        cfg = {**self.COOP_DEFAULTS, 'shift': tuple(spec['shift']), 'lift': float(spec['lift']),
               'grip_half': dims[1] / 2 + self.COOP_DEFAULTS['ghost_len']}
        self._coop = dict(assembly=asm, cfg=cfg, carrier=carrier, payload=asm.part_ids[carrier],
                          centre_local=(bounds[0] + bounds[1]) / 2, dims=dims, ghosts={}, grasp=None)
        print(f'[coop] carrier "{carrier}" {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f} m, shift {cfg["shift"]}, '
              f'lift {cfg["lift"]}', flush=True)
        return self._coop

    def _coop_payload_T(self) -> np.ndarray:
        """World pose of the carrier's GEOMETRIC CENTRE (its body origin is the assembly origin)."""
        pos, q = p.getBasePositionAndOrientation(self._coop['payload'])
        T_body = self._pose_to_4x4(pos, q)
        T_c = np.eye(4)
        T_c[:3, 3] = self._coop['centre_local']
        return T_body @ T_c

    def _coop_obstacles(self):
        """Only the desk block: the platform legs and the carrier are deliberately not obstacles
        (the flange sits above the leg tops and the ghost grippers have no collision)."""
        from modular_planner import ObstacleBox
        c = self.clamping_desk_centre
        top = self.clamping_desk_top_z
        return [ObstacleBox(center=[c[0], c[1], top / 2], half_extents=[0.6, 0.6, top / 2], rotation=np.eye(3))]

    def _coop_make_ghosts(self):
        """One visual-only 'gripper' per flange (no collision, no tool change), and the
        arms never collide with the carrier (it is held by constraints)."""
        g = self._coop['cfg']['ghost_len']
        for side in ('C', 'D'):
            if side in self._coop['ghosts']:
                continue
            robot = self._arm(side)[0]
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, g / 2], rgbaColor=[0.1, 0.8, 0.9, 0.7])
            gid = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                    basePosition=[0, 0, 0])
            cid = p.createConstraint(robot.urdf, robot._default_endeffector_id, gid, -1, p.JOINT_FIXED,
                                     [0, 0, 0], [0, 0, g / 2], [0, 0, 0])
            self._coop['ghosts'][side] = (gid, cid)
            for link in range(-1, p.getNumJoints(robot.urdf)):
                p.setCollisionFilterPair(self._coop['payload'], robot.urdf, -1, link, 0)

    def _coop_flange_T(self, side) -> np.ndarray:
        robot = self._arm(side)[0]
        pos, q = robot.get_endeffector_pose()
        return self._pose_to_4x4(pos, q)

    def _coop_find_grasp(self):
        """First flange orientation (z axis toward the payload from arm C's side, world +y)
        for which BOTH arms reach their grip pose and the lifted/shifted target with all limits."""
        cfg = self._coop['cfg']
        planner = self._active_planner
        pos0 = self._coop_payload_T()[:3, 3]
        delta = np.array(cfg['shift'], dtype=float) + [0, 0, cfg['lift']]
        bases = {}
        for side in ('C', 'D'):
            bp, bq = p.getBasePositionAndOrientation(self._arm(side)[0].urdf)
            bases[side] = self._pose_to_4x4(bp, bq)
        obstacles = self._coop_obstacles()
        cands = []
        for el in range(-90, 91, 15):
            for az in range(0, 360, 15):
                if abs(el) == 90 and az > 0:
                    continue
                e, a = math.radians(el), math.radians(az)
                z = [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
                if -z[1] >= 0.5:
                    cands.append(z)
        cands.sort(key=lambda z: (-z[1], -z[2]))
        for z in cands:
            for roll in np.radians(np.arange(0, 360, 45)):
                ga = coop_grasp_frame(pos0 + [0, cfg['grip_half'], 0], z, roll)
                gb = coop_grasp_frame(pos0 - [0, cfg['grip_half'], 0], [-z[0], -z[1], z[2]], roll)
                ta, tb = ga.copy(), gb.copy()
                ta[:3, 3] += delta
                tb[:3, 3] += delta
                qs = [planner.ik(T, robot_name=f'robot_{sd}', robot_base_pose=bases[sd], limit_margin=0.05,
                                 obstacles=obstacles)
                      for T, sd in ((ga, 'C'), (ta, 'C'), (gb, 'D'), (tb, 'D'))]
                if all(q is not None for q in qs):
                    self._coop['grasp'] = dict(z=z, roll=float(roll), ga=ga, gb=gb, qs=qs, bases=bases)
                    print(f'[coop] grasp found: flange z-axis {np.round(z, 2)}, roll {math.degrees(roll):.0f} deg',
                          flush=True)
                    return self._coop['grasp']
        return None

    def _coop_ik_line(self, side, T_from, T_to, n):
        """Joint configurations along a straight Cartesian line (IK seeded from the previous
        step, all limits and the desk checked) or None."""
        from scipy.spatial.transform import Rotation, Slerp
        planner = self._active_planner
        base = self._coop['grasp']['bases'][side]
        slerp = Slerp([0, 1], Rotation.from_matrix([T_from[:3, :3], T_to[:3, :3]]))
        seed = self._planner_start_conf(side)
        obstacles = self._coop_obstacles()
        out = []
        for k in range(1, n + 1):
            t = k / n
            T = np.eye(4)
            T[:3, :3] = slerp([t]).as_matrix()[0]
            T[:3, 3] = (1 - t) * T_from[:3, 3] + t * T_to[:3, 3]
            q = planner.ik(T, robot_name=f'robot_{side}', seed=seed, robot_base_pose=base, obstacles=obstacles)
            if q is None:
                return None
            out.append(q)
            seed = q
        return out

    def _coop_run_lockstep(self, lines_C, lines_D, feed=3000.0, kinematic=True, rel=None):
        """RA-line lists for both arms, executed waypoint by waypoint in lockstep and appended
        to the exported programs.  kinematic: set the joints directly (no physics, seconds
        instead of tens of minutes); rel = (rel_C, rel_D) makes the gripped carrier follow."""
        for qc, qd in zip(lines_C, lines_D):
            lc = {'G': 1, **{f'RA{i + 1}': float(v) for i, v in enumerate(qc)}, 'F': feed}
            ld = {'G': 1, **{f'RA{i + 1}': float(v) for i, v in enumerate(qd)}, 'F': feed}
            if kinematic:
                self._dry_set_conf('C', qc)
                self._dry_set_conf('D', qd)
                self._coop_follow(rel)
            else:
                self._interleave_gcode([lc], [ld])
            self.g_code_C.append(lc)
            self.g_code_D.append(ld)

    def _coop_follow(self, rel=None):
        """Kinematic mode: move the ghost grippers with the flanges and (when gripped) the carrier
        with arm C; the pose arm D implies is compared to it (rigidity check)."""
        from scipy.spatial.transform import Rotation
        g = self._coop['cfg']['ghost_len']
        for side, (gid, _) in self._coop['ghosts'].items():
            T = self._coop_flange_T(side)
            T[:3, 3] = T[:3, 3] + T[:3, 2] * g / 2
            p.resetBasePositionAndOrientation(gid, T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat())
        if rel is None:
            return
        Tc = self._coop_flange_T('C') @ rel[0]
        p.resetBasePositionAndOrientation(self._coop['payload'], Tc[:3, 3], Rotation.from_matrix(Tc[:3, :3]).as_quat())
        Td = self._coop_flange_T('D') @ rel[1]
        dev = float(np.linalg.norm(Td[:3, 3] - Tc[:3, 3]))
        self._coop['max_dev'] = max(self._coop.get('max_dev', 0.0), dev)

    def _coop_joint_move_both(self, T_C, T_D, kinematic=True):
        """Collision-aware joint move of both arms to two flange poses (cuMotion plan_multi)."""
        from scipy.spatial.transform import Rotation
        planner = self._active_planner
        quat = lambda T: Rotation.from_matrix(T[:3, :3]).as_quat()
        if not kinematic:
            self._execute_move_both(T_C[:3, 3], quat(T_C), T_D[:3, 3], quat(T_D), tool_targets=False)
            return
        self._unwind_continuous_joints('C')      # wrist turns beyond +-pi are invalid start states
        self._unwind_continuous_joints('D')
        bases = self._coop['grasp']['bases']
        planner.plan_multi(
            current_eef_poses=[self._coop_flange_T('C'), self._coop_flange_T('D')],
            target_eef_poses=[T_C, T_D], obstacles=self._coop_obstacles(), constraints=[],
            robot_names=['robot_C', 'robot_D'],
            start_confs=[self._planner_start_conf('C'), self._planner_start_conf('D')],
            robot_aabb_getters=[lambda: self._arm_obstacle_aabbs('C'), lambda: self._arm_obstacle_aabbs('D')],
            robot_base_poses=[bases['C'], bases['D']])
        jps = getattr(planner, 'last_joint_paths_multi', None)
        if not jps or not len(jps[0]) or not len(jps[1]):
            raise RuntimeError('cuMotion returned no joint path for the approach')
        n = max(len(jps[0]), len(jps[1]))
        pick = lambda jp: [jp[int(round(i))] for i in np.linspace(0, len(jp) - 1, n)]
        self._coop_run_lockstep(pick(jps[0]), pick(jps[1]), kinematic=True)

    def _coop_sync(self, why: str):
        """Same rendezvous marker in BOTH exported programs (WAITM(n,1,2) + a universal
        '; ==== SYNC k/K' comment): each channel waits here for the other."""
        self._sync_no = getattr(self, '_sync_no', 0) + 1
        n = 1 + (self._sync_no - 1) % self.sync_marker_count
        for gc in (self.g_code_C, self.g_code_D):
            gc.append({'WAITM': n, 'why': why, 'seq': self._sync_no, 'total': self._sync_total})
        print(f'[sync] WAITM({n}) SYNC {self._sync_no}/{self._sync_total}: {why}', flush=True)

    def _coop_start_header(self):
        """';START POSE' header (arms placed there by hand before starting), same format as the
        asynchronous export; the flange pose because no tool is coupled."""
        from scipy.spatial.transform import Rotation as R
        self._start_pose_header = {}
        for s_ in ('C', 'D'):
            T = self._coop_flange_T(s_)
            e = R.from_matrix(T[:3, :3]).as_euler('XYZ')
            q = self._planner_start_conf(s_)
            self._start_pose_header[s_] = (
                f';START POSE robot_{s_} (place the arm here before starting):\n'
                f';  flange X{T[0, 3] * 1000:.1f} Y{T[1, 3] * 1000:.1f} Z{T[2, 3] * 1000:.1f} '
                f'A{math.degrees(e[0]):.1f} B{math.degrees(e[1]):.1f} C{math.degrees(e[2]):.1f}\n'
                f';  axes  ' + ' '.join(
                    f'RA{i + 1}={math.degrees(v - (q[1] if (i == 2 and COMAU_AXIS3_ABSOLUTE) else 0.0)):.1f}'
                    for i, v in enumerate(q)) + '  (RA3 absolute, Comau convention)\n')

    def _coop_refix_carrier(self):
        """Put the carrier back under the assembly's fixed-constraint scheme at its NEW pose
        (parts do not collide with each other, so it would otherwise fall through the legs)."""
        asm, carrier, pid = self.assembly, self._coop['carrier'], self._coop['payload']
        bp, bq = p.getBasePositionAndOrientation(asm.part_ids['base'])
        pos, orn = p.getBasePositionAndOrientation(pid)
        rel_p, rel_q = p.multiplyTransforms(*p.invertTransform(bp, bq), pos, orn)
        inv_p, inv_q = p.invertTransform(rel_p, rel_q)
        asm.constraint_ids[carrier] = p.createConstraint(asm.part_ids['base'], -1, pid, -1, p.JOINT_FIXED,
                                                         [0, 0, 0], [0, 0, 0], inv_p, [0, 0, 0, 1], inv_q)
        asm.part_offset_vectors[carrier], asm.part_offset_orientations[carrier] = rel_p, rel_q

    def cooperative_carry(self, exec_stride=None, feed=3000.0, ghost_gripper_C=True, physics=False):
        """Both arms grip the carrier of the loaded assembly at its two long edges (ghost
        grippers, constraints), lift it, carry it to the target platform, set it down and
        retreat.  The carry is planned as ONE object path (plan_rigid_carry) so the two grips
        stay rigid.  Returns a JSON-friendly result dict."""
        from modular_planner import PlannerType
        from scipy.spatial.transform import Rotation
        t0 = time.time()
        planner = self._active_planner
        if not ghost_gripper_C:
            return {'ok': False, 'error': 'select "gripper" as the start tool of both robots (the cooperative task needs two grippers)'}
        if self._planner_type != PlannerType.CUMOTION or not hasattr(planner, 'plan_rigid_carry'):
            return {'ok': False, 'error': 'cooperative carry needs the cuMotion planner (start the sim with --planner cumotion)'}
        for tool in (self.screwdriver_C, self.screwdriver, self.SRG, self.DLG, self.millingtool):
            if tool.is_coupled():
                return {'ok': False, 'error': 'a tool is coupled to an arm: store it first (the demo uses ghost grippers on bare flanges)'}
        self._coop_mode = True
        try:
            self._coop_setup()
            self._coop_make_ghosts()
            cfg = self._coop['cfg']
            self._sync_no, self._sync_total = 0, 4
            self._unwind_continuous_joints('C')
            self._unwind_continuous_joints('D')
            self._coop_start_header()
            grasp = self._coop['grasp'] or self._coop_find_grasp()
            if grasp is None:
                return {'ok': False, 'error': 'no orientation reaches the grip poses and the target with both arms'}
            ga, gb, qs, bases = grasp['ga'], grasp['gb'], grasp['qs'], grasp['bases']
            pid, carrier = self._coop['payload'], self._coop['carrier']
            O0 = self._coop_payload_T()
            keys = [O0]
            for d in ([0, 0, cfg['lift']], list(cfg['shift']), [0, 0, -cfg['lift']]):
                nxt = keys[-1].copy()
                nxt[:3, 3] += d
                keys.append(nxt)
            obj_path = planner.densify_poses(keys, step_m=cfg['path_step'])
            plan = planner.plan_rigid_carry(
                obj_path, [np.linalg.inv(O0) @ ga, np.linalg.inv(O0) @ gb], obstacles=self._coop_obstacles(),
                robot_names=['robot_C', 'robot_D'], robot_base_poses=[bases['C'], bases['D']],
                start_confs=[qs[0], qs[2]], payload_half_extents=list(self._coop['dims'] / 2))
            if not plan['ok']:
                return {'ok': False, 'error': 'carry planning failed: ' + str(plan['fail'])}
            print(f'[coop] carry planned: {len(obj_path)} steps, T_rel error {plan["trel_err_mm"]:.2f} mm, '
                  f'min arm gap {plan["min_arm_gap_m"] * 1000:.0f} mm', flush=True)

            # ---- approach: joint move to 15 cm before the grips, then straight in
            back = cfg['approach_back']
            pre = []
            for T in (ga, gb):
                P = T.copy()
                P[:3, 3] -= T[:3, 2] * back
                pre.append(P)
            kin = not physics
            print(f'[coop] phase 1/5: joint move to the pre-grip poses ({"kinematic" if kin else "physics"})', flush=True)
            self._coop_joint_move_both(pre[0], pre[1], kinematic=kin)
            self._coop_sync('both arms at the pre-grip poses')
            n_in = 8
            in_C = self._coop_ik_line('C', self._coop_flange_T('C'), ga, n_in)
            in_D = self._coop_ik_line('D', self._coop_flange_T('D'), gb, n_in)
            if in_C is None or in_D is None:
                return {'ok': False, 'error': 'straight approach to the grips has no IK solution'}
            print('[coop] phase 2/5: straight approach to the grips', flush=True)
            self._coop_run_lockstep(in_C, in_D, feed, kinematic=kin)

            # ---- grip: release the carrier from the assembly, fix it to both flanges
            p.removeConstraint(self.assembly.constraint_ids[carrier])
            self.assembly.constraint_ids[carrier] = None
            cons, rel = [], None
            pp, pq = p.getBasePositionAndOrientation(pid)
            if kin:
                T_body = self._pose_to_4x4(pp, pq)
                rel = [np.linalg.inv(self._coop_flange_T(sd)) @ T_body for sd in ('C', 'D')]
                self._coop['max_dev'] = 0.0
            else:
                for side in ('C', 'D'):
                    rid, link = self._arm(side)[0].urdf, self._arm(side)[0]._default_endeffector_id
                    lp, lq = p.getLinkState(rid, link, computeForwardKinematics=True)[:2]
                    rel_p, rel_q = p.multiplyTransforms(*p.invertTransform(lp, lq), pp, pq)
                    cons.append(p.createConstraint(rid, link, pid, -1, p.JOINT_FIXED, [0, 0, 0],
                                                   rel_p, [0, 0, 0], rel_q, [0, 0, 0, 1]))
                for _ in range(30):
                    self.stepSimulation()

            self._coop_sync('both grippers closed on the carrier')
            print('[coop] phase 3/5: gripped, carrying', flush=True)
            # ---- carry (every exec_stride-th step; the full path is what was verified)
            stride = exec_stride or cfg['exec_stride']
            n = len(plan['joint_paths'][0])
            idx = list(range(0, n, stride))
            if idx[-1] != n - 1:
                idx.append(n - 1)
            self._coop_run_lockstep([plan['joint_paths'][0][k] for k in idx],
                                    [plan['joint_paths'][1][k] for k in idx], feed, kinematic=kin, rel=rel)

            self._coop_sync('carrier on the target platform, before release')
            print('[coop] phase 4/5: release and re-fix on the target platform', flush=True)
            # ---- release, re-fix the carrier on the target platform, retreat
            for c in cons:
                p.removeConstraint(c)
            if not kin:
                for _ in range(10):
                    self.stepSimulation()
            self._coop_refix_carrier()
            self._coop_sync('both grippers open')
            back_T = np.eye(4)
            back_T[2, 3] = -back
            out_C = self._coop_ik_line('C', self._coop_flange_T('C'), self._coop_flange_T('C') @ back_T, n_in)
            out_D = self._coop_ik_line('D', self._coop_flange_T('D'), self._coop_flange_T('D') @ back_T, n_in)
            if out_C is not None and out_D is not None:
                self._coop_run_lockstep(out_C, out_D, feed, kinematic=kin)
            print('[coop] phase 5/5: retreated, measuring', flush=True)
            self._last_export_kind = 'coop'
            err = float(np.linalg.norm(self._coop_payload_T()[:3, 3] - keys[0][:3, 3] - np.array(cfg['shift'])) * 1000)
            return {'ok': True, 'steps_planned': len(obj_path), 'steps_executed': len(idx),
                    'trel_err_mm': float(plan['trel_err_mm']), 'min_arm_gap_mm': float(plan['min_arm_gap_m'] * 1000),
                    'payload_final_err_mm': err, 'grip_dev_mm': float(self._coop.get('max_dev', 0.0) * 1000),
                    'mode': 'physics' if physics else 'kinematic',
                    'waypoints_C': len(self.g_code_C), 'waypoints_D': len(self.g_code_D),
                    'seconds': round(time.time() - t0, 1)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
        finally:
            self._coop_mode = False

    def start_server(self, host='127.0.0.1', port=12345):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"Server listening on {host}:{port}")
            while True:  # keep serving after a client disconnects
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    sim_server_protocol.serve_connection(conn, self._dispatch_request)

    def _dispatch_request(self, data_received):
        """Run one request and return the response dict. Wrapped by sim_server_protocol."""
        if data_received["request"] == "get_state":
            data_to_send = {"state": "running"}
        elif data_received["request"] == "update_robots":
            self.update_real_robot_positions()
            data_to_send = {"state": "robots updated"}
        elif data_received["request"] == "capture_image_of_cell":
            rgba_array = self.capture_image_of_cell()
            # Convert RGBA to a compressed PNG
            pil_image = Image.fromarray(rgba_array, 'RGBA')
            buffer = io.BytesIO()
            pil_image.save(buffer, format="png")
            #image_bytes = buffer.getvalue()
            data_to_send = {"state": "image captured", "buffer": buffer}
        elif data_received["request"] == "load_assembly": # hier wird assembly in SImulation reingeladen.
            # The request carries the mounting mode so loading never depends
            # on a separate set_mounting_mode having reached this process first.
            if "mounting_mode" in data_received:
                self.set_mounting_mode(data_received["mounting_mode"])
            _file = _win_to_linux_path(data_received["file"])
            _key = (_file, self._mounting_mode)
            if self._assembly_loaded == _key:
                print("[load_assembly] same assembly and mounting mode already loaded — "
                      "skipping duplicate request.", flush=True)
                data_to_send = {"state": f"assembly already loaded ({self._mounting_mode})"}
            else:
                self.loadAssembly(_file, data_received["use_loading_point"])
                self._assembly_loaded = _key
                data_to_send = {"state": f"assembly loaded ({self._mounting_mode})"}
        elif data_received["request"] == "load_assembly_op": # Neue Pose
            self.loadAssemblyop(_win_to_linux_path(data_received["file"]))
            data_to_send = {"state": "assembly loaded"}
        elif data_received["request"] == "get_zivid_image":
            self.update_real_robot_positions()
            self.zivid.get_image()
            start = time.time()
            print(f"look_at_part took {time.time()-start} seconds")
            inverted_dict = {value: key for key, value in self.assembly.part_ids.items() if
                             key != "base"}
            #resulting_list = [inverted_dict.get(num, str(num)) for num in self.zivid.last_frame["index_array"]]
            frame = self.zivid.last_frame
            #frame["part_array"]=resulting_list
            frame["inverted_dict"] = inverted_dict
            frame["fixture_id"]= self.fixture
            frame["assembly_base_pos"] = self.assembly.spawn_point
            frame["assembly_base_orn"] = self.assembly.spawn_orient
            data_to_send = {"state": "image captured", "frame": frame}
            print(f"data preparation took {time.time()-start} seconds")
        elif data_received["request"] == "look_at_part":
            start = time.time()
            self.g_code_processor_D.active_endeffector = len(self.g_code_processor_D.endeffector_list) - 1
            print(f" active eef list index: {self.g_code_processor_D.active_endeffector}")
            #self.safe_C() # muss vll auskommentiert sein, da Roboter mit Baugruppe dort bleiben soll. anders als in Fofa
            self.look_at_part(data_received["part_name"],
                              data_received["distance"],
                              data_received["theta"],
                              data_received["phi"])
            print(f"look_at_part took {time.time()-start} seconds")
            inverted_dict = {value: key for key, value in self.assembly.part_ids.items() if
                             key != "base"}
            #resulting_list = [inverted_dict.get(num, str(num)) for num in self.zivid.last_frame["index_array"]]
            frame = self.zivid.last_frame
            #frame["part_array"]=resulting_list
            frame["inverted_dict"] = inverted_dict
            frame["assembly_base_pos"] = self.assembly.spawn_point
            frame["assembly_base_orn"] = self.assembly.spawn_orient
            data_to_send = {"state": "image captured", "frame": frame}

            self.g_code_processor_D.active_endeffector = -1
            self.g_code_processor_D.active_endeffector = self.g_code_processor_D.get_active_endeffector()
            if self.g_code_processor_D.active_endeffector == len(self.g_code_processor_D.endeffector_list)-1:
                self.g_code_processor_D.active_endeffector = -1
            print(f"data preparation took {time.time()-start} seconds")
        elif data_received["request"] == "look_at_assembly":
            start = time.time()
            self.g_code_processor_D.active_endeffector = len(self.g_code_processor_D.endeffector_list) - 1
            #self.safe_D()
            temp_center_of_mass = np.array([0.0, 0.0, 0.0])
            for part in self.assembly.parts:
                temp_center_of_mass += part.center_of_mass
            temp_center_of_mass /= len(self.assembly.parts)
            self.look_at_point(temp_center_of_mass*self.assembly.scaleFactor, data_received["distance"],
                               data_received["theta"],
                               data_received["phi"],
                               False)
            print(f"look_at_part took {time.time() - start} seconds")
            inverted_dict = {value: key for key, value in self.assembly.part_ids.items()}
            # resulting_list = [inverted_dict.get(num, str(num)) for num in self.zivid.last_frame["index_array"]]
            frame = self.zivid.last_frame
            # frame["part_array"]=resulting_list
            frame["inverted_dict"] = inverted_dict
            frame["assembly_base_pos"] = self.assembly.spawn_point
            frame["assembly_base_orn"] = self.assembly.spawn_orient
            data_to_send = {"state": "image captured", "frame": frame}
            self.g_code_processor_D.active_endeffector = -1
            self.g_code_processor_D.active_endeffector = self.g_code_processor_D.get_active_endeffector()
            if self.g_code_processor_D.active_endeffector == len(self.g_code_processor_D.endeffector_list)-1:
                self.g_code_processor_D.active_endeffector = -1
            print(f"data preparation took {time.time() - start} seconds")
        elif data_received["request"] == "execute_action":
            active_eef_list_index = self.g_code_processor_D.get_active_endeffector()
            print(f"active eef list index: {active_eef_list_index}")
            if active_eef_list_index == len(self.g_code_processor_D.endeffector_list)-1:# without regarding the zivid camera
                active_eef_list_index = -1
            #self.safe_C() # muss vll auskommentiert sein, da Roboter mit Baugruppe dort bleiben soll. anders als in Fofa
            if data_received["type"] == "gripping":
                if active_eef_list_index <0:
                    self.unstore_SRG(data_received["part_name"])
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.SRG:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.midway_point_to_storage()
                        self.store_screwdriver()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.midway_point()
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.midway_point_to_storage()
                        self.store_millingtool()
                    #time.sleep(0.1)
                    #self.g_code_processor_D.active_endeffector = -1
                    #time.sleep(0.1)
                    self.unstore_SRG(data_received["part_name"])
                    #self.midway_point()
                self.grip(data_received["part_name"], data_received["path"], data_received["direction"])
                data_to_send = {"state": "gripping action simulated"}
            elif data_received["type"] == "cable_gripping":
                if active_eef_list_index<0:
                    self.unstore_SRG()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.SRG:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.store_screwdriver()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.store_millingtool()
                    self.unstore_SRG()
                self.grip_cable(data_received["part_name"], data_received["path"], data_received["orientations"])
                data_to_send = {"state": "cable_gripping action simulated"}
            elif data_received["type"] == "cable_gripping_MV":
                if active_eef_list_index<0:
                    self.unstore_SRG()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.SRG:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.store_screwdriver()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.store_millingtool()
                    self.unstore_SRG()
                self.grip_cable_MV(data_received["path"], data_received["orientations"])
                data_to_send = {"state": "cable_gripping_MV action simulated"}
            elif data_received["type"] == "SafetyRing":
                if active_eef_list_index<0:
                    self.unstore_SRG()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.SRG:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.store_screwdriver()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.store_millingtool()
                    self.unstore_SRG()
                self.general_grip(data_received["part_name"], data_received["path"], data_received["orientations"],
                                  data_received["grip_widths"])
                data_to_send = {"state": "gripping action simulated"}
            elif data_received["type"] == "screwing":
                with self._not_exported():   # tool fetch is manual in the cell
                  if active_eef_list_index <0:
                    self.unstore_screwdriver()
                  elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.screwdriver:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.SRG:
                        #self.midway_point()
                        self.midway_point_to_storage()
                        self.store_SRG()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.midway_point()
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.midway_point_to_storage()
                        self.store_millingtool()
                    self.g_code_processor_D.active_endeffector = -1
                    self.unstore_screwdriver()
                    self.midway_point()
                self.unscrew(data_received["part_name"], data_received["path"], data_received["direction"])
                data_to_send = {"state": "screwing action simulated"}
            elif data_received["type"] == "screwing_C":
                # Single screwing task assigned to robot_C by the CP-SAT scheduler.
                # Equips screwdriver_C if not already active, then calls unscrew_C().
                active_eef_C = self.g_code_processor_C.active_endeffector
                if active_eef_C < 0 or self.g_code_processor_C.endeffector_list[active_eef_C] != self.screwdriver_C:
                    with self._not_exported():   # tool fetch is manual in the cell
                        self.unstore_screwdriver_C()
                self.unscrew_C(data_received["part_name"], data_received["path"], data_received["direction"])
                data_to_send = {"state": "screwing_C action simulated"}
            elif data_received["type"] == "set_execution_mode":
                self.execution_mode = data_received.get("mode", "reactive")
                if data_received.get("rule") in ('segment', 'zone'):
                    self.tpg_rule = data_received["rule"]
                print(f'[sim] execution mode: {self.execution_mode}', flush=True)
                data_to_send = {"state": f"execution mode {self.execution_mode}"}
            elif data_received["type"] == "dual_screwing_batch":
                # Whole schedule at once: each arm runs its list of screws as
                # one continuous chain (transit -> unscrew -> drop, repeat) with
                # no per-pair barrier.  See unscrew_pipeline.
                self._equip_both_screwdrivers(active_eef_list_index)
                import time as _time
                _t0 = _time.perf_counter()
                _mode = data_received.get("execution_mode", self.execution_mode)
                if "physics" in data_received:      # HMI "Run without physics" box
                    self.tpg_execution = 'physics' if data_received["physics"] else 'dry'
                    print(f'[sim] TPG execution pass: {self.tpg_execution}', flush=True)
                _run = self.unscrew_pipeline_tpg if _mode == 'tpg' else self.unscrew_pipeline
                _ok = _run(data_received["ops_C"], data_received["ops_D"])
                wall_time = _time.perf_counter() - _t0
                data_to_send = {
                    "state": "dual screwing batch simulated" if _ok
                             else "dual screwing batch ABORTED on contact",
                    "ok": _ok,
                    "wall_time": wall_time,
                    "n_C": len(data_received["ops_C"]),
                    "n_D": len(data_received["ops_D"]),
                }
            elif data_received["type"] == "dual_screwing":
                # Parallel screwing: two independent screws assigned one per arm.
                #
                # Physical cell: robot_C and robot_D each carry a screwdriver
                # (screwdriver_C and screwdriver, respectively) and work
                # simultaneously on separate screws.  Both are loaded in the
                # simulation; _interleave_gcode() inside unscrew_both() advances
                # both G-code processors one physics tick at a time so PyBullet
                # sees both arms moving in the same simulation steps.
                #
                # Auto-equip: if either robot does not already hold its
                # screwdriver at dispatch time, the dispatcher swaps tools
                # automatically before the motion begins — no manual HMI step.
                #
                # Fields expected in data_received:
                #   part_name_C / path_C / direction_C  → task assigned to robot_C
                #   part_name_D / path_D / direction_D  → task assigned to robot_D
                self._equip_both_screwdrivers(active_eef_list_index)

                import time as _time
                _t0 = _time.perf_counter()

                # Both arms execute their unscrewing motions simultaneously.
                # _interleave_gcode() inside unscrew_both() advances both
                # G-code processors one step at a time, so PyBullet sees
                # both robots moving during the same physics ticks.
                self.unscrew_both(
                    data_received["part_name_C"],
                    data_received["path_C"],
                    data_received["direction_C"],
                    data_received["part_name_D"],
                    data_received["path_D"],
                    data_received["direction_D"],
                )

                wall_time = _time.perf_counter() - _t0
                data_to_send = {
                    "state": "dual screwing action simulated",
                    "wall_time": wall_time,
                    # Both arms ran in parallel so t_C ≈ t_D ≈ wall_time.
                    # Report equal durations; the HMI uses max() as the step cost.
                    "t_C": wall_time,
                    "t_D": wall_time,
                    "parallel_duration": wall_time,
                }
            elif data_received["type"] == "milling":
                if active_eef_list_index<0:
                    self.midway_point()
                    self.unstore_millingtool()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.millingtool:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.midway_point()
                        self.store_screwdriver()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.SRG:
                        self.midway_point()
                        self.store_SRG()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.midway_point()
                    #    self.store_BFG()
                    self.g_code_processor_D.active_endeffector = -1
                    self.unstore_millingtool()
                    self.midway_point()
                self.mill(data_received["part_name"], data_received["path"], data_received["direction"])
                data_to_send = {"state": "milling action simulated"}
            elif data_received["type"] == "general_gripping":
                if active_eef_list_index <0:
                    self.midway_point()
                    self.unstore_SRG()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.SRG:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.midway_point()
                        self.store_screwdriver()
                    #elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.BFG:
                    #    self.midway_point()
                    #    self.store_BFG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.millingtool:
                        self.midway_point()
                        self.store_millingtool()
                    self.g_code_processor_D.active_endeffector = -1
                    self.unstore_SRG()
                    self.midway_point()
                self.general_grip_SRG(data_received["part_name"], data_received["path"],
                                          data_received["orientations"],
                                          data_received["grip_widths"],
                                          data_received["position"],
                                          data_received["orientation"],)
                data_to_send = {"state": "gripping action simulated"}
            elif data_received["type"] == "move_assembly":
                if active_eef_list_index<0:
                    print("need to implement a new big gripper to move asm")
                    raise NotImplementedError("Moving assembly without an endeffector is not implemented")
                    # self.unstore_BFG()
                elif self.g_code_processor_D.endeffector_list[active_eef_list_index] != self.screwdriver:
                    if self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.SRG:
                        self.store_SRG()
                    elif self.g_code_processor_D.endeffector_list[active_eef_list_index] == self.screwdriver:
                        self.store_screwdriver()
                    #self.unstore_BFG()
                self.move_complete_assembly(data_received["pos"], p.getQuaternionFromEuler(data_received["orn"]))
                #self.store_BFG()
                data_to_send = {"state": "moving action simulated"}
            else:
                data_to_send = {"state": "unknown_action"}
            for _ in range(50):
                self.stepSimulation()
        elif data_received["request"] == "save_gcode":
          # The HMI runs on Windows and sends Windows paths; this process
          # runs in WSL, so translate exactly as load_assembly does.
          _file_C = _win_to_linux_path(data_received["file_C"])
          _file_D = _win_to_linux_path(data_received["file_D"])
          # EXPORT GATE: sweep every joint line of both programs (as the
          # controller interpolates them) against the table and the
          # assembly with the planner's spheres (arm + bars + tool).
          _gate = {s_: self._joint_clearance_report(s_) for s_ in ('C', 'D')}
          os.makedirs(os.path.dirname(_file_C), exist_ok=True)
          os.makedirs(os.path.dirname(_file_D), exist_ok=True)
          try:
            # INIT(2,…) + START(2) in robot_C's prefix launches channel 2
            # (robot_D) simultaneously.  Both programs share robot_C's base
            # frame as the coordinate origin — no per-robot transform needed.
            #
            # robot_C carries the screwdriver in the sim; in the cell it may
            # have none ("ghost": tool DATA selected, nothing mounted, so the
            # controller keeps the flange one tool-length off every target and
            # the arm moves exactly as if armed).  Either way the motions are
            # identical; only the physical tool interactions differ.
            _ghost_C = bool(data_received.get("ghost_tool_C", True))
            _d_name = os.path.basename(_file_D).upper()
            _coop_export = getattr(self, '_last_export_kind', None) == 'coop'
            _coop_note = (';COOPERATIVE CARRY with ghost grippers: no tool data is selected, '
                          'the arms move on axis words (RA1..RA6) only\n')
            _prefix_C = (
                f'R0=1\n'
                f'INIT(2,"{_d_name}")\n'
                f'START(2)\n'
                + (_coop_note if _coop_export else
                   (';GHOST TOOL: screwdriver tool data selected, none mounted\n'
                    if _ghost_C else '')
                   + f'T="{GHOST_SCREWDRIVER_TOOL_NAME if _ghost_C else SCREWDRIVER_TOOL_NAME}"\nM6\n')
                + getattr(self, '_start_pose_header', {}).get('C', '')
                + _gate['C']
            )
            if _ghost_C:
                # Every tool interaction becomes a comment; program stops and
                # the channel handshake (M0 / WAIT_FOR_GO) are kept for sync.
                _t_C = ghost_translation(T_TRANSLATION)
                _m_C = ghost_translation(M_TRANSLATION, keep={40})
            else:
                _t_C, _m_C = T_TRANSLATION, M_TRANSLATION
            write_g_code(transform_eulers_in_gcode(self.g_code_C), _file_C,
                         _prefix_C,
                         'T=0\nSTOPRE\nR0=0\nM30\n',
                         _m_C, _t_C)
            self.g_code_C = []
            # R0=1 … STOPRE R0=0 M30 is the OPC-UA handshake; keep it.
            write_g_code(transform_eulers_in_gcode(self.g_code_D), _file_D,
                         'R0=1\n' + (_coop_note if _coop_export else
                                     f'T="{SCREWDRIVER_TOOL_NAME}"\nM6\n')
                         + getattr(self, '_start_pose_header', {}).get('D', '')
                         + _gate['D'],
                         'STOPRE\nR0=0\nM30\n',
                         M_TRANSLATION, T_TRANSLATION)
            self.g_code_D = []
            self._last_export_kind = None
            data_to_send = {"state": f"gcode_saved -> {_file_C} , {_file_D}"}
          except Exception as _e:
            print(f"[save_gcode] FAILED: {_e!r}", flush=True)
            data_to_send = {"state": f"gcode save FAILED: {_e}"}
        elif data_received["request"] == "get_spawn_pose":
            data_to_send = {
                "spawn_point": self.assembly.spawn_point,
                "spawn_orient": self.assembly.spawn_orient
            }
        elif data_received["request"] == "get_station_anchors":
            # Robot bases expressed in the assembly's path frame (the
            # unscaled part coordinates the planner's paths use), so the
            # scheduler can give each arm the screws nearest to it.
            data_to_send = {"anchors": self._station_anchors_in_path_frame()}
        elif data_received["request"] == "allign":
            pos, orn = p.multiplyTransforms(data_received["pos"]*self.assembly.scaleFactor,
                                            data_received["orn"],
                                            self.assembly.spawn_point, self.assembly.spawn_orient)
            print(f"repositioning base to: {pos}, {p.getEulerFromQuaternion(orn)}")
            self.assembly.reposition_base(pos, orn)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"spawn_point.npy"), pos)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"spawn_orient.npy"), orn)
            for _ in range(100):
                self.stepSimulation()
            self.spawn_point = pos
            self.spawn_orient = orn
            data_to_send = {"state": "assembly repositioned"}
        elif data_received["request"] == "reposition_asm":
            #pos = data_received["pos_offset"]
            #orn = data_received["orn_offset"]
                            
            self.rotate_asm()

            data_to_send = {"state": "assembly repositioned"}
        elif data_received["request"] == "reposition_base":
            pos = data_received["pos"]
            orn = data_received["orn"]
            """p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=35.0,
                                         cameraPitch=-30,
                                         cameraTargetPosition=pos)"""
            print(f"repositioning base to: {pos}, {p.getEulerFromQuaternion(orn)}")
            self.assembly.reposition_base(pos, orn)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"spawn_point.npy"), pos)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"spawn_orient.npy"), orn)
            for _ in range(100):
                self.stepSimulation()
            self.spawn_point = pos
            self.spawn_orient = orn
            data_to_send = {"state": "assembly repositioned"}
            """p.resetDebugVisualizerCamera(cameraDistance=2.75, cameraYaw=35.0,
                                         cameraPitch=-30,
                                         cameraTargetPosition=np.array([0, 0, 0.5]))"""
        elif data_received["request"] == "reposition_part":
            print(f"repositioning {data_received['part_name']} to: "
                  f"{data_received['pos']}, {data_received['orn']}")
            self.assembly.offset_part(data_received["part_name"],
                                      data_received["pos"] * self.assembly.scaleFactor,
                                      data_received["orn"])
            for _ in range(100):
                self.stepSimulation()
            data_to_send = {"state": "part repositioned"}
        elif data_received["request"] == "set_mounting_mode":
            self.set_mounting_mode(data_received.get("mode", "floor"))
            data_to_send = {"state": f"mounting_mode set to {self._mounting_mode}"}
        elif data_received["request"] == "fix_assembly_to_world":
            part_name = data_received.get("part_name", "base")
            self.fix_assembly_to_world(part_name)
            data_to_send = {"state": "assembly fixed to world"}
        elif data_received["request"] == "load_fixture":
            fixture_type = data_received.get("fixture_type", "large")
            position = data_received.get("position", None)
            orientation = data_received.get("orientation", None)
            self.load_clamping_fixture(fixture_type, position, orientation)
            data_to_send = {"state": "fixture loaded", "fixture_id": self.fixture}
        elif data_received["request"] == "release_fixture":
            self.release_clamping_fixture()
            data_to_send = {"state": "fixture released"}

        elif data_received["request"] == "plan_test_sync":
            # Real-cell synchronous movement test.
            # Plans a sequence of safe target moves for both robots using the
            # active planner, executes them in the sim (so the user can visually
            # verify the paths), and populates self.g_code_C / g_code_D so that
            # a subsequent "save_gcode" request writes the .mpf files.
            import copy as _copy
            targets_C = data_received.get("targets_C", [
                ([0.0,  -1.5,  1.8], [0.0, 0.0, 0.0]),
                ([0.2,  -1.5,  1.8], [0.0, 0.0, 0.0]),
                ([0.0,  -1.5,  2.0], [0.0, 0.0, 0.0]),
            ])
            targets_D = data_received.get("targets_D", [
                ([-1.99, -2.0,  1.8], [0.0, 0.0, 0.0]),
                ([-2.2,  -2.0,  1.8], [0.0, 0.0, 0.0]),
                ([-1.99, -2.0,  2.0], [0.0, 0.0, 0.0]),
            ])
            feed = data_received.get("feed", 5000.0)
            self.g_code_C = []
            self.g_code_D = []
            for (pos_C, orn_C), (pos_D, orn_D) in zip(targets_C, targets_D):
                q_C = p.getQuaternionFromEuler(orn_C)
                q_D = p.getQuaternionFromEuler(orn_D)
                self._execute_move_both(
                    target_pos_C=pos_C, target_orn_q_C=q_C,
                    target_pos_D=pos_D, target_orn_q_D=q_D,
                    tool_targets=True,
                    feed_rate_C=feed, feed_rate_D=feed,
                )
            data_to_send = {"state": "test_sync_planned",
                            "waypoints_C": len(self.g_code_C),
                            "waypoints_D": len(self.g_code_D)}

        elif data_received["request"] == "cooperative_carry":
            self.g_code_C = []
            self.g_code_D = []
            data_to_send = self.cooperative_carry(exec_stride=data_received.get("exec_stride"),
                                                  ghost_gripper_C=data_received.get("ghost_gripper_C", False),
                                                  physics=bool(data_received.get("physics", False)))
        elif data_received["request"] in ("simulate", "check_feasibility", "configure_planner"):
            data_to_send = self.cell_simulation.handle(data_received)
        else:
            raise sim_server_protocol.InvalidRequest(f"unknown request {data_received['request']!r}")
        return data_to_send

    def stepSimulation(self):
        p.stepSimulation(self.pysics_client)

    def abfrage_orientierung_asm(self, vorschlag_orientierung, bes_asm_pose):
        root = tk.Tk()# Fenster für die Eingabe erstellen
        root.title("Position und Orientierung Eingabe")
        root.geometry("400x300")  
        root.withdraw()  # Fenster nicht anzeige nur Dialoge verwenden
        besondere_asm_pose = bes_asm_pose # simpledialog.askstring("Werkstueck Pose einstellen?", f"Besondere Werkstueck Pose gewuenscht? Alles außer Nein ist ja:", initialvalue="Nein") # falls mal anders gewünscht das hier wieder reinnehmen
        #besondere_asm_pose = "Nein"
        vorschlag_orient = p.getEulerFromQuaternion(vorschlag_orientierung)

        if besondere_asm_pose == 'Nein':
            orient_x = vorschlag_orient[0] # + 180, damit der Deckel weg vom EEF zeigt
            orient_y = vorschlag_orient[1] 
            orient_z = vorschlag_orient[2] + np.pi
        else:
            # Eingabe der Orientierung
            orient_x = (simpledialog.askfloat("Orientierung in Gradmaß", f"Orientierung X (0 bedeutet Ausgangstellung auf TCP angepasst. inkludierter Offset: {vorschlag_orient[0]}):", minvalue=-360, maxvalue=360, initialvalue=0)+180)*np.pi/180 - vorschlag_orient[0] # + 180, damit der Deckel weg vom EEF zeigt
            orient_y = simpledialog.askfloat("Orientierung in Gradmaß", f"Orientierung Y (0 bedeutet Ausgangstellung auf TCP angepasst. inkludierter Offset: {vorschlag_orient[1]}):", minvalue=-360, maxvalue=360, initialvalue=0)*np.pi/180 - vorschlag_orient[1]
            orient_z = simpledialog.askfloat("Orientierung in Gradmaß", f"Orientierung Z (0 bedeutet Ausgangstellung auf TCP angepasst. inkludierter Offset: {vorschlag_orient[2]}):", minvalue=-360, maxvalue=360, initialvalue=0)*np.pi/180 - vorschlag_orient[2]
            
        
        orientierung = p.getQuaternionFromEuler([orient_x, orient_y, orient_z])
        print(f"pos und ori: {orientierung}")
        return orientierung
    
    def working_pose(self, vorschlag_position, vorschlag_orientierung): # gewuenschte Pose vom Roboter EEF in der das Werkstueck gehalten werden soll. 
        root = tk.Tk()# Fenster für die Eingabe erstellen
        root.title("Position und Orientierung Eingabe")
        root.geometry("400x300")  
        root.withdraw()  # Fenster nicht anzeige nur Dialoge verwenden
        vorschlag_orient = p.getEulerFromQuaternion(vorschlag_orientierung)
        pos_x = simpledialog.askfloat(f"Aktuelle X Positionen: {vorschlag_position[0]}", f"Gewuenschte X-Kooridnate: ", initialvalue=vorschlag_position[0])
        pos_y = simpledialog.askfloat(f"Aktuelle X Positionen: {vorschlag_position[1]}", f"Gewuenschte Y-Kooridnate: ", initialvalue=vorschlag_position[1])
        pos_z = simpledialog.askfloat(f"Aktuelle X Positionen: {vorschlag_position[2]}", f"Gewuenschte Z-Kooridnate: ", initialvalue=vorschlag_position[2])
        orient_x = simpledialog.askfloat(f"Aktuelle X Rotation in Grad: {vorschlag_orient[0]*180/np.pi}", f"Gewuenschte Rotation um X in Grad: ", initialvalue=vorschlag_orient[0]*180/np.pi) * np.pi /180
        orient_y = simpledialog.askfloat(f"Aktuelle Y Rotation in Grad: {vorschlag_orient[1]*180/np.pi}", f"Gewuenschte Rotation um Y in Grad: ", initialvalue=vorschlag_orient[1]*180/np.pi) * np.pi /180
        orient_z = simpledialog.askfloat(f"Aktuelle Z Rotation in Grad: {vorschlag_orient[2]*180/np.pi}", f"Gewuenschte Rotation um Z in Grad: ", initialvalue=vorschlag_orient[2]*180/np.pi) * np.pi /180


        position = [pos_x, pos_y, pos_z]
        orientierung = p.getQuaternionFromEuler([orient_x, orient_y, orient_z])
        
        print(f"Workorientierung:  {orient_x, orient_y, orient_z}")
        self.robofixture.set_tool_pose(position, orientierung)


    def _load_assembly_on_surface(self, xy, surface_z, clearance=0.002):
        """Place self.assembly resting on a horizontal surface and pin it to the world.

        Loads at a safe height in the assembly's declared canonical orientation,
        reads the combined AABB, then teleports everything down so the lowest point
        sits `clearance` above `surface_z`.  Works for any assembly regardless of
        how its model frame is authored.
        """
        safe_z = surface_z + 1.0
        spawn = [float(xy[0]), float(xy[1]), safe_z]
        orn = self.assembly.canonical_orient
        self.assembly.load_in_sim(scale_factor=0.001,
                                  spawn_point=spawn,
                                  spawn_orient=orn,
                                  screw_scale_factor=self.assembly.screw_scale_factor)

        # AABBs are available straight from geometry, no physics steps needed.
        aabbs = [p.getAABB(pid) for pid in self.assembly.part_ids.values()]
        min_z = min(a[0][2] for a in aabbs)
        max_z = max(a[1][2] for a in aabbs)
        z_shift = surface_z + clearance - min_z
        print(f"[{self._mounting_mode} load] surface_z={surface_z:.3f}  "
              f"assembly_height={max_z - min_z:.4f} m  z_shift={z_shift:+.4f}", flush=True)

        for pid in self.assembly.part_ids.values():
            pos, q = p.getBasePositionAndOrientation(pid)
            p.resetBasePositionAndOrientation(pid, [pos[0], pos[1], pos[2] + z_shift], q)
        self.assembly.reposition_base([spawn[0], spawn[1], safe_z + z_shift], orn)

    def loadAssembly(self, path_to_xml, use_loading_point=False):
        self._loaded_assembly_file = os.path.basename(str(path_to_xml))
        
        print(f" robofixture pose - load assembly start: {self.robofixture.get_tool_pose('tcp')}")
        
        # Visualize the target position with a red debug point
        # Create a green sphere at the target position
        #visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1])
        #p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=self.work_point_C)

        # Draw coordinate system axes
        axis_length = 0.2  # Length of each axis
        orientation = self.work_orient_C
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        # Calculate the end points of each axis based on orientation
        x_end = self.work_point_C + axis_length * rot_matrix[:, 0]
        y_end = self.work_point_C + axis_length * rot_matrix[:, 1]
        z_end = self.work_point_C + axis_length * rot_matrix[:, 2]

        # Draw colored lines for each axis (X=red, Y=green, Z=blue)
        p.addUserDebugLine(self.work_point_C, x_end, [1, 0, 0], lineWidth=2)  # X-axis (red)
        p.addUserDebugLine(self.work_point_C, y_end, [0, 1, 0], lineWidth=2)  # Y-axis (green)
        p.addUserDebugLine(self.work_point_C, z_end, [0, 0, 1], lineWidth=2)  # Z-axis (blue)

        #self.robot_C.set_endeffector_pose(self.work_point_C, self.work_orient_C)
        self.robofixture.set_tool_pose(self.work_point_C, self.work_orient_C)
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        
        for _ in range(250):
            self.stepSimulation()
        
        #print(f" robofixture pose nach sim-step : {self.robofixture.get_tool_pose("tcp")}")
        #print(f" robot pose nach sim-step : {self.robot_C.get_endeffector_pose()}")

        del self.assembly
        self.assembly = None # damit nicht mehrere Assemblys in der Demo sind
        self._coop = None
        self.assembly = AssemblyPB.load_from_xml(path_to_xml)
        orienti = p.getEulerFromQuaternion(self.work_orient_C)  # always defined
        # The mounting mode decides only WHERE the assembly goes.  How the model is
        # oriented and scaled is declared in the assembly XML (see AssemblyPB), so
        # nothing here is specific to any one assembly.
        if use_loading_point:
            from modular_planner.tpg import DangerZone
            self._set_desk(self._mounting_mode != 'floor')
            if self._mounting_mode == 'floor':
                # Visual floor (room geometry at z=0), not the physics plane at -0.05.
                self._load_assembly_on_surface(self.FLOOR_SPAWN_POINT[:2], surface_z=0.0)
                self.danger_zone = DangerZone(tuple(self.FLOOR_SPAWN_POINT[:2]),
                                              self.danger_zone_half_m, 0.0)
            elif self._mounting_mode == 'table':
                self._load_assembly_on_surface(self.clamping_desk_centre,
                                               surface_z=self.clamping_desk_top_z)
                self.danger_zone = DangerZone(tuple(self.clamping_desk_centre),
                                              self.danger_zone_half_m, self.clamping_desk_top_z)
            else:
                self.danger_zone = None
                self.assembly.load_in_sim(scale_factor=0.001, spawn_point=self.loading_point,
                                          spawn_orient=self.loading_orient)
        else:
            #eefpos, eeforn =self.robofixture.get_tool_pose("tcp")
            adapted_spawn_ori = self.abfrage_orientierung_asm(self.work_orient_C, "Nein") #in der funktion kann man einfach +/- winkel machen um ori anzupassen # fixes orientation from assembly to the work orientation
            rot_mat = np.array(p.getMatrixFromQuaternion(self.work_orient_C)).reshape(3,3)
            x_axis = rot_mat[:,0]
            self.x_offset_anglegrinder = 0.09#.17

            offset = self.x_offset_anglegrinder * x_axis
            new_position = self.work_point_C + offset

            self.assembly.load_in_sim(scale_factor=0.001, spawn_point = new_position, spawn_orient = adapted_spawn_ori)
            self.stick_robofixture(self.work_point_C, self.work_orient_C)
            print(f"loading from given position and orientation: {self.work_point_C}, {self.work_orient_C}")
        for _ in range(50):
            self.stepSimulation()

        # Legacy robofixture workflow only: robot_C carries the assembly to its
        # work point, so that move opens the C program.  In floor/table mode
        # robot_C never goes there; exporting it put a (-1, -1.6, 2.0) block
        # before the first segment (Async Run 3 died on it: axis 5 + limit,
        # alarms 75313/10720 at block 11).
        if not use_loading_point:
            g_code_temp = [{'G': 1,
                            'X': self.work_point_C[0], 'Y': self.work_point_C[1],
                            'Z': self.work_point_C[2],
                            'A': orienti[0], 'B': orienti[1], 'C': orienti[2],
                            'F': 10000.0}]
            self.g_code_C = self.g_code_C + g_code_temp
        for _ in range(30):
            self.stepSimulation()

    def loadAssemblyop(self, path_to_xml):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        del self.assembly
        self.assembly = None # damit nicht mehrere Assemblys in der Demo sind
        self.assembly = AssemblyPB.load_from_xml(path_to_xml)
        eefpos, eeforn =self.robofixture.get_tool_pose("tcp")
    
        self.working_pose(eefpos, eeforn)
        root = tk.Tk()# Fenster für die Eingabe erstellen
        root.title("Werkstueck orientierung aendern?")
        root.geometry("500x300")  
        root.withdraw()  # Fenster nicht anzeige nur Dialoge verwenden
        ws_orientanpassung = simpledialog.askstring("Soll die Orientierung des Werkstuecks noch geaendert werden?: ", f"Alles ausser Nein bedeutet ja: ", initialvalue="Nein")
        if ws_orientanpassung == "Nein":
            so = self.abfrage_orientierung_asm(eeforn, "Nein")
        else:
            so = self.abfrage_orientierung_asm(eeforn, "Ja")
            
        self.assembly.load_in_sim(scale_factor=0.001, spawn_point = eefpos, spawn_orient = so)
        
        self.stick_robofixture(eefpos, eeforn)
                    
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        for _ in range(100):
            self.stepSimulation()

    # ------------------------------------------------------------------
    # Planner selection
    # ------------------------------------------------------------------

    def set_planner(self, planner_type, **kwargs):
        """Select the motion planner used for long-travel moves on robot_D.

        Parameters
        ----------
        planner_type : PlannerType
            ``PlannerType.DIRECT`` — original single G1 move (default, no extra
            dependencies).
            ``PlannerType.GNN``    — RoboBallet GNN generates intermediate
            collision-free waypoints.
        **kwargs
            Planner-specific keyword arguments:

            For PlannerType.GNN:
                checkpoint_dirs (required) Dict mapping task name → checkpoint
                    path: {'independent': '/path/stage4', 'cooperative': '/path/stage7'}.
                    A missing or None path disables that task (raises RuntimeError
                    if plan/plan_multi is called with that task).
                roboballet_root (optional) Path containing the roboballet package.
                max_plan_steps, convergence_tol, workspace_scale, … (see
                GNNWaypointPlanner for the full list).

        Examples
        --------
        sim = WSKAgiloDrive()

        # Keep original behaviour (explicit, or just don't call set_planner):
        sim.set_planner(PlannerType.DIRECT)

        # Enable GNN collision-aware planning:
        sim.set_planner(
            PlannerType.GNN,
            checkpoint_dirs={
                'independent': '/home/user/robotics_env/checkpoints/stage4_parallel',
                'cooperative': '/home/user/robotics_env/checkpoints/stage7',
            },
            roboballet_root='/home/user/robotics_env/robo_ballet',
        )
        """
        from modular_planner import PlannerType
        self._planner_type = planner_type

        if planner_type == PlannerType.DIRECT:
            self._active_planner = None
            print('[Planner] DIRECT selected — single G1 move (original behaviour).')

        elif planner_type == PlannerType.GNN:
            from modular_planner import GNNPlannerRegistry
            checkpoint_dirs = kwargs.pop('checkpoint_dirs')
            roboballet_root = kwargs.pop('roboballet_root', None)
            self._active_planner = GNNPlannerRegistry(
                checkpoint_dirs=checkpoint_dirs,
                roboballet_root=roboballet_root,
                **kwargs,
            )
            avail = self._active_planner.available_tasks()
            print(f'[Planner] GNN selected. Available tasks: {avail}')

        elif planner_type == PlannerType.CUMOTION:
            from modular_planner import CuMotionWaypointPlanner
            urdf_path = kwargs.pop('urdf_path', None)
            xrdf_path = kwargs.pop('xrdf_path', None)
            planning_config_path = kwargs.pop('planning_config_path', None)
            # robot_C is built from the plain URDF, robot_D from the sensor
            # URDF; cuMotion must plan each arm with its own tcp_frame chain
            # (they differ by 9 cm and a 90 deg yaw).
            overrides = kwargs.pop('urdf_overrides', None)
            if overrides is None:
                overrides = {'robot_C': self._urdf_robot_C, 'robot_D': self._urdf_robot_D}
            # Comau Ax.3 (absolute forearm angle) = q3 - q2 must stay in [-220, 0] deg
            limits = []
            if PARALLELOGRAM_LIMIT:
                limits.append({'name': 'parallelogram', 'i': 1, 'j': 2, 'k': 1.0,
                               'lo': math.radians(PARALLELOGRAM_LIMIT['lo']),
                               'hi': math.radians(PARALLELOGRAM_LIMIT['hi'])})
            # wrist cable sum: -lim <= q6 + q4 <= lim   (q[j] - k*q[i] with k = -1)
            limits.append({'name': 'wrist cables', 'i': 3, 'j': 5, 'k': -1.0,
                           'lo': -math.radians(WRIST_CABLE_SUM_LIMIT_DEG),
                           'hi': math.radians(WRIST_CABLE_SUM_LIMIT_DEG)})
            kwargs.setdefault('coupled_limit', limits)
            self._active_planner = CuMotionWaypointPlanner(
                urdf_path=urdf_path,
                xrdf_path=xrdf_path,
                planning_config_path=planning_config_path,
                urdf_overrides=overrides,
                **kwargs,
            )
            print(f'[Planner] CUMOTION selected (Comau NJ290, tcp_frame; per-arm URDFs: '
                  f'{ {k: os.path.basename(v) for k, v in overrides.items()} }).')

        else:
            raise ValueError(f'Unknown PlannerType: {planner_type}')

    def _get_eef_pose_4x4(self, robot) -> np.ndarray:
        """Return current EEF pose as a 4×4 matrix (metres)."""
        pos, orn_q = robot.get_endeffector_pose()
        rot = R.from_quat(orn_q)  # scipy: scalar-last
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = np.array(pos)
        return T

    def _get_robot_base_pose_4x4(self, robot) -> np.ndarray:
        """Return robot base pose as a 4×4 world-frame matrix (metres).

        cuMotion plans in robot-local frame (base at origin).  Passing this
        matrix lets the planner transform targets and obstacles correctly.
        """
        pos, orn_q = p.getBasePositionAndOrientation(robot.urdf)
        rot = R.from_quat(orn_q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = np.array(pos)
        return T

    def _pose_to_4x4(self, pos, orn_q) -> np.ndarray:
        """Build 4×4 matrix from position and quaternion (scalar-last)."""
        rot = R.from_quat(orn_q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = np.array(pos)
        return T

    @staticmethod
    def _resample_waypoints(waypoints: list, n: int) -> list:
        """Linearly interpolate a G-code waypoint list to exactly n entries.

        Interpolates X/Y/Z/A/B/C between consecutive waypoints so the robot
        covers its path in n equal steps.  Used to synchronise two robots with
        different path lengths so they appear to move at the same visual speed.
        Feed-rate and tool keys (T, G) are carried over from the nearest original
        waypoint.
        """
        if len(waypoints) == 0 or n <= 0:
            return waypoints
        if len(waypoints) == 1:
            return [dict(waypoints[0]) for _ in range(n)]

        INTERP_KEYS = ('X', 'Y', 'Z', 'A', 'B', 'C', 'RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6')
        result = []
        src = len(waypoints) - 1
        for i in range(n):
            # Fractional position along the original list (0 … src).
            t = i * src / (n - 1)
            lo = int(t)
            hi = min(lo + 1, src)
            frac = t - lo

            base = dict(waypoints[lo])
            if frac > 0 and lo != hi:
                for k in INTERP_KEYS:
                    if k in waypoints[lo] and k in waypoints[hi]:
                        base[k] = waypoints[lo][k] + frac * (waypoints[hi][k] - waypoints[lo][k])
            result.append(base)
        return result

    def _execute_move(self, target_pos, target_orn_q,
                      robot, g_code_processor,
                      feed_rate: float = 10000.0,
                      step_sims: int = 5,
                      robot_name: str = 'robot',
                      constraints=None,
                      g_code_log: list = None):
        """Move a robot arm to target using the currently selected planner.

        Robot-agnostic: accepts any robot object and matching g_code_processor.
        Dispatches to the planner chosen by set_planner():
          PlannerType.DIRECT  — original single G1 move (unchanged behaviour).
          PlannerType.GNN     — RoboBallet GNN generates collision-free waypoints;
                                falls back to DIRECT if planning fails.
          PlannerType.CUMOTION — NVIDIA cuMotion GPU planner; falls back to DIRECT.

        Parameters
        ----------
        target_pos:       Target EEF position [x, y, z] in metres.
        target_orn_q:     Target EEF orientation as quaternion [x, y, z, w].
        robot:            Robot object (e.g. self.robot_C or self.robot_D).
        g_code_processor: G-code processor for this robot.
        feed_rate:        G-code feed rate for all waypoints.
        step_sims:        Physics steps executed between waypoint commands.
        robot_name:       Label used in log messages and planner dispatch.
        constraints:      Optional list of SceneConstraint objects (GNN only).
        g_code_log:       Optional list to append executed waypoints to (e.g. self.g_code_D).
        """
        from modular_planner import PlannerType

        # ---- GNN planner ------------------------------------------------
        if self._planner_type == PlannerType.GNN and self._active_planner is not None:
            try:
                target_pose = self._pose_to_4x4(target_pos, target_orn_q)
                current_pose = self._get_eef_pose_4x4(robot)

                # Build obstacle list from loaded assembly parts.
                obstacles = []
                if self.assembly is not None and self.assembly.part_ids:
                    try:
                        from modular_planner import ObstacleBox
                        for part_id in self._planning_part_ids():
                            aabb_min, aabb_max = p.getAABB(part_id)
                            obstacles.append(ObstacleBox.from_pybullet_aabb(
                                np.array(aabb_min), np.array(aabb_max)))
                    except Exception as obs_err:
                        print(f'[Planner:GNN] Warning: could not build obstacles: {obs_err}')

                waypoints = self._active_planner.plan(
                    current_eef_pose=current_pose,
                    target_eef_pose=target_pose,
                    obstacles=obstacles,
                    constraints=constraints or [],
                    robot_name=robot_name,
                    task='independent',
                )
                for wp in waypoints:
                    wp['F'] = feed_rate

                print(f'[Planner:GNN] Executing {len(waypoints)} waypoints for {robot_name}.')
                if g_code_log is not None:
                    g_code_log += waypoints
                g_code_processor.g_code = waypoints
                for _ in g_code_processor:
                    for _ in range(step_sims):
                        self.stepSimulation()
                return  # Success — skip DIRECT fallback.

            except Exception as gnn_err:
                print(f'[Planner:GNN] Planning failed ({gnn_err}). Falling back to DIRECT.')

        # ---- cuMotion planner -------------------------------------------
        elif self._planner_type == PlannerType.CUMOTION and self._active_planner is not None:
            try:
                target_pose = self._pose_to_4x4(target_pos, target_orn_q)
                current_pose = self._get_eef_pose_4x4(robot)
                base_pose = self._get_robot_base_pose_4x4(robot)

                obstacles = []
                if self.assembly is not None and self.assembly.part_ids:
                    try:
                        from modular_planner import ObstacleBox
                        for part_id in self._planning_part_ids():
                            aabb_min, aabb_max = p.getAABB(part_id)
                            obstacles.append(ObstacleBox.from_pybullet_aabb(
                                np.array(aabb_min), np.array(aabb_max)))
                    except Exception as obs_err:
                        print(f'[Planner:CUMOTION] Warning: could not build obstacles: {obs_err}')

                _rid = robot.urdf
                start_conf = np.array(
                    [p.getJointState(_rid, j)[0]
                     for j in range(p.getNumJoints(_rid))
                     if p.getJointInfo(_rid, j)[2] != p.JOINT_FIXED],
                    dtype=np.float64,
                )

                waypoints = self._active_planner.plan(
                    current_eef_pose=current_pose,
                    target_eef_pose=target_pose,
                    obstacles=obstacles,
                    constraints=constraints or [],
                    robot_name=robot_name,
                    start_conf=start_conf,
                    robot_base_pose=base_pose,
                )
                for wp in waypoints:
                    wp['F'] = feed_rate

                print(f'[Planner:CUMOTION] Executing {len(waypoints)} waypoints for {robot_name}.')
                if g_code_log is not None:
                    g_code_log += waypoints
                g_code_processor.g_code = waypoints
                for _ in g_code_processor:
                    for _ in range(step_sims):
                        self.stepSimulation()
                return

            except Exception as cm_err:
                print(f'[Planner:CUMOTION] Planning failed ({cm_err}). Falling back to DIRECT.')

        # ---- DIRECT (original behaviour, always the final fallback) -----
        orn_euler = p.getEulerFromQuaternion(target_orn_q)
        g_code_temp = [{'G': 1,
                        'X': target_pos[0], 'Y': target_pos[1], 'Z': target_pos[2],
                        'A': orn_euler[0], 'B': orn_euler[1], 'C': orn_euler[2],
                        'F': feed_rate}]
        if g_code_log is not None:
            g_code_log += g_code_temp
        g_code_processor.g_code = g_code_temp
        for _ in g_code_processor:
            for _ in range(step_sims):
                self.stepSimulation()

    def _execute_move_both(self,
                           target_pos_C, target_orn_q_C,
                           target_pos_D, target_orn_q_D,
                           feed_rate_C: float = 10000.0,
                           feed_rate_D: float = 10000.0,
                           step_sims: int = 5,
                           constraints=None,
                           cooperative: bool = False,
                           tool_targets: bool = False):
        """Move robot_C and robot_D simultaneously using the selected planner.

        tool_targets: the targets are poses of the held TOOL's TCP (what G-code
        commands) rather than of the robot's flange (cuMotion's tcp_frame).
        True for task moves (screw standoffs, cell test targets); False for the
        tool-fetch moves, whose targets are flange poses over a tool holder or
        the other arm's own flange pose ("stay where you are").

        For PlannerType.GNN both robots are planned in a single GNN forward-pass
        so the network can coordinate their motions and avoid inter-robot
        collision.  Execution is interleaved step-by-step so PyBullet's
        single-threaded physics sees both robots moving at the same time.

        For PlannerType.DIRECT (or if GNN planning fails) the method falls back
        to the original behaviour: robot_C executes its move first, then robot_D.

        Parameters
        ----------
        target_pos_C / target_orn_q_C:
            Target EEF position (metres) and quaternion (x,y,z,w) for robot_C.
            Pass the current pose to keep robot_C stationary while robot_D moves.
        target_pos_D / target_orn_q_D:
            Target EEF position and quaternion for robot_D.
        feed_rate_C / feed_rate_D:
            G-code feed rates for each robot's waypoints.
        step_sims:
            Physics steps executed after each interleaved waypoint command.
        constraints:
            Optional list of SceneConstraint objects (GNN only).
        """
        from modular_planner import PlannerType

        # ---- GNN: single multi-robot rollout, interleaved execution ---------
        if self._planner_type == PlannerType.GNN and self._active_planner is not None:
            try:
                current_pose_C = self._get_eef_pose_4x4(self.robot_C)
                current_pose_D = self._get_eef_pose_4x4(self.robot_D)
                target_pose_C  = self._pose_to_4x4(target_pos_C, target_orn_q_C)
                target_pose_D  = self._pose_to_4x4(target_pos_D, target_orn_q_D)

                # Build obstacles from the cell furniture and the assembly parts.
                obstacles = self._scene_obstacles()
                if self.assembly is not None and self.assembly.part_ids:
                    try:
                        from modular_planner import ObstacleBox
                        for part_id in self._planning_part_ids():
                            aabb_min, aabb_max = p.getAABB(part_id)
                            obstacles.append(ObstacleBox.from_pybullet_aabb(
                                np.array(aabb_min), np.array(aabb_max)))
                    except Exception as obs_err:
                        print(f'[Planner:GNN] Warning: could not build obstacles: {obs_err}')

                # Index 0 = robot_C, index 1 = robot_D.
                gnn_task = 'cooperative' if cooperative else 'independent'
                waypoints_C, waypoints_D = self._active_planner.plan_multi(
                    current_eef_poses=[current_pose_C, current_pose_D],
                    target_eef_poses=[target_pose_C,  target_pose_D],
                    obstacles=obstacles,
                    constraints=constraints or [],
                    robot_names=['robot_C', 'robot_D'],
                    task=gnn_task,
                )

                for wp in waypoints_C:
                    wp['F'] = feed_rate_C
                for wp in waypoints_D:
                    wp['F'] = feed_rate_D

                n_sync = max(len(waypoints_C), len(waypoints_D))
                waypoints_C = self._resample_waypoints(waypoints_C, n_sync)
                waypoints_D = self._resample_waypoints(waypoints_D, n_sync)

                print(f'[Planner:GNN] Simultaneous move — '
                      f'robot_C: {len(waypoints_C)} wps, '
                      f'robot_D: {len(waypoints_D)} wps.')

                self.g_code_C = self.g_code_C + waypoints_C
                self.g_code_D = self.g_code_D + waypoints_D

                # Interleaved execution: advance both processors one step at a
                # time so PyBullet sees both robots moving simultaneously.
                self.g_code_processor_C.g_code = waypoints_C
                self.g_code_processor_D.g_code = waypoints_D
                iter_C = iter(self.g_code_processor_C)
                iter_D = iter(self.g_code_processor_D)
                done_C = done_D = False
                while not (done_C and done_D):
                    if not done_C:
                        try:
                            next(iter_C)
                        except StopIteration:
                            done_C = True
                    if not done_D:
                        try:
                            next(iter_D)
                        except StopIteration:
                            done_D = True
                    for _ in range(step_sims):
                        self.stepSimulation()
                return  # Success — skip DIRECT fallback.

            except Exception as gnn_err:
                print(f'[Planner:GNN] plan_multi failed ({gnn_err}). '
                      f'Falling back to DIRECT (sequential).')

        # ---- cuMotion: decoupled replanning loop, interleaved execution -----
        elif self._planner_type == PlannerType.CUMOTION and self._active_planner is not None:
            try:
                self._unwind_continuous_joints('C')
                self._unwind_continuous_joints('D')
                current_pose_C = self._get_eef_pose_4x4(self.robot_C)
                current_pose_D = self._get_eef_pose_4x4(self.robot_D)
                target_pose_C  = self._pose_to_4x4(target_pos_C, target_orn_q_C)
                target_pose_D  = self._pose_to_4x4(target_pos_D, target_orn_q_D)

                obstacles = self._scene_obstacles()
                if self.assembly is not None and self.assembly.part_ids:
                    try:
                        from modular_planner import ObstacleBox
                        for part_id in self._planning_part_ids():
                            aabb_min, aabb_max = p.getAABB(part_id)
                            obstacles.append(ObstacleBox.from_pybullet_aabb(
                                np.array(aabb_min), np.array(aabb_max)))
                    except Exception as obs_err:
                        print(f'[Planner:CUMOTION] Warning: obstacles failed: {obs_err}')

                def _body_id(robot_obj):
                    return robot_obj.urdf

                def _active_joints(robot_obj):
                    rid = _body_id(robot_obj)
                    return [j for j in range(p.getNumJoints(rid))
                            if p.getJointInfo(rid, j)[2] != p.JOINT_FIXED]

                start_conf_C = self._planner_start_conf('C')
                start_conf_D = self._planner_start_conf('D')

                # Each arm as obstacles for the other: cuMotion's own collision
                # spheres + tool (p.getAABB(body) would be the pedestal only).
                def _aabb_C():
                    return self._arm_obstacle_aabbs('C')
                def _aabb_D():
                    return self._arm_obstacle_aabbs('D')

                # Base transforms: cuMotion plans in robot-local frame.
                def _robot_base_4x4(robot_obj):
                    body_id = robot_obj.urdf
                    pos, orn_q = p.getBasePositionAndOrientation(body_id)
                    from scipy.spatial.transform import Rotation as R
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = R.from_quat(orn_q).as_matrix()
                    T[:3, 3] = np.array(pos, dtype=np.float64)
                    return T

                # Targets are TOOL poses; cuMotion plans the flange (see
                # _plan_transit_gcode): offset the targets, un-offset the result.
                T_off_C = self._tool_offset('C') if tool_targets else np.eye(4)
                T_off_D = self._tool_offset('D') if tool_targets else np.eye(4)
                waypoints_list = self._active_planner.plan_multi(
                    current_eef_poses=[current_pose_C, current_pose_D],
                    target_eef_poses=[target_pose_C @ np.linalg.inv(T_off_C),
                                      target_pose_D @ np.linalg.inv(T_off_D)],
                    obstacles=obstacles,
                    constraints=constraints or [],
                    robot_names=['robot_C', 'robot_D'],
                    start_confs=[start_conf_C, start_conf_D],
                    robot_aabb_getters=[_aabb_C, _aabb_D],
                    robot_base_poses=[_robot_base_4x4(self.robot_C),
                                      _robot_base_4x4(self.robot_D)],
                )
                waypoints_C, waypoints_D = waypoints_list[0], waypoints_list[1]
                self._flange_wps_to_tool(waypoints_C, T_off_C)
                self._flange_wps_to_tool(waypoints_D, T_off_D)
                jps = getattr(self._active_planner, 'last_joint_paths_multi', None)
                if self.joint_space_transits and jps is not None:
                    # Execute the planner's joint paths (see _plan_transit_gcode).
                    # The exact target poses appended by plan_multi stay Cartesian.
                    def as_joint_lines(wps, jp, feed):
                        n = min(len(wps), len(jp))
                        out = [{'G': 1, **{f'RA{i + 1}': float(v) for i, v in enumerate(q)}, 'F': feed}
                               for q in jp[:n]]
                        return out + wps[n:]
                    waypoints_C = as_joint_lines(waypoints_C, jps[0], feed_rate_C)
                    waypoints_D = as_joint_lines(waypoints_D, jps[1], feed_rate_D)

                for wp in waypoints_C:
                    wp['F'] = feed_rate_C
                for wp in waypoints_D:
                    wp['F'] = feed_rate_D

                # Upsample the shorter list so both robots take the same number
                # of sim steps and appear to move at coordinated visual speeds.
                n_sync = max(len(waypoints_C), len(waypoints_D))
                waypoints_C = self._resample_waypoints(waypoints_C, n_sync)
                waypoints_D = self._resample_waypoints(waypoints_D, n_sync)

                print(f'[Planner:CUMOTION] Simultaneous move — '
                      f'robot_C: {len(waypoints_C)} wps, '
                      f'robot_D: {len(waypoints_D)} wps.')

                self.g_code_C = self.g_code_C + waypoints_C
                self.g_code_D = self.g_code_D + waypoints_D

                self.g_code_processor_C.g_code = waypoints_C
                self.g_code_processor_D.g_code = waypoints_D
                iter_C = iter(self.g_code_processor_C)
                iter_D = iter(self.g_code_processor_D)
                done_C = done_D = False
                while not (done_C and done_D):
                    if not done_C:
                        try:
                            next(iter_C)
                        except StopIteration:
                            done_C = True
                    if not done_D:
                        try:
                            next(iter_D)
                        except StopIteration:
                            done_D = True
                    for _ in range(step_sims):
                        self.stepSimulation()
                return

            except Exception as cm_err:
                print(f'[Planner:CUMOTION] plan_multi failed ({cm_err}). '
                      f'Falling back to DIRECT (sequential).')

        # ---- DIRECT fallback: original sequential behaviour -----------------
        # robot_C first, then robot_D.
        print('[Planner:DIRECT] Simultaneous move (sequential fallback) — robot_C then robot_D.', flush=True)
        orn_euler_C = p.getEulerFromQuaternion(target_orn_q_C)
        g_code_C = [{'G': 1,
                     'X': target_pos_C[0], 'Y': target_pos_C[1], 'Z': target_pos_C[2],
                     'A': orn_euler_C[0],  'B': orn_euler_C[1],  'C': orn_euler_C[2],
                     'F': feed_rate_C}]
        self.g_code_C = self.g_code_C + g_code_C
        self.g_code_processor_C.g_code = g_code_C
        for _ in self.g_code_processor_C:
            for _ in range(step_sims):
                self.stepSimulation()

        orn_euler_D = p.getEulerFromQuaternion(target_orn_q_D)
        g_code_D = [{'G': 1,
                     'X': target_pos_D[0], 'Y': target_pos_D[1], 'Z': target_pos_D[2],
                     'A': orn_euler_D[0],  'B': orn_euler_D[1],  'C': orn_euler_D[2],
                     'F': feed_rate_D}]
        self.g_code_D = self.g_code_D + g_code_D
        self.g_code_processor_D.g_code = g_code_D
        for _ in self.g_code_processor_D:
            for _ in range(step_sims):
                self.stepSimulation()

    # ------------------------------------------------------------------

    def safe_D(self):
        g_code_temp = []
        pos = self.save_point_D
        orn = p.getEulerFromQuaternion(self.save_orient_D)
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C':orn[2], 'F': 10000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(20):
            self.stepSimulation()

    def safe_C(self):
        g_code_temp = []
        pos = self.save_point_C
        orn = p.getEulerFromQuaternion(self.save_orient_C)
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C':orn[2], 'F': 10000.0})
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(20):
            self.stepSimulation()

    def store_millingtool(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_mill)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_mill)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.pos_mill[0], 'Y': self.pos_mill[1], 'Z': self.pos_mill[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_mill[0], 'Y': self.pos_mill[1], 'Z': self.pos_mill[2],
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 7})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_mill[0], 'Y': self.start_pos_mill[1], 'Z': self.start_pos_mill[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 7})
        self.g_code_D = self.g_code_D + g_code_temp

    def unstore_millingtool(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_mill)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_mill)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_mill[0], 'Y': self.start_pos_mill[1], 'Z': self.start_pos_mill[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_mill[0], 'Y': self.start_pos_mill[1], 'Z': self.start_pos_mill[2],
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 8})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_mill[0], 'Y': self.pos_mill[1], 'Z': self.pos_mill[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 0, 'X': self.pos_mill[0], 'Y': self.pos_mill[1], 'Z': self.pos_mill[2] + 0.4,
             'A': 0.0, 'B': 0.0, 'C': np.pi/2, 'F': 5000.0})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(100):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 8})
        self.g_code_D = self.g_code_D + g_code_temp

    def mill(self, part_name: str, path: list, direction: np.array):
        tool_path = pi.ToolPath(
            np.transpose([pnt * self.assembly.scaleFactor for pnt in path]))  # , np.transpose(quaternions))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path, orn)
        tool_path.translate(pos)
        direction_r, _ = p.multiplyTransforms([0, 0, 0],
                                              orn,
                                              direction,
                                              [0, 0, 0, 1])
        _, current_dir_mill = self.millingtool.get_tool_pose()
        _, mill_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction_r,
                                                                     current_dir_mill),
                                              [0, 0, 0],
                                              current_dir_mill)
        tool_path.orientations = np.array([mill_orient for _ in path]).transpose()
        tool_path.draw(False, [1, 0, 0])
        g_code_temp = []
        pos = np.array(tool_path.positions[:, 0]) + 0.05 * np.array(direction_r)
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        g_code_temp.append({'M': 50})
        for i in range(len(path)):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append(
                {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 500.0}) # Speed while milling,  @TODO: connect to cell_config  
        
        pos = np.array(tool_path.positions[:, -1]) + 0.05 * np.array(direction_r)
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 1000.0}) # pulling milling tool out of the workpiece, geht in travel- oder toolchangetime ein
        g_code_temp.append({'M': 51})    
        pos = np.array(tool_path.positions[:, -1]) + 0.25 * np.array(direction_r)
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0}) #getting more distance to the workpiece
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
    
    def rotate_asm(self):
        # Create a quaternion for 90-degree rotation around Z-axis
        rot_z_90 = p.getQuaternionFromEuler([0, 0, -np.pi/2])  # 90 degrees in radians is π/2

        print(f" urpsrungsorientierung: {self.work_orient_C}")
        # Multiply the original orientation with the rotation
        _, new_orientation = p.multiplyTransforms([0, 0, 0], rot_z_90, [0, 0, 0], self.work_orient_C)
        new_ori_euler = p.getEulerFromQuaternion(new_orientation)

        print(f"new pose: point: {self.work_point_C}, orientation (euler): {new_orientation}")
        p.removeConstraint(self.robofixture_gripping_constraint) # remove constraint to allow rotation
        
        g_code_temp = []
        self.robofixture.set_tool_pose(self.work_point_C, new_orientation)
        ori_adapted_kos = self.abfrage_orientierung_asm(new_orientation, "Nein")
        self.assembly.reposition_base(self.work_point_C, ori_adapted_kos)
        for _ in range(100):
            self.stepSimulation()

        self.stick_robofixture(self.work_point_C, new_orientation)
        
        g_code_temp.append(
            {'G': 0, 'X': self.work_point_C[0], 'Y': self.work_point_C[1], 'Z': self.work_point_C[2],
             'A': new_ori_euler[0], 'B': new_ori_euler[1], 'C': new_ori_euler[2], 'F': 1000.0})
        print("G-Code - rotate assembly: ", g_code_temp)
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()

    def store_screwdriver(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_sc)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_sc)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.pos_sc[0], 'Y': self.pos_sc[1], 'Z': self.pos_sc[2]+0.5,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 10000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_sc[0], 'Y': self.pos_sc[1], 'Z': self.pos_sc[2],
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 1})
        """g_code_temp.append(
            {'G': 1, 'X': self.start_pos_sc[0], 'Y': self.start_pos_sc[1], 'Z': self.start_pos_sc[2]+0.2,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})"""
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        #time.sleep(0.5)
        self.g_code_processor_D.active_endeffector = -1
        g_code_temp = []
        g_code_temp.append({'T': 1})
        self.g_code_D = self.g_code_D + g_code_temp

    def unstore_screwdriver(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_sc)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_sc)
        g_code_temp = []
        _pos_C, _orn_C = self.robot_C.get_endeffector_pose()
        self._execute_move_both(
            target_pos_C=np.array(_pos_C),
            target_orn_q_C=np.array(_orn_C),
            target_pos_D=np.array([self.start_pos_sc[0], self.start_pos_sc[1], self.start_pos_sc[2] + 0.5]),
            target_orn_q_D=self.start_orientation_sc,
            feed_rate_D=10000.0)
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_sc[0], 'Y': self.start_pos_sc[1], 'Z': self.start_pos_sc[2],
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 2})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_sc[0], 'Y': self.pos_sc[1], 'Z': self.pos_sc[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0}) # pose die gehalten werden soll nachdem unstore spielt auch danach in das ranfahren an das Werkstück eine rolle

        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(100):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 2})
        self.g_code_D = self.g_code_D + g_code_temp #this lin saves the gcode to the file. the other line ("self.g_code_processor_D.g_code = g_code_temp") not!

    def switch_screw_tool(self, target_tool):
        if self.screwdriver_config['General']['current_screw_tool'] == target_tool:
            return
        else:
            self.store_screw_tool()

    def store_screw_tool(self):
        pos_0 = np.array(self.screwdriver_config['General']['magazine_start_point'].split(',')).astype(float)
        pos_n = np.array(self.screwdriver_config['General']['magazine_end_point'].split(',')).astype(float)
        num_tools = int(self.screwdriver_config['General']['magazine_size'])
        current_tool = self.screwdriver_config['General']['current_screw_tool']
        target_position = pos_0 + (pos_n - pos_0) / num_tools * int(self.screwdriver_config['Tool_positions'][current_tool])
        g_code_temp = []
        # Move above the magazine
        pos = target_position + np.array([0, 0, 0.250])
        orn = np.array([0, 0, np.pi/2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        pos = target_position + np.array([0, 0, 0.005])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        # Move to tool position and grip
        g_code_temp.append({'M': 201})
        pos = target_position
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 100.0})
        pos = target_position+ np.array([0, 0, -0.01])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 500.0})
        pos = pos + np.array([0, 0, 0.001])

        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 500.0})

        pos = pos + np.array([0, -0.030, 0.0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 500.0})
        pos[2] = 1.0
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})

        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []

    def unscrew(self, part_name: str, path: list, direction: np.array):
        if part_name not in self.assembly.part_ids:
            print(f'[unscrew] ERROR: part "{part_name}" not in assembly.part_ids. '
                  f'Available keys: {list(self.assembly.part_ids.keys())}', flush=True)
            return
        tool_path = pi.ToolPath(
            np.transpose([pnt * self.assembly.scaleFactor for pnt in path]))  # , np.transpose(quaternions))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path, orn)
        tool_path.translate(pos)
        direction_r, _ = p.multiplyTransforms([0, 0, 0],
                                              orn,
                                              direction,
                                              [0, 0, 0, 1])
        _, current_dir_sc = self.screwdriver.get_tool_pose()
        _, grip_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction_r,
                                                                     current_dir_sc),
                                              [0, 0, 0],
                                              current_dir_sc)
        tool_path.orientations = np.array([grip_orient for _ in path]).transpose()
        tool_path.draw(False, [1, 0, 0])
        g_code_temp = []

        #Pfadplanen um Schrauber in Post 0 zu bringen

        pos = tool_path.positions[:, 0] # schnelles ranfahren an Start des roten Pfades (vom Schrauben die Linie); Teil von Traveltime oder Toolchangetime
        self._execute_move(pos, tool_path.orientations[:, 0],
                           robot=self.robot_D, g_code_processor=self.g_code_processor_D,
                           feed_rate=10000.0, robot_name='robot_D', g_code_log=self.g_code_D)
        pos = tool_path.positions[:, 1]
        pos = tool_path.positions[:, 1] # mäßig schnelles ranfahren an Schraubenkopf
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0}) #if speed is changed, then also in planning.py
        g_code_temp.append({'T': 12}) #starting the screwdriver slowly, with torque triggert turnoff.
        g_code_measured_distance = []
        g_code_measured_distance.append(g_code_temp[-1]) # starting point of the red path
        #g_code_temp.append({'M': 22})
                
        for i in range(2, len(path)-1): #Such und Spiralbeweugung bei Schraubenkopf bisher nicht in berechnung einbezogen/ midpoint
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append(
                {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 1000.0})
            
        g_code_temp.append({'M':0}) # stop, so w can adjust the feed rate on sinumerik panel to 100% for the next program
        g_code_temp.append({'T': 11}) # call the unscrew programm. programm contains turning screw and moving screwdriver backwards at same time @adrian
        
        g_code_measured_distance.append(g_code_temp[-1]) # point of the red path, which matches the screw head
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        #self.stick_sc(part_name)
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        #g_code_temp.append({'M': 20})
        self.stick_sc(part_name)  # releases assembly constraint internally
        pos = tool_path.positions[:, -1] #gerade rausziehen zum ende des roten Pfades (vom Schrauben die Linie)
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 1500.0}) #if speed is changed, then also in planning.py
        g_code_measured_distance.append(g_code_temp[-1]) # end point of the red path, after pulling out the screw
        self.calculate_action_duration(g_code_measured_distance)

        #Pfadplaner um vom letzen Prozess punkt zum "approach_point" zu kommen

        drop_delta = tool_path.positions[:, -1] - tool_path.positions[:, -2]
        approach = self.storage_spaces[1]["approachpoint"] + drop_delta
        drop = self.storage_spaces[1]["droppoint"] + drop_delta*0.25
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2], 'A': 0.0, 'B':0.0, 'C': DROP_YAW_RAD, 'F': 10000.0})
        g_code_temp.append({'G': 1, 'X': drop[0], 'Y': drop[1], 'Z': drop[2]})
        g_code_temp.append({'M': 21})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
        p.removeConstraint(self.sc_gripping_constraint)
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2], 'F': 5000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
    def store_SRG(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_grip)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_grip)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.pos_grip[0], 'Y': self.pos_grip[1], 'Z': self.pos_grip[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_grip[0], 'Y': self.pos_grip[1], 'Z': self.pos_grip[2],
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 3})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_grip[0], 'Y': self.start_pos_grip[1], 'Z': self.start_pos_grip[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 3})
        self.g_code_D = self.g_code_D + g_code_temp

    def unstore_SRG(self, partname=None):
        
        if partname == "BG_Tellerrad_V1p" or  str(partname) == "BG_Tellerrad_V1p":
            self.pos_grip, self.orn_grip = self.SRG.get_tool_pose("tcp_tr")
            print(f"partname erkannt. also tellerrad fuer tcp (euler): {self.orn_grip} ")
        else:
            #t_orn_tcp = p.getEulerFromQuaternion(self.orn_grip)
            print("kein tellerrad fuer tcp")

        t_orn_tcp = p.getEulerFromQuaternion(self.orn_grip)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_grip)
        print(t_orn_tcp , t_orn)
        g_code_temp = []
        _pos_C, _orn_C = self.robot_C.get_endeffector_pose()
        self._execute_move_both(
            target_pos_C=np.array(_pos_C),
            target_orn_q_C=np.array(_orn_C),
            target_pos_D=np.array([self.start_pos_grip[0], self.start_pos_grip[1], self.start_pos_grip[2] + 0.3]),
            target_orn_q_D=self.start_orientation_grip,
            feed_rate_D=5000.0)
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_grip[0], 'Y': self.start_pos_grip[1], 'Z': self.start_pos_grip[2],
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0})
        
        print(f"Partname: {partname}, und type: {type(partname)}")
        if partname == "BG_Tellerrad_V1p" or  str(partname) == "BG_Tellerrad_V1p":
            g_code_temp.append(
                {'T': 13}) #gets the same gripper but with a tcp offset bc of uncentric gripping planning for Angle grinder Tellerrad
            print("partname erkannt. also tellerrad")
        else:
            g_code_temp.append(
                {'T': 4})
            print("kein tellerrad")
            
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_grip[0], 'Y': self.start_pos_grip[1], 'Z': self.start_pos_grip[2]+0.4, 'F': 500.0}) #srtaight up
        """g_code_temp.append(
            {'G': 1, 'X': self.pos_grip[0], 'Y': self.pos_grip[1], 'Z': self.pos_grip[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})"""
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        print(g_code_temp)
        g_code_temp = []
        if partname == "BG_Tellerrad_V1p" or  str(partname) == "BG_Tellerrad_V1p":
            g_code_temp.append(
                {'T': 13}) #gets the same gripper but with a tcp offset bc of uncentric gripping planning for Angle grinder Tellerrad
            print("partname erkannt. also tellerrad")
        else:
            g_code_temp.append(
                {'T': 4})
            print("kein tellerrad")
    
        self.g_code_D = self.g_code_D + g_code_temp
    
    def midway_point(self):
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': -1.3, 'Y': -3, 'Z': 1.5,
                     'A': 0.0, 'B': 0.0, 'C': 0.0, 'F': 10000.0})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
        self.g_code_D = self.g_code_D + g_code_temp

    def midway_point_to_storage(self):
        """Move robot_D to a safe midway pose before approaching tool storage."""
        g_code_temp = [
            {'G': 1, 'X': -1.3, 'Y': -3, 'Z': 1.5,
             'A': 0, 'B': 0, 'C': 0, 'F': 10000.0},
            {'T': 18},
        ]
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()

    def unstore_DLG(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_DLG)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_DLG)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_DLG[0], 'Y': self.start_pos_DLG[1], 'Z': self.start_pos_DLG[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_DLG[0], 'Y': self.start_pos_DLG[1], 'Z': self.start_pos_DLG[2],
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 10})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_DLG[0], 'Y': self.pos_DLG[1], 'Z': 1.0,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 10})
        self.g_code_D = self.g_code_D + g_code_temp
    def grip(self, part_name: str, path: list, direction: np.array):
        if part_name not in self.assembly.part_ids:
            print(f'[grip] ERROR: part "{part_name}" not in assembly.part_ids. '
                  f'Available keys: {list(self.assembly.part_ids.keys())}', flush=True)
            return
        #quaternions = [p.getQuaternionFromEuler(direction_to_rpy(direction)) for _ in path]
        """_, grip_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction, p.getQuaternionFromEuler(self.save_orient_D)),
                                              [0, 0, 0],
                                              p.getQuaternionFromEuler(self.save_orient_D))
        quaternions = [grip_orient for _ in path]"""
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]))#, np.transpose(quaternions))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path, orn)
        tool_path.translate(pos)
        """direction_r, _ = p.multiplyTransforms([0, 0, 0],
                                              orn,
                                              direction,
                                              [0,0,0,1])

        _, grip_orient = p.multiplyTransforms([0, 0, 0],
                                              quaternion_to_allign_z(direction_r,
                                                                     self.save_orient_D),
                                              [0, 0, 0],
                                              self.save_orient_D)"""
        direction_r, _ = p.multiplyTransforms([0, 0, 0],
                                              orn,
                                              direction,
                                              [0, 0, 0, 1])
        """# Reverse the direction to point in the opposite direction
        direction_r = -1 * np.array(direction_r)"""
        
        if part_name == "BG_Tellerrad_V1p" or  str(part_name) == "BG_Tellerrad_V1p":
            posiiiition, current_dir_srg = self.SRG.get_tool_pose("tcp_tr")
            #self.draw_coordinate_frame(posiiiition, current_dir_srg, 0.1)
            print(f"partname erkannt. also tellerrad fuer tcp (quaternion): {self.orn_grip} ")
        else:
            _, current_dir_srg = self.SRG.get_tool_pose()
            print("kein tellerrad fuer tcp")

        print(f"direction_r: {direction_r} current orn srg: {p.getEulerFromQuaternion(current_dir_srg)} rotation in euler: "
              f"{p.getEulerFromQuaternion(quaternion_to_allign_z(direction_r, current_dir_srg))}")

        rot_90_z = p.getQuaternionFromEuler([0, 0, np.pi/2])
        _, grip_orient = p.multiplyTransforms([0, 0, 0],
                              quaternion_to_allign_z(direction_r, current_dir_srg),
                              [0, 0, 0],
                              current_dir_srg)
        
        
        #self.draw_coordinate_frame(pos, grip_orient, 0.1)
        # Apply additional +90° rotation around the local z-axis (direction_r)
        _, grip_orient = p.multiplyTransforms([0, 0, 0], grip_orient, [0, 0, 0], rot_90_z)

        #time.sleep(1) # wait for the coordinate frame to be drawn
        #self.draw_coordinate_frame(pos, grip_orient, 0.1)


        tool_path.orientations = np.array([grip_orient for _ in path]).transpose()
        tool_path.draw(False, [1,0,0])

        # --- GNN-planned approach to the first grip waypoint ---
        # The initial long-travel move (robot_D → start of grip path) is the
        # segment most likely to cause collisions.  If the GNN planner is
        # active we replace it with a collision-aware trajectory; otherwise
        # we fall back to the original single-line G1 move.
        approach_start_pos = tool_path.positions[:, 0]
        approach_start_orn_q = tool_path.orientations[:, 0]

        if part_name == "BG_Tellerrad_V1p" or str(part_name) == "BG_Tellerrad_V1p":
            # Hard-coded intermediate waypoint for this specific part.
            g_code_pre = [{'G': 1, 'X': -0.5, 'Y': -2.5, 'Z': 1.5, 'F': 10000.0}]
            self.g_code_D = self.g_code_D + g_code_pre
            self.g_code_processor_D.g_code = g_code_pre
            for _ in self.g_code_processor_D:
                for _ in range(5):
                    self.stepSimulation()

        # Open gripper before approach.
        g_code_temp_pre_grip = [{'M': 11}, {'M': 0}]
        self.g_code_D = self.g_code_D + g_code_temp_pre_grip
        self.g_code_processor_D.g_code = g_code_temp_pre_grip
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()

        # Long-travel approach: both robots planned simultaneously.
        # robot_C target = its current pose so it holds position while robot_D
        # moves; the GNN still sees robot_C as an active agent and routes
        # robot_D around it.  DIRECT fallback moves C (no-op) then D.
        pos_C_hold, orn_q_C_hold = self.robot_C.get_endeffector_pose()
        self._execute_move_both(
            target_pos_C=pos_C_hold,
            target_orn_q_C=orn_q_C_hold,
            target_pos_D=approach_start_pos,
            target_orn_q_D=approach_start_orn_q,
            feed_rate_C=10000.0,
            feed_rate_D=10000.0,
        )

        # Fine-grained slow approach to the part (always direct — short move).
        g_code_temp = []
        pos = tool_path.positions[:, 1]  # height of part where we grip
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        g_code_temp.append({'M': 10})  # program to close gripper

        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        self.stick_SRG(part_name)
        g_code_temp = []
        pos = tool_path.positions[:, 2]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 1000.0}) # if pullout speed of part is changed, then also in planning.py
        drop_delta = tool_path.positions[:, 2] - tool_path.positions[:, 1]
        drop_delta[2] = min(drop_delta[2], 0.05)  # cap vertical offset so parts land in the box
        approach = self.storage_spaces[0]["approachpoint"] + drop_delta
        drop = self.storage_spaces[0]["droppoint"] + drop_delta

        if part_name == "BG_Tellerrad_V1p" or  str(part_name) == "BG_Tellerrad_V1p":
            g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2], 'A': 1.57, 'B': 0, 'C': 0, 'F': 10000.0})
        else:
            g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2], 'F': 10000.0})
        
        g_code_temp.append({'G': 1, 'X': drop[0], 'Y': drop[1], 'Z': drop[2]})
        g_code_temp.append({'M': 11}) #program to open gripper
        
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        p.removeConstraint(self.SRG_gripping_constraint)
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2]})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        """for i in range(tool_path.positions.shape[1]):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 5000.0})
        g_code_temp.insert(2, {'M':10})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(20):
                self.stepSimulation()"""
        print("done")

    def general_grip_SRG(self, part_name: str, path: list, orientations: list, grip_widths: list,
                         target_pos: np.array=None, target_orn: np.array=None,):
        quaternions = []
        offset_grip = 66.4#-10.553#94#103.3 #Zange -11 kurz 66.4
        factor = -1#-1.0 #Zange 1.342 kurz -1.0
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]),
                                np.transpose(np.array(quaternions)))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)
        tool_path.draw(True, [1,0,0])

        if target_pos is None:
            target_pos = np.array(self.storage_spaces[0]["droppoint"])
        if target_orn is None:
            target_orn = orn
        tool_path_target = invert_toolpath(tool_path)
        grip_widths_target = np.flip(grip_widths)
        apply_tool_path_rotation(tool_path_target,target_orn)
        tool_path_target.translate(target_pos)

        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)


        tool_path_target.draw(True, [0,1,0])

        g_code_temp = []
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        g_code_temp.append({'M': 12, 'F': factor*grip_widths[0] + offset_grip})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        g_code_temp.append({'M':0})
        #execute approach
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
            #print(self.SRG.get_tool_pose("tcp"))
        for _ in range(20):
            self.stepSimulation()
        #stick part to SRG
        self.stick_SRG(part_name)
        g_code_temp = []

        for i in range(1, tool_path.positions.shape[1]):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 500.0})
            if grip_widths[i] != grip_widths[i-1]:
                g_code_temp.append({'M': 12, 'F': factor*grip_widths[i]+ offset_grip})

        if target_pos is None:
            target_pos = np.array(self.storage_spaces[0]["droppoint"])
            approach = self.storage_spaces[0]["approachpoint"]
        else:
            approach = target_pos + tool_path.positions[:, -1] - tool_path.positions[:, -2]
        if target_orn is None:
            target_orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        else:
            target_orn = p.getEulerFromQuaternion(target_orn)

        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2],
                                    'A': target_orn[0], 'B': target_orn[1], 'C': target_orn[2],
                                    'F': 10000.0})
        g_code_temp.append({'G': 1, 'X': target_pos[0], 'Y': target_pos[1], 'Z': target_pos[2],
                                    'A': target_orn[0], 'B': target_orn[1], 'C': target_orn[2],
                                    'F': 5000.0})
        #g_code_temp.append({'M': 12, 'F': factor*grip_widths[0]+ offset_grip})
        g_code_temp.append({'M': 12, 'F': 0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        p.removeConstraint(self.SRG_gripping_constraint)
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2],
                                    'F': 10000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()

    def general_grip_DLG(self, part_name: str, path: list, orientations: list, grip_widths: list,
                         target_pos: np.array=None, target_orn: np.array=None, whole_assembly: bool=False,):
        temp_constraints = {}
        quaternions = []
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]),
                                np.transpose(np.array(quaternions)))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])

        if target_pos is None:
            target_pos = np.array(self.storage_spaces[0]["droppoint"])
        if target_orn is None:
            target_orn = orn
        tool_path_target = invert_toolpath(tool_path)
        grip_widths_target = np.flip(grip_widths)
        apply_tool_path_rotation(tool_path_target,target_orn)
        tool_path_target.translate(target_pos)

        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)

        tool_path.draw(True, [1,0,0])
        tool_path_target.draw(True, [0,1,0])
        g_code_temp = []
        g_code_temp.append({'M': 60})
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        pos = tool_path.positions[:, 2]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})

        for i in range(2, tool_path.positions.shape[1]-1):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 1000.0})
            if grip_widths[i] != grip_widths[i-1]:
                g_code_temp.append({'M': 61})
                g_code_temp.append({'M': 0})
                # execute approach
                self.g_code_D = self.g_code_D + g_code_temp
                self.g_code_processor_D.g_code = g_code_temp
                for _ in self.g_code_processor_D:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()
                # stick all parts that are still constraint to base to DLG
                """active_constraints = get_all_active_constraint_ids()
                print(f"active constraints: {active_constraints}")
                for part in self.assembly.parts:
                    if self.assembly.constraint_ids[part.name] in active_constraints:
                        temp_constraints[part.name] = self.stick_DLG(part.name)
                    print(f"sticking part {part.name} to DLG")"""
                #p.removeConstraint(self.assembly.constraint_ids["base"])
                if whole_assembly:
                    self.stick_DLG("base")
                else:
                    self.stick_DLG(part_name)
                print(f"active constraints: {get_all_active_constraint_ids()}")
                g_code_temp = []

        pos = tool_path.positions[:, -1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})

        pos = tool_path_target.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        break_index = 1
        for i in range(1, tool_path_target.positions.shape[1]):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 1000.0})
            if grip_widths_target[i] != grip_widths_target[i-1]:
                g_code_temp.append({'M': 0})
                g_code_temp.append({'M': 60})
                # execute approach
                self.g_code_D = self.g_code_D + g_code_temp
                self.g_code_processor_D.g_code = g_code_temp
                for _ in self.g_code_processor_D:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()
                # stick part to SRG
                """for key in temp_constraints.keys():
                    p.removeConstraint(temp_constraints[key])
                    self.assembly.offset_part_relative(key)"""
                p.removeConstraint(self.DLG_gripping_constraint)
                if whole_assembly:
                    self.assembly.reposition_base(target_pos, target_orn)
                g_code_temp = []
                break_index = i
                break

        for i in range(break_index, tool_path_target.positions.shape[1]):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 2000.0})
            #if grip_widths_target[i] != grip_widths_target[i-1]:
               # g_code_temp.append({'M': 60})#not strictly necessary, but might be useful if actual grip_widths are relevant
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()

    def general_grip_BFG(self, part_name: str, path: list, orientations: list, grip_widths: list,
                         target_pos: np.array=None, target_orn: np.array=None, whole_assembly: bool=False,):
        offset = -30.0
        scale = -5000.0
        quaternions = []
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]),
                                np.transpose(np.array(quaternions)))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])

        if target_pos is None:
            target_pos = np.array(self.storage_spaces[0]["droppoint"])
        if target_orn is None:
            target_orn = orn
        tool_path_target = invert_toolpath(tool_path)
        grip_widths_target = np.flip(grip_widths)
        apply_tool_path_rotation(tool_path_target,target_orn)
        tool_path_target.translate(target_pos)

        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)

        tool_path.draw(True, [1,0,0])
        tool_path_target.draw(True, [0,1,0])
        g_code_temp = []
       # g_code_temp.append({'M': 30})
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        g_code_temp.append({'M': 30, 'F': (grip_widths[0]+offset)*scale})

        for i in range(1, tool_path.positions.shape[1]-1):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 5000.0})
            if grip_widths[i] != grip_widths[i-1]:
                g_code_temp.append({'M': 32})
                g_code_temp.append({'M': 30, 'F': (grip_widths[1]+offset)*scale})
                g_code_temp.append({'M': 33})
                #g_code_temp.append({'M': 0})
                g_code_temp.append({'M': 30, 'F': (grip_widths[1]-0.2 + offset) * scale})
                g_code_temp.append({'M': 31})
                #g_code_temp.append({'M': 0})
                # execute approach
                self.g_code_C = self.g_code_C + g_code_temp
                self.g_code_processor_C.g_code = g_code_temp
                for _ in self.g_code_processor_C:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()

                if whole_assembly:
                    self.stick_BFG("base")
                else:
                    self.stick_BFG(part_name)
                print(f"active constraints: {get_all_active_constraint_ids()}")
                g_code_temp = []

        pos = tool_path.positions[:, -1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 3000.0})

        pos = tool_path_target.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        break_index = 1
        for i in range(1, tool_path_target.positions.shape[1]-1):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 3000.0})
            if grip_widths_target[i] != grip_widths_target[i+1]:
                g_code_temp.append({'M': 30, 'F': (grip_widths_target[i+1]+offset)*scale})
                # execute approach
                self.g_code_C = self.g_code_C + g_code_temp
                self.g_code_processor_C.g_code = g_code_temp
                for _ in self.g_code_processor_C:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()
                # stick part to SRG
                """for key in temp_constraints.keys():
                    p.removeConstraint(temp_constraints[key])
                    self.assembly.offset_part_relative(key)"""
                p.removeConstraint(self.BFG_gripping_constraint)
                if whole_assembly:
                    self.assembly.reposition_base(target_pos, target_orn)
                g_code_temp = []
                break_index = i
                break

        for i in range(break_index, tool_path_target.positions.shape[1]):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 5000.0})
            #if grip_widths_target[i] != grip_widths_target[i-1]:
               # g_code_temp.append({'M': 60})#not strictly necessary, but might be useful if actual grip_widths are relevant TODO: CHECK!!!
        self.g_code_C = self.g_code_C + g_code_temp
        self.g_code_processor_C.g_code = g_code_temp
        for _ in self.g_code_processor_C:
            for _ in range(5):
                self.stepSimulation()


    """def grip_and_place_assembly(self, part_name: str, path: list, orientations: list, grip_widths: list, target_pos: np.array=None, target_orn: np.array=None):
        temp_constraints = {}
        quaternions = []
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]),
                                np.transpose(np.array(quaternions)))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])

        if target_pos is None:
            target_pos = np.array(self.storage_spaces[0]["droppoint"])
        if target_orn is None:
            target_orn = orn
        tool_path_target = invert_toolpath(tool_path)
        grip_widths_target = np.flip(grip_widths)
        apply_tool_path_rotation(tool_path_target,target_orn)
        tool_path_target.translate(target_pos)

        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)

        tool_path.draw(True, [1,0,0])
        tool_path_target.draw(True, [0,1,0])
        g_code_temp = []
        g_code_temp.append({'M': 60})
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        pos = tool_path.positions[:, 2]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})

        for i in range(2, tool_path.positions.shape[1]-1):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 1000.0})
            if grip_widths[i] != grip_widths[i-1]:
                g_code_temp.append({'M': 61})
                g_code_temp.append({'M': 0})
                # execute approach
                self.g_code_D = self.g_code_D + g_code_temp
                self.g_code_processor_D.g_code = g_code_temp
                for _ in self.g_code_processor_D:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()
                # stick all parts that are still constraint to base to DLG
                "" "active_constraints = get_all_active_constraint_ids()
                print(f"active constraints: {active_constraints}")
                for part in self.assembly.parts:
                    if self.assembly.constraint_ids[part.name] in active_constraints:
                        temp_constraints[part.name] = self.stick_DLG(part.name)
                    print(f"sticking part {part.name} to DLG")" ""
                #p.removeConstraint(self.assembly.constraint_ids["base"])
                self.stick_DLG("base")
                print(f"active constraints: {get_all_active_constraint_ids()}")
                g_code_temp = []

        pos = tool_path.positions[:, -1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})

        pos = tool_path_target.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': max(pos[2], 0.75), 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        break_index = 1
        for i in range(1, tool_path_target.positions.shape[1]):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 1000.0})
            if grip_widths_target[i] != grip_widths_target[i-1]:
                g_code_temp.append({'M': 0})
                g_code_temp.append({'M': 60})
                # execute approach
                self.g_code_D = self.g_code_D + g_code_temp
                self.g_code_processor_D.g_code = g_code_temp
                for _ in self.g_code_processor_D:
                    for _ in range(5):
                        self.stepSimulation()
                    #print(self.DLG.get_tool_pose("tcp"))
                for _ in range(10):
                    self.stepSimulation()
                # stick part to SRG
                "" "for key in temp_constraints.keys():
                    p.removeConstraint(temp_constraints[key])
                    self.assembly.offset_part_relative(key)" ""
                p.removeConstraint(self.DLG_gripping_constraint)
                self.assembly.reposition_base(target_pos, target_orn)
                g_code_temp = []
                break_index = i
                break

        for i in range(break_index, tool_path_target.positions.shape[1]):
            pos = tool_path_target.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path_target.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 2000.0})
            #if grip_widths_target[i] != grip_widths_target[i-1]:
               # g_code_temp.append({'M': 60})#not strictly necessary, but might be useful if actual grip_widths are relevant
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        "" "approach = self.storage_spaces[0]["approachpoint"]
        drop = self.storage_spaces[0]["droppoint"]
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2], 'F': 10000.0})
        g_code_temp.append({'G': 1, 'X': drop[0], 'Y': drop[1], 'Z': drop[2]})
        g_code_temp.append({'M': 60})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        p.removeConstraint(self.DLG_gripping_constraint)
        self.assembly.fix_in_place()
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': approach[0], 'Y': approach[1], 'Z': approach[2]})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()"" " """

    def open_fixture(self):
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.350, 'Z': 0.700, 'A': 0, 'B': -np.pi/2, 'C': np.radians(120.0), 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.02383, 'Y': -0.109798, 'Z':0.328, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.020075, 'Y': -0.084798, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.02383, 'Y': -0.109798, 'Z': 0.328, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.350, 'Z': 0.700, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.850, 'Z': 0.700, 'A': 0, 'B': 0, 'C': np.radians(90.0), 'F': 5000.0})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'M': 70})
        self.g_code_D = self.g_code_D + g_code_temp

    def close_fixture(self):
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.350, 'Z': 0.700, 'A': 0, 'B': -np.pi/2, 'C': np.radians(120.0), 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.02383, 'Y': -0.109798, 'Z':0.328, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.020075, 'Y': -0.084798, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.02383, 'Y': -0.109798, 'Z': 0.328, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.350, 'Z': 0.700, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'X': 0.190, 'Y': -0.850, 'Z': 0.700, 'A': 0, 'B': 0, 'C': np.radians(90.0), 'F': 5000.0})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'M': 71})
        self.g_code_D = self.g_code_D + g_code_temp

    def set_mounting_mode(self, mode: str):
        """Set the assembly mounting mode.

        'floor'       — assembly fixed to world on the floor at FLOOR_SPAWN_POINT; both arms free.
                        Also removes the clamping desk from the scene, as both the physical body
                        and the planning obstacle block returned by `_scene_obstacles` (a desk
                        left in the scene in floor mode silently invalidated planner comparisons
                        run on the floor-mounted battery benchmark — see PROGRESS_LOG.md 2026-09-21).
        'table'       — assembly fixed to world on the clamping table top; both arms free; desk present.
        'robofixture' — robot_C holds the assembly; only robot_D is free to screw; desk present.
        """
        if mode not in ('floor', 'table', 'robofixture'):
            print(f"[set_mounting_mode] Unknown mode '{mode}'. Use 'floor', 'table' or 'robofixture'.")
            return
        self._mounting_mode = mode
        print(f'[set_mounting_mode] mounting_mode = {mode}')

    def fix_assembly_to_world(self, part_name="base"):
        """Pin the assembly base rigidly to the world at its current pose.

        Removes any existing robofixture or world-pin constraint first, then creates
        a correct world-fixed constraint at the assembly's current position.
        """
        if self.assembly is None or part_name not in self.assembly.part_ids:
            print(f"fix_assembly_to_world: part '{part_name}' not found in assembly")
            return
        # Remove robofixture constraint so the two constraints do not fight.
        if self.robofixture_gripping_constraint is not None:
            try:
                p.removeConstraint(self.robofixture_gripping_constraint)
            except Exception:
                pass
            self.robofixture_gripping_constraint = None
        # Remove any existing world-pin.
        if self.fixture_constraint is not None:
            try:
                p.removeConstraint(self.fixture_constraint)
            except Exception:
                pass
            self.fixture_constraint = None
        part_id = self.assembly.part_ids[part_name]
        part_pos, part_orn = p.getBasePositionAndOrientation(part_id)
        # Correct PyBullet world-fixed constraint:
        #   parentBody=part, childBody=-1 (world).
        #   parentFrame* = identity (joint at part origin, no rotation).
        #   childFrame*  = current world pose of the part.
        self.fixture_constraint = p.createConstraint(
            part_id, -1, -1, -1,
            p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=list(part_pos),
            childFrameOrientation=list(part_orn),
        )
        for _ in range(50):
            self.stepSimulation()

    def load_clamping_fixture(self, fixture_type="large", position=None, orientation=None):
        """Load a clamping fixture into the simulation and pin the assembly base to it.

        fixture_type: "table" | "small" | "large"
        position/orientation: world-frame pose; defaults to assembly spawn point.
        Called on-demand (e.g. for the parallel benchmark) — NOT loaded at startup.
        """
        fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
        urdf_map = {
            "table": os.path.join(fixtures_dir, "Spanntisch", "spanntisch.urdf"),
            "small": os.path.join(fixtures_dir, "Small_system", "small_clamping_system.urdf"),
            "large": os.path.join(fixtures_dir, "Large_system", "large_clamping_system.urdf"),
        }
        if fixture_type not in urdf_map:
            raise ValueError(f"Unknown fixture_type '{fixture_type}'. Choose from: {list(urdf_map)}")

        if position is None:
            position = list(self.spawn_point)
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])

        if self.fixture is not None:
            self.release_clamping_fixture()

        self.fixture = p.loadURDF(urdf_map[fixture_type], position, orientation, useFixedBase=True)

        if self.assembly is not None and "base" in self.assembly.part_ids:
            part_id = self.assembly.part_ids["base"]
            fix_pos, fix_orn = p.getBasePositionAndOrientation(self.fixture)
            part_pos, part_orn = p.getBasePositionAndOrientation(part_id)
            inv_fix_pos, inv_fix_orn = p.invertTransform(fix_pos, fix_orn)
            rel_pos, rel_orn = p.multiplyTransforms(inv_fix_pos, inv_fix_orn, part_pos, part_orn)
            self.fixture_constraint = p.createConstraint(
                self.fixture, -1, part_id, -1,
                p.JOINT_FIXED, [0, 0, 0],
                parentFramePosition=rel_pos,
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=rel_orn,
            )

        for _ in range(100):
            self.stepSimulation()

    def release_clamping_fixture(self):
        """Remove the clamping fixture and its constraint from the simulation."""
        if self.fixture_constraint is not None:
            p.removeConstraint(self.fixture_constraint)
            self.fixture_constraint = None
        if self.fixture is not None:
            p.removeBody(self.fixture)
            self.fixture = None

    def bit_on(self):
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X': -0.253, 'Y': -0.9225, 'Z': 0.500, 'A': 0, 'B': 0, 'C': np.radians(90.0), 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Z':0.265, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Y':-0.962, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Z':0.500, 'F': 5000.0})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'M': 73})
        self.g_code_D = self.g_code_D + g_code_temp

    def bit_off(self):
        g_code_temp = []
        g_code_temp.append({'G': 1, 'X':-0.253, 'Y': -0.962, 'Z': 0.500, 'A': 0, 'B': 0, 'C': np.radians(90.0), 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Z':0.265, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Y':-0.9225, 'F': 5000.0})
        g_code_temp.append({'G': 1, 'Z':0.500, 'F': 5000.0})
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'M': 72})
        self.g_code_D = self.g_code_D + g_code_temp

    def grip_cable(self, part_name: str, path: list, orientations: list):
        quaternions = []
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt * self.assembly.scaleFactor for pnt in path]),
                                np.transpose(np.array(quaternions)))
        pos, orn = p.getBasePositionAndOrientation(self.assembly.part_ids[part_name])
        #tool_path.rotate(orn)
        apply_tool_path_rotation(tool_path,orn)
        tool_path.translate(pos)
        tool_path.draw(True, [1,0,0])
        g_code_temp = []
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        g_code_temp.append({'M':10})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
            print(self.SRG.get_tool_pose("tcp"))
        for _ in range(20):
            self.stepSimulation()
        self.stick_SRG(part_name)
        g_code_temp = []
        for i in range(1, tool_path.positions.shape[1]):
            pos = tool_path.positions[:, i]
            orn = p.getEulerFromQuaternion(tool_path.orientations[:, i])
            g_code_temp.append({'G': 1, 'X':pos[0], 'Y':pos[1], 'Z':pos[2], 'A':orn[0], 'B':orn[1], 'C':orn[2], 'F': 1000.0})
        g_code_temp.append({'M': 11})
        pos = tool_path.positions[:, -1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 5000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        p.removeConstraint(self.SRG_gripping_constraint)
        print("done")

    def grip_cable_MV(self,path: list, orientations: list):
        quaternions = []
        for orn_mat in orientations:
            rot = R.from_matrix(orn_mat)
            quaternions.append(rot.as_quat())
        tool_path = pi.ToolPath(np.transpose([pnt for pnt in path]),
                                np.transpose(np.array(quaternions)))
        tool_path.draw(True, [1,0,0])
        g_code_temp = []
        pos = tool_path.positions[:, 0]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 0])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z':0.8, 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        pos = tool_path.positions[:, 1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        g_code_temp.append({'M':10})
        pos = tool_path.positions[:, 2]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, 2])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 2000.0})
        g_code_temp.append({'M':11})
        pos = tool_path.positions[:, -1]
        orn = p.getEulerFromQuaternion(tool_path.orientations[:, -1])
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        print("done")
    def store_BFG(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_BFG)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_BFG)
        t_orn = [np.pi, 0, np.pi/2]
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0]+0.7, 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append({'M': 31})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0]+0.7, 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2]+0.05,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0], 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2]+0.05,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0], 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2],
             'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 5})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_BFG[0], 'Y': self.start_pos_BFG[1], 'Z': self.start_pos_BFG[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()

        g_code_temp = []
        g_code_temp.append({'T': 5})
        self.g_code_D = self.g_code_D + g_code_temp

    def unstore_BFG(self):
        t_orn_tcp = p.getEulerFromQuaternion(self.orn_BFG)
        t_orn = p.getEulerFromQuaternion(self.start_orientation_BFG)
        print(t_orn, t_orn_tcp,  self.pos_BFG)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_BFG[0], 'Y': self.start_pos_BFG[1], 'Z': self.start_pos_BFG[2]+0.1,
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.start_pos_BFG[0], 'Y': self.start_pos_BFG[1], 'Z': self.start_pos_BFG[2],
                     'A': t_orn[0], 'B': t_orn[1], 'C': t_orn[2], 'F': 500.0})
        g_code_temp.append(
            {'T': 6})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0]+0.1, 'Y': self.pos_BFG[1], 'Z': 0,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})

        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0]+0.7, 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2],
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': self.pos_BFG[0]+0.7, 'Y': self.pos_BFG[1], 'Z': self.pos_BFG[2]+0.4,
                     'A': t_orn_tcp[0], 'B': t_orn_tcp[1], 'C': t_orn_tcp[2], 'F': 5000.0})
        #g_code_temp.append({'M': 30})
        #self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        g_code_temp.append({'T': 6})
        self.g_code_D = self.g_code_D + g_code_temp
    def move_complete_assembly(self, pos, orn): #TODO orientation change does not work yet
        orn = p.getEulerFromQuaternion(orn)
        print(pos, orn)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.assembly.part_ids["base"])
        base_orn = p.getEulerFromQuaternion(base_orn)
        g_code_temp = []
        g_code_temp.append(
            {'G': 1, 'X': base_pos[0], 'Y': base_pos[1], 'Z': base_pos[2]+0.2,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 10000.0})
        g_code_temp.append({'M': 30})
        g_code_temp.append(
            {'G': 1, 'X': base_pos[0], 'Y': base_pos[1], 'Z': base_pos[2],
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 2000.0})
        g_code_temp.append({'M': 31})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(50):
            self.stepSimulation()
        g_code_temp = []
        self.stick_BFG("base")
        g_code_temp.append(
            {'G': 1, 'X': base_pos[0], 'Y': base_pos[1], 'Z': base_pos[2]+0.2,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 2000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2]+0.2,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2]+0.05,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 2000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2],
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 500.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        for _ in range(100):
            self.stepSimulation()
        g_code_temp = []
        p.removeConstraint(self.BFG_gripping_constraint)
        self.assembly.fix_in_place()
        g_code_temp.append({'M': 30})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2]+0.1,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 5000.0})
        g_code_temp.append(
            {'G': 1, 'X': pos[0], 'Y': pos[1], 'Z': pos[2] + 0.4,
             'A': base_orn[0], 'B': base_orn[1], 'C': base_orn[2], 'F': 10000.0})
        self.g_code_D = self.g_code_D + g_code_temp
        self.g_code_processor_D.g_code = g_code_temp
        for _ in self.g_code_processor_D:
            for _ in range(5):
                self.stepSimulation()
        g_code_temp = []
    def switch_tool(self, required_tool):
        current_tool_index_D = self.g_code_processor_D.get_active_endeffector()
        current_tool_D = None if current_tool_index_D < 0 else self.g_code_processor_D.endeffector_list[current_tool_index_D]

        current_tool_index_C = self.g_code_processor_C.get_active_endeffector()
        current_tool_C = None if current_tool_index_C < 0 else self.g_code_processor_C.endeffector_list[current_tool_index_C]

        if current_tool_C != required_tool and required_tool in self.g_code_processor_C.endeffector_list:
            if current_tool_C in self.store_tool_map:
                self.store_tool_map[current_tool_C]()
            else:
                print(f"Tool {current_tool_C} not found in store_tool_map.")
            if required_tool in self.unstore_tool_map:
                self.unstore_tool_map[required_tool]()
            else:
                print(f"Tool {required_tool} not found in unstore_tool_map.")

        if current_tool_D != required_tool and required_tool in self.g_code_processor_D.endeffector_list:
            if current_tool_D in self.store_tool_map:
                self.store_tool_map[current_tool_D]()
            else:
                print(f"Tool {current_tool_D} not found in store_tool_map.")
            if required_tool in self.unstore_tool_map:
                self.unstore_tool_map[required_tool]()
            else:
                print(f"Tool {required_tool} not found in unstore_tool_map.")
    def take_frame(self, pos, orn, save_move = True):
        #@TODO: hier muss vll noch die zivid als tool eingebracht werden und der berfehl für die sinumerik in dem gcode eingefügt werden
        g_code_temp = []
        g_code_temp.append({'T':'"ZIVID"'})
        g_code_temp.append({'M':6}) #change the active tool at sinumerik to the zivid. 
        g_code_temp.append(
            {'G': 0, 'X': pos[0], 'Y': pos[1], 'Z': pos[2], 'A': orn[0], 'B': orn[1], 'C': orn[2], 'F': 10000.0})
        g_code_temp.append({'M': 40})
        
        # Check if robot D has any end effectors other than the zivid, to change back the tcp
        active_eef_index = self.g_code_processor_D.get_active_endeffector() #gets the first from list. since zivid is last, another EEF is noch index len-1
        if active_eef_index >= 0 and self.g_code_processor_D.endeffector_list[active_eef_index] != self.zivid:
            # There is an active end effector that's not the zivid
            active_eef = self.g_code_processor_D.endeffector_list[active_eef_index]
            print(f"Robot C has active end effector: {type(active_eef).__name__}")
            g_code_temp.append(
                {f'T="{active_eef}" M6'})
        else:
            self.g_code_processor_D.active_endeffector = -1
            #if no EEF is coupled then from now on the robot EEF is the active tool at sinumerik again
            g_code_temp.append({'T': 0})
            g_code_temp.append({'M': 6})

        self.g_code_D = self.g_code_D + g_code_temp
        #print(g_code_temp)
        orn_q = p.getQuaternionFromEuler(orn)
        self.zivid.set_tool_pose(pos, orn_q)
        for _ in range(150):
            self.stepSimulation()
        
        self.zivid.get_image()
        
        for _ in range(10):
            self.stepSimulation()

        self.g_code_processor_D.active_endeffector = -1  # Reset active end effector after taking the frame

    def look_at_part(self, part_name, distance = 0.45, theta = 0.0, phi = 0.0):
        """part_center = self.assembly.get_part_by_name(part_name).center_of_mass * self.assembly.scaleFactor
        rot_matrix = p.getMatrixFromQuaternion(self.assembly.spawn_orient) #TODO: this needs to change to base coordinate system of part or maybe not since its actually exactly what i need
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        part_center = rot_matrix @ part_center
        part_center += self.assembly.spawn_point
        self.take_frame(part_center + np.array([0,0,0.4]), np.array([np.pi,0,np.pi/2]))"""
        #part_center = np.array(p.getLinkState(self.assembly.part_ids[part_name], 0, physicsClientId = self.pysics_client)[0])
        #return self.take_frame(part_center + np.array([0,0,0.4]), np.array([np.pi,0,np.pi/2]))
        part_center = self.assembly.get_part_by_name(part_name).center_of_mass * self.assembly.scaleFactor
        self.look_at_point(part_center, distance, theta, phi, False)


    #ToDo look at point on assembly point distance and spherical coordinates
    def look_at_point(self, point, distance, theta, phi, save_move = True):
        v1 = np.sin(theta) * np.cos(phi)
        v2 = np.sin(theta) * np.sin(phi)
        v3 = np.cos(theta)
        direction = np.array([v1,v2,v3])
        rot_matrix = p.getMatrixFromQuaternion(self.assembly.spawn_orient) #TODO: this needs to change to base coordinate system of part or maybe not since its actually exactly what i need
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        point = rot_matrix @ point
        point += self.assembly.spawn_point # das hier vll noch auf assembly Ort machen oder Greifer endeffektor
        view_point = point + distance*direction
        #current_pos, current_orn = self.zivid.get_tool_pose()
        current_orn = self.save_orient_D
        _, view_orient = p.multiplyTransforms([0,0,0], quaternion_to_allign_z(-direction, current_orn),[0,0,0], current_orn)#p.getQuaternionFromEuler(direction_to_rpy(-direction))#vector_to_quaternion(-direction)
        self.draw_coordinate_frame(view_point, view_orient)
        self.take_frame(view_point, p.getEulerFromQuaternion(view_orient), save_move)

    def draw_coordinate_frame(self, position, orientation, length=0.2, duration=0):
        """
        Draw a coordinate frame at the specified position and orientation.
        
        Args:
            position: [x, y, z] position of the frame origin
            orientation: Quaternion [x, y, z, w] or Euler angles [roll, pitch, yaw]
            length: Length of the coordinate axes
            duration: How long to show the frame (0 = forever)
        
        Returns:
            List of line IDs for later removal if needed
        """
        if len(orientation) == 3:  # If Euler angles are provided
            orientation = p.getQuaternionFromEuler(orientation)
        
        # Create rotation matrix from quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Calculate axis endpoints
        x_axis = position + length * rot_matrix.dot([1, 0, 0])
        y_axis = position + length * rot_matrix.dot([0, 1, 0])
        z_axis = position + length * rot_matrix.dot([0, 0, 1])
        
        # Draw the three axis lines
        line_ids = [
            p.addUserDebugLine(position, x_axis, [1, 0, 0], lineWidth=2, lifeTime=duration),  # X-axis (red)
            p.addUserDebugLine(position, y_axis, [0, 1, 0], lineWidth=2, lifeTime=duration),  # Y-axis (green)
            p.addUserDebugLine(position, z_axis, [0, 0, 1], lineWidth=2, lifeTime=duration)   # Z-axis (blue)
        ]
        
        return line_ids

    def capture_image_of_cell(self):
        # Capture image from PyBullet
        image = p.getCameraImage(self.camera_width, self.camera_height, self.camera_view_matrix,
                                 self.camera_projection_matrix)#,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return np.array(image[2], dtype=np.uint8).reshape((self.camera_height, self.camera_width, 4))
    
    def calculate_action_duration(self, gcode):
        # Calculate the time for each operation step
        operation_times = []
        current_pos = None
                
        for command in gcode:
            if all(key in command for key in ['X', 'Y', 'Z']):
                new_pos = np.array([command['X'], command['Y'], command['Z']])
                
                if current_pos is not None and 'F' in command:
                    distance = np.linalg.norm(new_pos - current_pos)
                    time = distance / command['F']
                    operation_times.append(time)
                
                current_pos = new_pos

        print("operation times: ", operation_times)
        return operation_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for optional file path")

    # Define optional arguments
    parser.add_argument('--server_active', help='Wether or not to activate the TCP/IP server', default=False)

    # Planner selection
    parser.add_argument(
        '--planner',
        choices=['direct', 'gnn', 'cumotion'],
        default='direct',
        help='Motion planner for robot_D long-travel moves. '
             '"direct" = original single G1 move (default). '
             '"gnn" = RoboBallet GNN collision-aware waypoints (requires --checkpoint_dir). '
             '"cumotion" = NVIDIA cuMotion GPU motion planner.',
    )
    parser.add_argument(
        '--execution',
        choices=['reactive', 'tpg'],
        default='reactive',
        help='Dual-arm execution layer: reactive = live yield arbitration; '
             'tpg = offline Temporal Plan Graph (rehearsal, pairwise safety, WAITM per edge).')
    parser.add_argument(
        '--checkpoint_dir_independent',
        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'stage4_parallel'),
        help='GNN checkpoint for independent reaching (screws, single-robot approach). '
             'Defaults to <repo>/checkpoints/stage4_parallel/.',
    )
    parser.add_argument(
        '--checkpoint_dir_cooperative',
        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'stage7'),
        help='GNN checkpoint for cooperative transport (battery cover dual-arm carry). '
             'Defaults to <repo>/checkpoints/stage7/ (configure when Stage 7 is trained).',
    )
    parser.add_argument(
        '--roboballet_root',
        default=None,
        help='Path to the directory containing the roboballet Python package '
             '(optional when --planner gnn; defaults to <sim>/robo_ballet).',
    )
    parser.add_argument(
        '--assembly',
        default=None,
        help='Path to the assembly XML file to load (e.g. WS_ass3.xml). '
             'If omitted, no assembly is loaded at startup.',
    )

    # Parse arguments
    args = parser.parse_args()
    if args.server_active is not None:
        server_active = args.server_active
    else:
        server_active = False

    # Construct without starting the server so set_planner runs first.
    cell = WSKAgiloDrive(server_active=False, activate_shadow=False)

    # Apply planner selection before the blocking server loop starts.
    from modular_planner import PlannerType
    if args.planner == 'gnn':
        ckpt_indep = args.checkpoint_dir_independent
        ckpt_coop  = args.checkpoint_dir_cooperative
        if not os.path.isdir(ckpt_indep):
            print(f'[Planner] WARNING: checkpoint_dir_independent not found: {ckpt_indep}. '
                  f'Falling back to DIRECT. Train Stage 4 and place checkpoint under '
                  f'checkpoints/stage4_parallel/.', flush=True)
            args.planner = 'direct'
        else:
            if not os.path.isdir(ckpt_coop):
                print(f'[Planner] NOTE: checkpoint_dir_cooperative not found: {ckpt_coop}. '
                      f'Cooperative transport (Stage 7) will not be available; '
                      f'train Stage 7 or pass --checkpoint_dir_cooperative to enable.',
                      flush=True)
    if args.planner == 'gnn':
        cell.set_planner(
            PlannerType.GNN,
            checkpoint_dirs={
                'independent': args.checkpoint_dir_independent,
                'cooperative': args.checkpoint_dir_cooperative,
            },
            roboballet_root=args.roboballet_root,
            waypoint_step=20,   # 200 steps / 20 = 10 waypoints + target = 11 total
            max_plan_steps=200,
        )
    elif args.planner == 'cumotion':
        print('[Planner] initialising CUMOTION (may take 30-60s)...', flush=True)
        try:
            cell.set_planner(PlannerType.CUMOTION, waypoint_step=8)
            print('[Planner] CUMOTION ready.', flush=True)
        except Exception as _e:
            print(f'[Planner] CUMOTION init FAILED: {_e} — falling back to DIRECT', flush=True)
            cell.set_planner(PlannerType.DIRECT)
    else:
        cell.set_planner(PlannerType.DIRECT)
    cell.execution_mode = args.execution
    print(f'[sim] dual-arm execution layer: {cell.execution_mode}', flush=True)

    # Now start the blocking server loop.
    cell.start_server()

    #//////bis hier ist neu und kann probleme geben
    for _ in range(100):
        cell.stepSimulation()

    #cell.loadAssembly(r"C:\MaH\Data\Assemblies\EMO_SR_V2.xml")
    # TODO: pass assembly path via --assembly argument or environment variable
    # cell.loadAssembly(r"C:\Users\david\OneDrive\...\WS_ass3.xml")  # removed hardcoded path

    if args.assembly:
        cell.loadAssembly(args.assembly)
    #cell.assembly.reposition_base([0.0, 0.0, 1.0], p.getQuaternionFromEuler([0, np.pi/4, 0]))
    for _ in range(500000):
        cell.stepSimulation()


    transformation = np.array([[0.9736, -0.2255, -0.03513, 30.56],
                               [0.2257, 0.9742, 0.0021, 13.410],
                               [0.0337, -0.0100, 0.9994, 11.611],
                               [0,0,0,1]])
    # Extract the translation vector
    translation = np.array(transformation[:3, 3])
    # Extract the rotation matrix
    rotation_matrix = np.array(transformation[:3, :3])
    # Convert the rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns (x, y, z, w)
    temp = np.rad2deg(p.getEulerFromQuaternion(quaternion))
    #translation = np.array([0.0, 0.0, 30])
    #quaternion = p.getQuaternionFromEuler([0, np.pi/4, 0])
    cell.assembly.offset_part("05_screw_2",translation*cell.assembly.scaleFactor, quaternion)
    for _ in range(100):
        cell.stepSimulation()
    translation = np.array([0.0, 0.0, 30])
    quaternion = p.getQuaternionFromEuler([0, 0, 0])
    cell.assembly.offset_part("05_screw_2",translation*cell.assembly.scaleFactor, quaternion)
    for _ in range(1000000):
        cell.stepSimulation()
    """cell.unstore_screwdriver()
    for _ in range(500):
        cell.stepSimulation()
    cell.store_screwdriver()"""

    cell.unstore_millingtool()
    for _ in range(5000):
        cell.stepSimulation()
    cell.store_millingtool()


    cell.unstore_BFG()
    cell.loadAssembly(r"C:\MaH\Data\Assemblies\AD2V5_Sensor_cut.xml", use_loading_point=True) # kp wieso das hier drin ist. Pfad ist eh nicht vorhanden
    cell.move_complete_assembly(cell.spawn_point, cell.spawn_orient)
    cell.store_BFG()

    write_g_code(transform_eulers_in_gcode(cell.g_code_D), str(gcode_path / "004_Test.mpf"),
                 'EXTERN SRG_GRIP_LIGHT(INT)\nEXTERN DEPRAG_UNSCREW(INT, BOOL)\nEXTERN BFG_POS(REAL)\nCALL "SG_AUF.MPF"\nR0=1\n',
                 'CALL "SG_AUF.MPF"\nSTOPRE\nR0=0\nM30\n',
                 {10: "SRG_GRIP_LIGHT(1)",
                  11: "SRG_GRIP_LIGHT(0)",
                  20: ";DEPRAG_UNSCREW(1,TRUE)\n;DEPRAG_UNSCREW(2, TRUE)\n;DEPRAG_UNSCREW(2, FALSE)\n",
                  21: 'CALL "SG_ZU.MPF"',
                  30: 'BFG_MATRIX_OPEN\nBFG_POS(-1200000)\nBFG_MATRIX_CLOSE',
                  31: 'BFG_POS(-2000000)'},
                 {1: 'CALL "012_STORE_SCREW.MPF"',
                  2: 'CALL "012_GET_SCREW.MPF"',
                  3: 'CALL "010_STORE_SRG.MPF"',
                  4: 'CALL "010_GET_SRG.MPF"',
                  5: 'CALL "013_STORE_BFG.MPF"',
                  6: 'CALL "013_GET_BFG.MPF"'})

    for _ in range(10000):
        cell.stepSimulation()
    #cell.unstore_SRG()
    #cell.store_SRG()
    cell.unstore_screwdriver()
    cell.safe_D()
    #cell.store_screwdriver()
    write_g_code(transform_eulers_in_gcode(cell.g_code_D), str(gcode_path / "004_Test.mpf"),
                 'EXTERN SRG_GRIP_LIGHT(INT)\nR0=1\nT="SRG"\nSRG_GRIP_LIGHT(0)\n',
                 "T=0\nSTOPRE\nR0=0\nM30\n",
                 {10: "SRG_GRIP_LIGHT(1)", 11: "SRG_GRIP_LIGHT(0)"})
    #Calibration capture
    distance = 0.75
    steps = 5
    #pos_chessboard = np.array([-0.50,-0.65,0.34])
    pos_chessboard = np.array([-0.0,0.1,0.5])
    cell.look_at_point(pos_chessboard, distance, 0, 0, False)
    for i in range(steps):
        cell.look_at_point(pos_chessboard, distance, np.pi/8, -np.pi/2 + np.pi/(steps-1)*i, False)
    for i in range(steps):
        cell.look_at_point(pos_chessboard, distance, np.pi / 4, +np.pi / 2 - np.pi /(steps-1)* i, False)

    write_g_code(transform_eulers_in_gcode(cell.g_code_D), str(gcode_path / "004_Calibration.mpf"),
                 'R0=1\nT="ZIVID"\n',
                 "T=0\nSTOPRE\nR0=0\nM30\n",
                 {40: "WAIT_FOR_GO"})
    cell.g_code_D = []


    cell.loadAssembly(resolve_path("EMO_SR.xml"))
    contact_points = p.getContactPoints()
    path = [ np.array([-39.59797416331712, -39.59797416331712, 203.0000015058219 ]),
                        np.array([-39.59797416331712, -39.59797416331712, 103.0000015058219 ]),
                        np.array([-39.59797416331712, -39.59797416331712, 218.0000015058219 ])]
    direction = np.array([0,0,1])
    pos, orn = p.multiplyTransforms(np.array([0.0,0.0,0.0]),
                                    p.getQuaternionFromEuler([0, np.pi/8, 0]),
                                    cell.assembly.spawn_point, cell.assembly.spawn_orient)
    cell.assembly.reposition_base(pos, orn)
    for _ in range(100):
        cell.stepSimulation()

    #cell.look_at_point(np.array([0,0,0]), 0.5, np.pi/8, 0)
    cell.grip("04_screw_1", path, direction)
# cell.look_at_part("03_S1")
    """cell.look_at_part("01_upper_lid", theta=np.pi/8)
    # Convert the NumPy array to an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cell.zivid.last_frame["points"])

    # Save the PointCloud as a PLY file
    ply_filename = "01_upper_lid.ply"
    o3d.io.write_point_cloud(ply_filename, point_cloud)"""

    #inverted_dict = {value: key for key, value in cell.assembly.part_ids.items()}
    #resulting_list = [inverted_dict.get(num, str(num)) for num in cell.zivid.last_frame["index_array"]]
    cell.assembly.reposition_base(cell.assembly.spawn_point + np.array([0.2,0,0]))
    for _ in range(500):
        cell.stepSimulation()
    inv_p, inv_o = p.invertTransform(cell.assembly.spawn_point, cell.assembly.spawn_orient)
    pos, orn = p.multiplyTransforms(np.array([0.0,0.0,0.0]),
                                    p.getQuaternionFromEuler([np.pi/4, 0,0]),
                                    cell.assembly.spawn_point, cell.assembly.spawn_orient)
    cell.assembly.reposition_base(pos, orn)
    """while True:
        cell.stepSimulation()
        contact_points = p.getContactPoints()
        time.sleep(0.1)"""
    #cell.look_at_part("01_upper_lid")
    for _ in range(100):
        cell.stepSimulation()
    cell.assembly.offset_part("01_upper_lid",[0,0,0.1], p.getQuaternionFromEuler([np.pi/4, 0,0]))
    #cell.look_at_part("01_upper_lid")
    #cell.take_frame(np.array([0,0,1.0]), np.array([-np.pi,0,np.pi/2]))
    #write_g_code(transform_eulers_in_gcode(cell.g_code_D, "test.mpf", "T=ZIVID\n", "T=0\nM30\n"))
    while True:
        cell.stepSimulation()
        time.sleep(0.1)

