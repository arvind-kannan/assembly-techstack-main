import os
import sys
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two directories to reach the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
print("project_root", project_root)
# Add the relative path to the target directory
sys.path.insert(0, os.path.join(project_root, 'submodule_crc_diss_assembly_dataclasses', 'src'))
from diss_assembly_dataclasses import Assembly, Part, Connection
from diss_assembly_dataclasses.connection import Geometric_Connection, Screw_Connection, Safety_Ring_Connection, Fixed_Connection
from scipy.spatial.transform import Rotation as R
import os

class AssemblyPB(Assembly):
    def __init__(self, name, parts: list[Part] = [], connections: list[Connection] = []):
        super().__init__(name, parts, connections)
        self.part_ids = {}
        self.constraint_ids = {}
        self.part_offset_vectors = {}
        self.part_offset_orientations = {}
        self.spawn_point = [0,0,0]
        self.spawn_orient = p.getQuaternionFromEuler([0,0,0])
        self.scaleFactor = 1.0
        # Declared per assembly in the XML root so the sim needs no per-assembly code:
        #   orientation="rx ry rz"  euler (rad) that puts the model screws-up in the cell
        #   screw_scale="0.0006"    per-part scale override for parts named screw*
        self.canonical_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.screw_scale_factor = None
        # optional cooperative task (dual grip benchmark): cooperative_carrier / _shift_m / _lift_m
        self.cooperative = None

    @classmethod
    def load_from_xml(cls, filename):
        """Load assembly from XML, normalising mesh paths to absolute Linux paths.

        The XML may have been saved on Windows with backslash separators and
        paths relative to the XML file's directory.  On Linux those paths must
        be converted before trimesh / PyBullet try to open them.
        """
        tree = ET.parse(filename)
        root = tree.getroot()
        xml_dir = os.path.dirname(os.path.abspath(filename))

        assembly_name = root.attrib['name']
        assembly = cls(assembly_name)

        if 'orientation' in root.attrib:
            rx, ry, rz = (float(v) for v in root.attrib['orientation'].split())
            assembly.canonical_orient = p.getQuaternionFromEuler([rx, ry, rz])
        if 'screw_scale' in root.attrib:
            assembly.screw_scale_factor = float(root.attrib['screw_scale'])
        if 'cooperative_carrier' in root.attrib:
            assembly.cooperative = {
                'carrier': root.attrib['cooperative_carrier'],
                'shift': [float(v) for v in root.attrib.get('cooperative_shift_m', '0 0 0').split()],
                'lift': float(root.attrib.get('cooperative_lift_m', '0.25'))}

        for part_element in root.findall('part'):
            raw_mesh = part_element.attrib.get('mesh_path', '')
            # Normalise: backslash → forward slash, then make absolute
            norm_mesh = raw_mesh.replace('\\', '/')
            if not os.path.isabs(norm_mesh):
                norm_mesh = os.path.join(xml_dir, norm_mesh)
            density = float(part_element.attrib.get('density', 8000.0))
            part = Part(part_element.attrib['name'], norm_mesh, density)
            assembly.add_part(part)

        # Attributes in connection elements that may contain relative file paths
        _conn_path_attrs = ('points_filename', 'normals_filename', 'distances_filename')

        for conn_element in root.findall('connection'):
            # Fix any relative .npy paths to absolute before passing to from_xml_element
            for attr in _conn_path_attrs:
                if attr in conn_element.attrib:
                    raw = conn_element.attrib[attr].replace('\\', '/')
                    if not os.path.isabs(raw):
                        conn_element.set(attr, os.path.join(xml_dir, raw))

            conn_type = conn_element.attrib['type']
            if conn_type == 'generic':
                connection = Connection.from_xml_element(conn_element)
            elif conn_type == 'geometric':
                connection = Geometric_Connection.from_xml_element(conn_element)
            elif conn_type == 'screw':
                connection = Screw_Connection.from_xml_element(conn_element)
            elif conn_type == 'safety_ring':
                connection = Safety_Ring_Connection.from_xml_element(conn_element)
            elif conn_type == 'fixed':
                connection = Fixed_Connection.from_xml_element(conn_element)
            else:
                connection = Connection.from_xml_element(conn_element)
            assembly.add_connection(connection)

        assembly.xml_path = filename
        return assembly

    def load_in_sim(self, scale_factor=None, spawn_point=None, spawn_orient=None,
                    screw_scale_factor=None):
        if spawn_point is not None:
            self.spawn_point = spawn_point
        if spawn_orient is not None:
            self.spawn_orient = spawn_orient
        if scale_factor is not None:
            self.scaleFactor = scale_factor
        fixedBase = True
        self.parts.reverse()
        """visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=0.05,
                                            rgbaColor=[1, 0, 0, 1])
        collisionShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                            radius=0.05,
                                            rgbaColor=[1, 0, 0, 1])"""
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=f"{os.path.dirname(__file__)}/Coordinate_System.stl",
                                            rgbaColor=[0.0, 0.6, 0.0, 1],
                                            meshScale=[0.0001, 0.0001, 0.0001])
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=f"{os.path.dirname(__file__)}/Coordinate_System.stl",
                                                  meshScale=[0.0001, 0.0001, 0.0001])
        self.part_ids["base"] = p.createMultiBody(basePosition = spawn_point,
                                                  baseOrientation = spawn_orient,
                                                  baseMass=0.1,
                                                  baseCollisionShapeIndex=collisionShapeId,
                                                  baseVisualShapeIndex=visualShapeId
                                                  )
        for part in self.parts:
            # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
            self.part_offset_vectors[part.name] = [0.0,0.0,0.0]
            self.part_offset_orientations[part.name] = p.getQuaternionFromEuler([0.0,0.0,0.0])

            is_screw = part.name.startswith("screw")
            part_scale = (screw_scale_factor if (is_screw and screw_scale_factor is not None)
                          else scale_factor)
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=part.mesh_path,
                                                rgbaColor=[0.6, 0.6, 0.6, 1],
                                                meshScale=[part_scale, part_scale, part_scale])
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                      fileName=part.mesh_path,
                                                      meshScale=[part_scale, part_scale, part_scale])

            self.part_ids[part.name] = p.createMultiBody(   basePosition = spawn_point,
                                                            baseOrientation = spawn_orient,
                                                            baseMass = 0.1,
                                                            baseCollisionShapeIndex = collisionShapeId,
                                                            baseVisualShapeIndex = visualShapeId)
            """self.part_ids[part.name] = p.createMultiBody(   basePosition = spawn_point,
                                                            baseOrientation = spawn_orient,
                                                            linkMasses = [0.1],
                                                            linkPositions=[[0,0,0]],
                                                            linkOrientations=[p.getQuaternionFromEuler([0,0,0])],
                                                            linkInertialFramePositions=[part.center_of_mass*scale_factor],
                                                            linkInertialFrameOrientations=[[1,0,0,0]],
                                                            linkCollisionShapeIndices = [collisionShapeId],
                                                            linkVisualShapeIndices = [visualShapeId],
                                                            linkParentIndices = [0],
                                                            linkJointTypes = [p.JOINT_FIXED],
                                                            linkJointAxis = [[0,0,1]])"""

        # Generate all unique pairs of keys
        n_pairs = 0
        for key1, key2 in itertools.combinations(self.part_ids, 2):
            p.setCollisionFilterPair(self.part_ids[key1], self.part_ids[key2], -1, -1, False)
            n_pairs += 1
        print(f"Collision filters disabled for {n_pairs} part pairs.", flush=True)

        """self.constraint_ids[self.parts[0].name] = p.createConstraint(self.part_ids[self.parts[0].name],-1,
                                                                     -1, -1,
                                                                     p.JOINT_FIXED,
                                                                     [0, 0, 0], [0, 0, 0], spawn_point,
                                                                     None,
                                                                     p.getQuaternionFromEuler(spawn_orient))"""

        self.constraint_ids["base"] = p.createConstraint(self.part_ids["base"], -1,
                                                         -1, -1,
                                                         p.JOINT_FIXED,
                                                         [0, 0, 0],
                                                         parentFramePosition=[0, 0, 0],
                                                         childFramePosition=spawn_point,
                                                         parentFrameOrientation=[0, 0, 0, 1],
                                                         childFrameOrientation=spawn_orient)

        for i in range(0, len(self.parts)):
            inv_pos, inv_orn = p.invertTransform(self.part_offset_vectors[self.parts[i].name],self.part_offset_orientations[self.parts[i].name])
            self.constraint_ids[self.parts[i].name] = p.createConstraint(self.part_ids["base"], -1,
                                                                         self.part_ids[self.parts[i].name], -1,
                                                                         p.JOINT_FIXED,
                                                                         [0, 0, 0],
                                                                         parentFramePosition=[0, 0, 0],
                                                                         childFramePosition=inv_pos,
                                                                         parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                                                         childFrameOrientation=inv_orn)

        print("done")

    def reposition_base(self, pos=None, orn=None):
        if pos is None:
            pos = self.spawn_point
        if orn is None:
            orn = self.spawn_orient
        p.removeConstraint(self.constraint_ids["base"])
        p.resetBasePositionAndOrientation(self.part_ids["base"], pos, orn)
        self.constraint_ids["base"] = p.createConstraint(self.part_ids["base"],-1,
                                                                     -1, -1,
                                                                     p.JOINT_FIXED,
                                                                     [0, 0, 0],
                                                                     parentFramePosition=[0, 0, 0],
                                                                     childFramePosition=pos,
                                                                     parentFrameOrientation=[0,0,0,1],
                                                                     childFrameOrientation=orn)
        self.spawn_point = pos
        self.spawn_orient = orn
    def offset_part_relative(self, part_name, pos=None, orn=None):
        if pos is None:
            pos = self.part_offset_vectors[part_name]
        if orn is None:
            orn = self.part_offset_orientations[part_name]
        """inv_pos_global, inv_orn_global = p.invertTransform(self.spawn_point,self.spawn_orient)
        pos, orn = p.multiplyTransforms(np.array([0,0,0]), inv_orn_global, pos, orn)"""
        inv_pos, inv_orn = p.invertTransform(pos, orn)
        p.removeConstraint(self.constraint_ids[part_name])
        self.constraint_ids[part_name] = p.createConstraint( self.part_ids["base"], -1,
                                                             self.part_ids[part_name], -1,
                                                             p.JOINT_FIXED,
                                                             [0, 0, 0],
                                                             parentFramePosition=[0, 0, 0],
                                                             childFramePosition=inv_pos,
                                                             parentFrameOrientation=p.getQuaternionFromEuler(
                                                                 [0, 0, 0]),
                                                             childFrameOrientation=inv_orn)
        self.part_offset_vectors[part_name] = pos
        self.part_offset_orientations[part_name] = orn
    def offset_part(self, part_name, pos, orn):
        r_matrix = R.from_quat(orn)
        T_G = np.eye(4)
        T_G[:3,:3] = r_matrix.as_matrix()
        T_G[:3,3] = pos
        r_matrix_global = R.from_quat(self.spawn_orient)
        T_GL = np.eye(4)
        T_GL[:3,:3] = r_matrix_global.as_matrix()
        T_GL[:3,3] = self.spawn_point
        T_LG = np.linalg.inv(T_GL)
        T_L = np.dot(T_LG, np.dot(T_G, T_GL))
        pos_local = T_L[:3,3]
        orn_local = R.from_matrix(T_L[:3,:3]).as_quat()
        pos_local, orn_local = p.multiplyTransforms(pos_local, orn_local, self.part_offset_vectors[part_name], self.part_offset_orientations[part_name])
        inv_pos, inv_orn = p.invertTransform(pos_local, orn_local)
        p.removeConstraint(self.constraint_ids[part_name])
        self.constraint_ids[part_name] = p.createConstraint( self.part_ids["base"], -1,
                                                             self.part_ids[part_name], -1,
                                                             p.JOINT_FIXED,
                                                             [0, 0, 0],
                                                             parentFramePosition=[0, 0, 0],
                                                             childFramePosition=inv_pos,
                                                             parentFrameOrientation=p.getQuaternionFromEuler(
                                                                 [0, 0, 0]),
                                                             childFrameOrientation=inv_orn)
        self.part_offset_vectors[part_name] = pos_local
        self.part_offset_orientations[part_name] = orn_local

    def fix_in_place(self):
        pos, orn = p.getBasePositionAndOrientation(self.part_ids["base"])
        self.reposition_base(pos, orn)

        """if self.constraint_ids["base"] is not None:
            p.removeConstraint(self.constraint_ids["base"])
        self.constraint_ids["base"] = p.createConstraint(self.part_ids["base"], -1,
                                                         -1, -1,
                                                         p.JOINT_FIXED,
                                                         [0, 0, 0],
                                                         parentFramePosition=[0, 0, 0],
                                                         childFramePosition=pos,
                                                         parentFrameOrientation=[0, 0, 0, 1],
                                                         childFrameOrientation=orn)
        self.spawn_point = pos
        self.spawn_orient = orn"""
    def __del__(self):
        for part in self.parts:
            if part.name in self.part_ids:
                p.removeBody(self.part_ids[part.name])
        if "base" in self.part_ids:
            p.removeBody(self.part_ids["base"])
