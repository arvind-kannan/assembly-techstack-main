import os
import sys
import shutil
import glob
import trimesh
import itertools
import math
import time
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import socket
import datetime
from collections import Counter


from PyQt5.QtWidgets import QApplication, QFileDialog
import fnmatch
from opcua import Client
from opcua import ua

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, simpledialog
import tkinter.font as tkfont

from PIL import Image, ImageTk
import io
import copy
import subprocess

import open3d as o3d
#from sklearn.decomposition import PCA
#import heapq

from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodule_crc_diss_assembly_dataclasses', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GraphMDPAnalyser_main')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GraphMDPAnalyser_main', 'GraphMDPAnalyser')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dasg_gym', 'envs')))

from diss_assembly_dataclasses.connection import Connection, Geometric_Connection, Screw_Connection, Safety_Ring_Connection
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part, load_parts_from_folder

def _unavailable(name, err):
    # placeholder for an optional dependency: the HMI starts, the feature raises when used
    def _raise(*args, **kwargs):
        raise ImportError(f"'{name}' is not available in this environment: {err}") from err
    return _raise

#import graphMDPconverter as mdpconverter
# GraphMDPAnalyser_main and dasg_gym are separate projects, expected next to this repo
for _mod, _name in [("mdpAnalyser", "graph_zu_markov_graph"), ("mdpAnalyser", "dasg_graph_einlesen"),
                    ("mdpAnalyser", "save_graph_as_pkl"), ("graph_visualizer", "show_in_dash")]:
    try:
        globals()[_name] = getattr(__import__(_mod), _name)
    except (ImportError, AttributeError) as e:
        globals()[_name] = _unavailable(_name, e)
try:
    from dasg_gym.envs.Agent_Q_learner import train_q_agent
    from dasg_gym.envs.Agent_Q_testing import get_best_action
except ImportError as e:
    train_q_agent = _unavailable("train_q_agent", e)
    get_best_action = _unavailable("get_best_action", e)

from operations import GeneralGripping
from planning import HierarchicalDisassemblySequencePlanner

from utility import create_screw_dummy, generate_orthogonal_vectors, rotation_matrix_from_zx, rotation_matrix_from_zy, \
                    rotation_matrix_to_align_z, fit_minimum_volume_cylinder_along_axis


# the camera needs the Zivid SDK, optional for simulation-only use
try:
    from Zivid import ZividCamera
except ImportError as e:
    ZividCamera = _unavailable("ZividCamera", e)
from Registration import calculate_registration
from perceptionModule import PerceptionModule

import pyrender
from pyrender import RenderFlags

import configparser
import xml.etree.ElementTree as ET

import json

from colors import wbk_color_dict

# to install pygraphviz install graphviz and then run IN THE VIRTUAL ENVIRONMENT!!!
# pip install --use-pep517 --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include"' ---config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz

"""try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                              "PyGraphviz or PyDotPlus")

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'"""

from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
script_dir_path = Path(__file__).parent.resolve()
base_path = script_dir_path.parent
_SELF_DIR_NAME = script_dir_path.name
def resolve_path(relative_path):
    """Resolve a path to an absolute path.
    Paths that contain this folder's name (e.g. machine specific absolute paths in config.ini)
    are re-based onto the actual location of this folder."""
    parts = Path(relative_path).parts
    if _SELF_DIR_NAME in parts:
        i = len(parts) - 1 - parts[::-1].index(_SELF_DIR_NAME)
        return script_dir_path.joinpath(*parts[i + 1:]).resolve()
    return (base_path / relative_path).resolve()

def assembly_to_trimesh_scene(assembly):
    """trimesh scene of an assembly (replaces the removed Assembly.scene(); parts hold open3d meshes now).
    First / last part are colored like before."""
    parts = list(assembly.parts.values())
    geoms = []
    for i, part in enumerate(parts):
        m = part.mesh
        tm = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles), process=False)
        if i == 0:
            tm.visual.vertex_colors = [0, 150, 130, 255]
        elif i == len(parts) - 1:
            tm.visual.vertex_colors = [20, 68, 102, 255]
        geoms.append(tm)
    return trimesh.Scene(geoms)

def o3d_geoms_to_trimesh_scene(geoms):
    """trimesh scene from open3d geometries (TriangleMesh and LineSet; other types are skipped)."""
    out = []
    for g in geoms:
        if isinstance(g, o3d.geometry.TriangleMesh):
            tm = trimesh.Trimesh(vertices=np.asarray(g.vertices), faces=np.asarray(g.triangles), process=False)
            if g.has_vertex_colors():
                tm.visual.vertex_colors = (np.asarray(g.vertex_colors) * 255).astype(np.uint8)
            out.append(tm)
        elif isinstance(g, o3d.geometry.LineSet):
            pts = np.asarray(g.points)
            for a, b in np.asarray(g.lines):
                if np.linalg.norm(pts[b] - pts[a]) > 1e-9:
                    seg = trimesh.creation.cylinder(radius=0.5, segment=[pts[a], pts[b]])
                    seg.visual.vertex_colors = [255, 0, 0, 255]
                    out.append(seg)
    return trimesh.Scene(out)

def operation_scene(operation, assembly):
    """trimesh scene for an operation; uses the open3d scene_o3d() and falls back to the legacy scene()."""
    try:
        return o3d_geoms_to_trimesh_scene(operation.scene_o3d(assembly))
    except NotImplementedError:
        return operation.scene(assembly)

def create_assembly_object(parts, assembly_name = "example_assembly", contact_tollerance=1.0, dot_product_threshold = -0.99,):
    asm = Assembly(assembly_name, parts,
                   contact_tollerance=contact_tollerance,
                   dot_product_threshold=dot_product_threshold)
    asm.find_contacts()
    return asm

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

# Tkinter GUI setup
class BaseDialog(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.create_widgets()


    def center_window(self):
        self.update_idletasks()  # Update the internal layout to get accurate dimensions
        width = self.winfo_width()
        height = self.winfo_height()

        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()

        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry(f'+{int(x)}+{int(y)}')

    def create_widgets(self):
        raise NotImplementedError("Create widgets method must be implemented in subclass")

    def create_entry(self, label_text, default_value):
        label = tk.Label(self, text=label_text)
        label.pack()
        entry = tk.Entry(self)
        entry.insert(0, default_value)
        entry.pack()
        setattr(self, label_text.replace(" ", "_").lower() + "_entry", entry)
        return entry

    def submit(self):
        raise NotImplementedError("Submit method must be implemented in subclass")
class LoadPartsDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Load Parts")

    def create_widgets(self):
        self.name_entry = self.create_entry("Assembly name:", "example_asm")
        self.depth_entry = self.create_entry("Depth:", "1")
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        name = self.name_entry.get()
        try:
            depth = int(self.depth_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid depth. Please enter an integer.")
            return
        self.parent.load_parts(name, depth)
        self.destroy()

class PlannerDialog(BaseDialog):
    def __init__(self, parent):
        self.milling_var = tk.BooleanVar(value=False)
        self.screwing_var = tk.BooleanVar(value=True)
        self.gripping_var = tk.BooleanVar(value=True)
        self.safety_ring_var = tk.BooleanVar(value=False)
        super().__init__(parent, "Create Planner")

    def create_widgets(self):
        # Create checkboxes for operation types
        tk.Label(self, text="Select Operations:").pack(anchor='w', padx=10, pady=(10, 5))
        
        milling_chk = tk.Checkbutton(self, text="Milling", variable=self.milling_var)
        milling_chk.pack(anchor='w', padx=20, pady=2)
        
        screwing_chk = tk.Checkbutton(self, text="Screwing", variable=self.screwing_var)
        screwing_chk.pack(anchor='w', padx=20, pady=2)
        
        gripping_chk = tk.Checkbutton(self, text="Gripping", variable=self.gripping_var)
        gripping_chk.pack(anchor='w', padx=20, pady=2)
        
        safety_ring_chk = tk.Checkbutton(self, text="Safety Ring", variable=self.safety_ring_var)
        safety_ring_chk.pack(anchor='w', padx=20, pady=2)
        
        # Keep numeric entry for mill diameter
        self.create_entry("Mill Diameter", "6.0")

        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)
        self.center_window()

    def submit(self):
        try:
            # Convert checkbox values to costs (-1 if unchecked, 0 if checked)
            cost_milling = 0 if self.milling_var.get() else -1
            cost_screwing = 0 if self.screwing_var.get() else -1
            cost_gripping = 0 if self.gripping_var.get() else -1
            cost_safety_ring = 0 if self.safety_ring_var.get() else -1
            
            # Get mill diameter from entry field
            mill_diameter = float(self.mill_diameter_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid mill diameter. Please enter a valid number.")
            return
            
        # Save settings to JSON
        data = {
            "gripping": {"cost": cost_gripping}, 
            "screwing": {"cost": cost_screwing}, 
            "milling": {"cost": cost_milling}, 
            "mill_diameter": {"diameter": mill_diameter}, 
            "safety_ring": {"cost": cost_safety_ring}
        }
        with open("planner_settings.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
            
        self.parent.create_planner(cost_milling, cost_screwing, cost_gripping, cost_safety_ring,
                                   mill_diameter, self.parent.data_directory)
        file_path = os.path.join(self.parent.operations_folder, self.parent.planner.assembly_states[0].name + "_gripping_operations.pkl")
        if os.path.isfile(file_path):
            self.parent.add_output_line(f"Predefined gripping operations from {file_path} loaded.")
            self.parent.planner.import_gripping_operations(file_path)
        self.destroy()

class ViewParametersDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Set View Parameters")

    def create_widgets(self):
        # Add entry fields for angle and distance with default values
        self.create_entry("Angle X",self.parent.view_angle_X)  # Default angle
        self.create_entry("Angle Y",self.parent.view_angle_Y)  # Default angle
        self.create_entry("Angle Z",self.parent.view_angle_Z)  # Default angle
        self.create_entry("Distance", self.parent.view_distance)  # Default distance

        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        try:
            angle_x = float(self.angle_x_entry.get())
            angle_y = float(self.angle_y_entry.get())
            angle_z = float(self.angle_z_entry.get())
            distance = float(self.distance_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
            return

        self.parent.set_view_parameters(angle_x, angle_y, angle_z, distance)
        self.destroy()

class PlanSequenceDialog(BaseDialog):
    def __init__(self, parent):
        self.parts_list = []  # List of possible parts to be disassembled
        for part in parent.assembly.parts.values():
            self.parts_list.append(part.name)
        
        with open("parts_in_asm_raw.json", "w") as json_file:
            json.dump(self.parts_list, json_file, indent=4)
        print(f"parts_in_asm_raw saved to json file: {self.parts_list}")

        self.checked_parts = []  # To store selected parts
        super().__init__(parent, "Plan Sequence to Target")

    def create_widgets(self):
        self.label_part_select = tk.Label(self, text="Select parts to be disassembled")
        self.label_part_select.pack()
        # Create a checkbox for each part
        """for part in self.parts_list:
            chk_var = tk.BooleanVar()
            chk_box = tk.Checkbutton(self, text=part, variable=chk_var, font=self.parent.medium_font)
            chk_box.pack()
            self.checked_parts.append((chk_var, part))"""

        # Create frames for checkbox columns
        frames = []
        columns = (len(self.parts_list) - 1) // 32 + 1  # Calculate the number of columns needed
        for i in range(columns):
            frame = tk.Frame(self)
            frame.pack(side='left', fill='both', expand=True)
            frames.append(frame)

        # Create a checkbox for each part in the appropriate column
        for index, part in enumerate(self.parts_list):
            frame = frames[index // 32]  # Determine which column/frame to place the checkbox
            chk_var = tk.BooleanVar()
            chk_box = tk.Checkbutton(frame, text=part, variable=chk_var, font=self.parent.medium_font if self.parent else 'TkDefaultFont')
            chk_box.pack()
            self.checked_parts.append((chk_var, part))

        # Create select all and unselect all buttons
        self.select_all_button = tk.Button(self, text="Select All", command=self.select_all)
        self.select_all_button.pack()

        self.unselect_all_button = tk.Button(self, text="Unselect All", command=self.unselect_all)
        self.unselect_all_button.pack()

        # Additional widgets
        self.create_entry("Max iterations", "100")
        self.chk_var_save = tk.BooleanVar()
        self.chk_box_save = tk.Checkbutton(self, text="log results", variable=self.chk_var_save)
        self.chk_box_save.pack()
        #self.create_entry("Result path", "")
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def select_all(self):
        # Set all checkbox variables to True
        for chk_var, part in self.checked_parts:
            chk_var.set(True)

    def unselect_all(self):
        # Set all checkbox variables to False
        for chk_var, part in self.checked_parts:
            chk_var.set(False)

    def submit(self):
        #part_names_list = [part.strip() for part in self.parts_to_be_disassembled_entry.get().split(',')]
        self.parent.disassembly_target_part_names = [part for chk_var, part in self.checked_parts if chk_var.get()]
        print(self.parent.disassembly_target_part_names, len(self.parent.disassembly_target_part_names))
        part_list = []
        for part_name in self.parent.disassembly_target_part_names:
            part_list.append(self.parent.assembly.get_part_by_name(part_name))
        try:
            max_iter = int(self.max_iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid max iterations. Please enter an integer.")
            return
        if self.chk_var_save.get():
            # Set the initial directory to the current file's directory
            #initial_dir = os.path.dirname(os.path.abspath(__file__))
            initial_dir = os.path.dirname(os.path.abspath(__file__))
            # Open file dialog with initial directory set to the current file's directory
            result_path = filedialog.askdirectory(initialdir=initial_dir)
            #self.result_path_entry.get()
        else:
            result_path = None
        if all(element is not None for element in part_list):
            self.destroy()
            self.parent.add_output_line(f"Starting planning process")
            start = time.time()
            self.parent.plan_sequence_to_target(part_list, max_iter, result_path)
            end = time.time()
            #self.parent.planner.export_states(os.path.join(self.parent.data_directory, "Working_dir")) #exports assembly_states to Working_dir
            messagebox.showinfo("Success", f"Planning process finished in {end - start} seconds")
            self.parent.add_output_line(f"Planning process finished in {end - start} seconds")
        else:
            messagebox.showerror("Error", "Please enter only valid part names")
            self.destroy()
class PlanMultipleSequenceDialog(PlanSequenceDialog):
    def __init__(self, parent):
        super().__init__(parent)

    def submit(self):
        self.parent.disassembly_target_part_names = [part for chk_var, part in self.checked_parts if chk_var.get()]
        print(self.parent.disassembly_target_part_names, len(self.parent.disassembly_target_part_names))
        part_list = []
        for part_name in self.parent.disassembly_target_part_names:
            part_list.append(self.parent.assembly.get_part_by_name(part_name))
        try:
            max_iter = int(self.max_iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid max iterations. Please enter an integer.")
            return
        if all(element is not None for element in part_list):
            self.destroy()
            start = time.time()
            self.parent.planner.explore_all_nodes_with_parts(part_list, max_iter)
            end = time.time()
            sequences, end_states = self.parent.planner.find_all_sequences_to_target(part_list)
            #print(result)
            print(f'In {end - start} found {len(sequences)} possible sequences for {len(end_states)} end states:\n {end_states}')

            # Count occurrences of each node in the valid paths
            node_occurrences = Counter(node for seq in sequences for node in seq)

            # Get the 30 most common nodes
            most_common_nodes = node_occurrences.most_common(30)

            # Print the 30 most common nodes and their counts
            for node, count in most_common_nodes:
                print(f"Node {node} appears in {count} paths.")

            for state in end_states:
                print_string = f"{state}:"
                for part in self.parent.planner.assembly_states[state].parts.values():
                    print_string = f"{print_string} {part.name}"
                print(print_string)
                #scene = self.parent.planner.assembly_states[state].scene()
                #scene.show()
            messagebox.showinfo("Success", f"Planning process finished in {end - start} seconds")
            self.parent.add_output_line(f"Planning process finished in {end - start} seconds")
        else:
            messagebox.showerror("Error", "Please enter only valid part names")
            self.destroy()

class PlanGraphDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Set View Parameters")

    def create_widgets(self):
        self.create_entry("Max iterations", "500")
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        try:
            max_iter = int(self.max_iterations_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid max iterations. Please enter an integer.")
            return
        self.parent.planner.generate_full_graph(max_iter)
        #self.parent.planner.export_states(os.path.join(self.parent.data_directory, "Working_dir")) #exports assembly_states to Working_dir
        self.parent.show_graph()
        messagebox.showinfo("Success", f"Planning process finished")
        self.destroy()
class GetSequenceToStateDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Show State")

    def create_widgets(self):
        self.create_entry("State number", "")
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        state = int(self.state_number_entry.get())
        self.parent.path = copy.deepcopy(self.parent.planner.graph.nodes[state]['cheapest_sequence_to_node'])
        self.parent.path.append(state)
        self.parent.show_graph()
        self.parent.update_sequence_lable()
        self.destroy()

class ShowStateDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Get Sequence to State")

    def create_widgets(self):
        self.create_entry("State number", "")
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        state = int(self.state_number_entry.get())
        if self.parent.path is None:
            self.parent.currently_showing["index"]=state
        self.parent.show_state(state)
        self.destroy()

class ShowActionDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Show State")

    def create_widgets(self):
        self.create_entry("From state", "")
        self.create_entry("to state", "")
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        from_state = int(self.from_state_entry.get())
        to_state = int(self.to_state_entry.get())
        self.parent.show_action(from_state, to_state)
        self.destroy()

class AddRelationDialog(BaseDialog):
    def __init__(self, parent):
        self.parent = parent
        self.parts_list = []  # List of possible parts to be selected
        for part in parent.assembly.parts.values():
            self.parts_list.append(part.name)
        self.selections = []  # To store the selected values from dropdowns
        self.relation_options = ['screw', 'safety_ring']
        self.option_var1 = tk.StringVar(self.parent)
        self.option_var2 = tk.StringVar(self.parent)
        self.option_var3 = tk.StringVar(self.parent)
        self.textField4  = None  # Variable for the optional fourth dropdown
        self.label4= None
        super().__init__(parent, "Select Parts")


    def create_widgets(self):
        self.dropdown_frames = tk.Frame(self)
        self.dropdown_frames.pack(fill=tk.BOTH, expand=True, pady=10)

        # Dropdown Menu 1
        tk.Label(self.dropdown_frames, text="Relation Type:").grid(row=0, column=0, padx=10, sticky=tk.W)
        self.dropdown1 = tk.OptionMenu(self.dropdown_frames, self.option_var1, *self.relation_options)
        self.dropdown1.grid(row=1, column=0, padx=10, sticky=tk.EW)
        self.option_var1.trace("w", self.update_dropdowns)  # Add trace to option_var1

        # Dropdown Menu 2
        tk.Label(self.dropdown_frames, text="Part 1:").grid(row=0, column=1, padx=10, sticky=tk.W)
        self.option_var2 = tk.StringVar(self)
        self.dropdown2 = tk.OptionMenu(self.dropdown_frames, self.option_var2, *self.parts_list)
        self.dropdown2.grid(row=1, column=1, padx=10, sticky=tk.EW)

        # Dropdown Menu 3
        tk.Label(self.dropdown_frames, text="Part 2:").grid(row=0, column=2, padx=10, sticky=tk.W)
        self.option_var3 = tk.StringVar(self)
        self.dropdown3 = tk.OptionMenu(self.dropdown_frames, self.option_var3, *self.parts_list)
        self.dropdown3.grid(row=1, column=2, padx=10, sticky=tk.EW)

        # Adjust column configuration for equal width distribution
        self.dropdown_frames.grid_columnconfigure(1, weight=1)

        # Creating the submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, pady=10)
        self.center_window()

    def update_dropdowns(self, *args):
        selected_value = self.option_var1.get()
        if selected_value == "screw" and not self.textField4:  # Check for a specific option and if textField4 does not exist
            self.textField4 = tk.Entry(self.dropdown_frames)
            self.textField4.grid(row=1, column=3, padx=10, sticky=tk.EW)
            self.label4 = tk.Label(self.dropdown_frames, text="Screw Type:")
            self.label4.grid(row=0, column=3, padx=10, sticky=tk.W)
        elif selected_value != "Specific Option" and self.textField4:
            self.textField4.destroy()  # Remove dropdown4 if it exists and the specific option is not selected
            self.label4.destroy()  # Remove dropdown4 if it exists and the specific option is not selected
            self.textField4 = None
            self.label4 = None


    def submit(self):
        self.selections = [self.option_var1.get(), self.option_var2.get(), self.option_var3.get()]
        if self.option_var1.get() == "screw":
            self.parent.assembly.add_connection(Screw_Connection(part_name1= self.option_var2.get(), part_name2= self.option_var3.get(), screw_head= self.textField4.get()))
        elif self.option_var1.get() == "safety_ring":
            self.parent.assembly.add_connection(Safety_Ring_Connection(part_name1= self.option_var2.get(),
                                                                       part_name2= self.option_var3.get()))

        print("Selected Parts: ", self.selections)
        self.destroy()

class SelectPartDialog(BaseDialog):
    def __init__(self, parent, callback_function=None):
        self.parent = parent
        self.callback_function = callback_function  # Function to call with selected part
        self.parts_list = [part.name for part in parent.assembly.parts]
        self.selected_part = tk.StringVar(self.parent)
        self.result = None  # Will hold the result if no callback is given
        super().__init__(parent, "Select a Part")

    def create_widgets(self):
        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Label and Dropdown Menu
        tk.Label(self.dropdown_frame, text="Choose a Part:").grid(row=0, column=0, padx=10, sticky=tk.W)
        self.dropdown = tk.OptionMenu(self.dropdown_frame, self.selected_part, *self.parts_list)
        self.dropdown.grid(row=1, column=0, padx=10, sticky=tk.EW)

        self.dropdown_frame.grid_columnconfigure(0, weight=1)

        # Submit Button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, pady=10)
        self.center_window()

    def submit(self):
        part_name = self.selected_part.get()
        if part_name:
            if self.callback_function:
                self.callback_function(part_name)
            else:
                self.result = part_name  # Store result
        print("Selected Part:", part_name)
        self.destroy()

    def show(self):
        """Shows the dialog and waits if no callback is provided."""
        self.wait_window()
        return self.result if self.callback_function is None else None
class SelectMountingConfigurationDialog(BaseDialog):
    def __init__(self, parent, callback_function=None):
        self.parent = parent
        self.callback_function = callback_function  # Function to call with selected config name
        self.config_keys = list(self.parent.mounting_configurations.keys())
        self.selected_key = tk.StringVar(self.parent)
        self.result = None  # Store result if no callback is given

        # Checkbox states
        self.open_fixture_var = tk.BooleanVar()
        self.close_fixture_var = tk.BooleanVar()

        super().__init__(parent, "Select Mounting Configuration")

    def create_widgets(self):
        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Label and Dropdown Menu
        tk.Label(self.dropdown_frame, text="Choose a Configuration:").grid(row=0, column=0, padx=10, sticky=tk.W)
        self.dropdown = tk.OptionMenu(self.dropdown_frame, self.selected_key, *self.config_keys)
        self.dropdown.grid(row=1, column=0, padx=10, sticky=tk.EW)

        self.dropdown_frame.grid_columnconfigure(0, weight=1)

        # Checkboxes
        self.checkbox1 = tk.Checkbutton(self, text="Open Fixture", variable=self.open_fixture_var)
        self.checkbox1.pack(anchor=tk.W, padx=10)
        self.checkbox2 = tk.Checkbutton(self, text="Close Fixture", variable=self.close_fixture_var)
        self.checkbox2.pack(anchor=tk.W, padx=10)

        # Submit Button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, pady=10)
        self.center_window()

    def submit(self):
        selected_name = self.selected_key.get()
        open_fixture_enabled = self.open_fixture_var.get()
        close_fixture_enabled = self.close_fixture_var.get()

        if selected_name:
            if self.callback_function:
                # Optionally pass checkbox states too
                self.callback_function(selected_name, open_fixture_enabled, close_fixture_enabled)
            else:
                self.result = (selected_name, open_fixture_enabled, close_fixture_enabled)

        print("Selected Configuration:", selected_name)
        print("Open Fixture Enabled:", open_fixture_enabled)
        print("Close Fixture Enabled:", close_fixture_enabled)
        self.destroy()

    def show(self):
        """Shows the dialog and waits if no callback is provided."""
        self.wait_window()
        return self.result if self.callback_function is None else None
class SelectPartToLookAtDialog(BaseDialog):
    def __init__(self, parent):
        self.parent = parent
        self.parts_list = []  # List of possible parts to be selected
        for part in parent.assembly.parts.values():
            self.parts_list.append(part.name)
        self.option_var1 = tk.StringVar(self.parent)
        self.option_var1.set(self.parts_list[0])
        super().__init__(parent, "Select Parts")


    def create_widgets(self):
        self.dropdown_frames = tk.Frame(self)
        self.dropdown_frames.pack(fill=tk.BOTH, expand=True, pady=10)

        # Dropdown Menu 1
        tk.Label(self.dropdown_frames, text="Select Part").grid(row=0, column=0, padx=10, sticky=tk.W)
        self.dropdown1 = tk.OptionMenu(self.dropdown_frames, self.option_var1, *self.parts_list)
        self.dropdown1.grid(row=1, column=0, padx=10, sticky=tk.EW)

        # Adjust column configuration for equal width distribution
        self.dropdown_frames.grid_columnconfigure(1, weight=1)

        self.distance_entry = self.create_entry("Distance [m]", "0.5")
        self.theta_entry = self.create_entry("Theta [°]:", "30.0")
        self.theta_delta_entry = self.create_entry("delta theta [°]:", "18.0")
        self.theta_number_entry = self.create_entry("num theta:", "1")
        self.phi_entry = self.create_entry("Phi [°]:", "-90.0")
        self.phi_delta_entry = self.create_entry("delta phi [°]:", "30.0")
        self.phi_number_entry = self.create_entry("num phi:", "1")

        # Creating the submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, pady=10)

        self.center_window()

    def submit(self):
        #self.option_var1.get()
        distance = float(self.distance_entry.get())
        theta = np.radians(float(self.theta_entry.get()))
        theta_delta = np.radians(float(self.theta_delta_entry.get()))
        theta_number = int(self.theta_number_entry.get())
        phi = np.radians(float(self.phi_entry.get()))
        phi_delta = np.radians(float(self.phi_delta_entry.get()))
        phi_number = int(self.phi_number_entry.get())

        for t in range(theta_number):
            for p in range(phi_number):
                if (t % 2) == 0:
                    self.parent.look_at_part(self.option_var1.get(), distance, theta + t*theta_delta,
                                             phi + p*phi_delta)
                    print(self.option_var1.get(), distance, theta + t*theta_delta,
                                             phi + p*phi_delta)
                else:
                    self.parent.look_at_part(self.option_var1.get(), distance, theta + t*theta_delta,
                                             phi + (phi_number-1-p)*phi_delta)
                    print(self.option_var1.get(), distance, theta + t*theta_delta,
                                             phi + (phi_number-1-p)*phi_delta)

                if theta + t*theta_delta <=0.00001:
                    break

        #self.parent.look_at_part(self.option_var1.get(), distance, theta, phi)
        print("finished looking")
        self.destroy()
class ShowConnectionDialog(BaseDialog):
    def __init__(self, parent):
        self.parent = parent
        self.connection_list = []  # List of possible parts to be selected
        self.gc_to_recheck = [conn for conn in parent.assembly.connections if conn.connection_type == 'geometric']
        self.connection_list = [str(conn) for conn in self.gc_to_recheck]  # List of possible parts to be selected
        self.option_var1 = tk.StringVar(self.parent)
        if len(self.connection_list)>0:
            self.option_var1.set(self.connection_list[0])
        else:
            return
        super().__init__(parent, "Select Connection")

    def create_widgets(self):
        self.dropdown_frames = tk.Frame(self)
        self.dropdown_frames.pack(fill=tk.BOTH, expand=True, pady=10)

        # Dropdown Menu 1
        tk.Label(self.dropdown_frames, text="Select Connection").grid(row=0, column=0, padx=10, sticky=tk.W)
        self.dropdown1 = tk.OptionMenu(self.dropdown_frames, self.option_var1, *self.connection_list)
        self.dropdown1.grid(row=1, column=0, padx=10, sticky=tk.EW)

        # Adjust column configuration for equal width distribution
        self.dropdown_frames.grid_columnconfigure(1, weight=1)

        # Creating the submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(side=tk.BOTTOM, pady=10)

        self.center_window()

    def submit(self):
        connection_to_show = self.gc_to_recheck[self.connection_list.index(self.option_var1.get())]
        self.parent.assembly.show_contact_points(connection_to_show)
        self.destroy()


class ManualRepositionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.current_pos, self.current_orient = self.parent.request_spawn_pose()

        # Use preloaded configurations
        self.configurations = self.parent.mounting_configurations
        self.selected_config = tk.StringVar(self)

        self.title("Manual Reposition")
        self.create_widgets()

    def create_widgets(self):
        if self.current_pos is None or self.current_orient is None:
            messagebox.showerror("Error", "Failed to retrieve current spawn position.")
            self.destroy()
            return

        self.move_step = 0.1
        self.rot_step = 5
        self.entries = {}

        # Dropdown for mounting configurations
        if self.configurations:
            tk.Label(self, text="Load from saved configuration:").pack()
            options = list(self.configurations.keys())
            self.selected_config.set(options[0])
            dropdown = tk.OptionMenu(self, self.selected_config, *options, command=self.apply_configuration)
            dropdown.pack(pady=5)

        self.create_position_controls()
        self.create_orientation_controls()

        self.submit_button = tk.Button(self, text="Reposition", command=self.submit)
        self.submit_button.pack(pady=10)
        self.center_window()

    def apply_configuration(self, name):
        """Fill fields with selected configuration values."""
        pos = self.configurations[name]['position']
        orn = self.configurations[name]['orientation']  # Should be Euler angles in degrees

        # Update entries
        for i, val in enumerate(pos):
            self.entries[(i, False)].delete(0, tk.END)
            self.entries[(i, False)].insert(0, str(val))

        for i, val in enumerate(orn):
            self.entries[(i, True)].delete(0, tk.END)
            self.entries[(i, True)].insert(0, str(val))

        # Update internal pose and simulate
        self.current_pos = np.array(pos)
        self.current_orient = R.from_euler('xyz', orn, degrees=True).as_quat()
        self.parent.reposition_assembly_in_sim(self.current_pos, self.current_orient)

    def create_position_controls(self):
        tk.Label(self, text="Position").pack()
        self.create_adjustable_entry("X:", 0, self.current_pos[0])
        self.create_adjustable_entry("Y:", 1, self.current_pos[1])
        self.create_adjustable_entry("Z:", 2, self.current_pos[2])

    def create_orientation_controls(self):
        tk.Label(self, text="Orientation").pack()
        euler_orient = R.from_quat(self.current_orient).as_euler('xyz', degrees=True)
        self.create_adjustable_entry("Roll:", 0, euler_orient[0], orientation=True)
        self.create_adjustable_entry("Pitch:", 1, euler_orient[1], orientation=True)
        self.create_adjustable_entry("Yaw:", 2, euler_orient[2], orientation=True)

    def create_adjustable_entry(self, label, index, default_value, orientation=False):
        frame = tk.Frame(self)
        frame.pack(fill="x", pady=2)

        tk.Label(frame, text=label).pack(side="left")
        entry = tk.Entry(frame, width=10)
        entry.pack(side="left")
        entry.insert(0, str(default_value))
        self.entries[(index, orientation)] = entry

        tk.Button(frame, text="+",
                  command=lambda: self.adjust_value(index, self.move_step if not orientation else self.rot_step,
                                                    orientation)).pack(side="left")
        tk.Button(frame, text="-",
                  command=lambda: self.adjust_value(index, -self.move_step if not orientation else -self.rot_step,
                                                    orientation)).pack(side="left")

    def adjust_value(self, index, delta, orientation):
        key = (index, orientation)
        try:
            value = float(self.entries[key].get()) + delta
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(value))

            if orientation:
                euler_orient = R.from_quat(self.current_orient).as_euler('xyz', degrees=True)
                euler_orient[index] = value
                self.current_orient = R.from_euler('xyz', euler_orient, degrees=True).as_quat()
            else:
                self.current_pos[index] = value

            self.parent.reposition_assembly_in_sim(self.current_pos, self.current_orient)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a valid number.")

    def submit(self):
        try:
            pos = np.array([
                float(self.entries[(0, False)].get()),
                float(self.entries[(1, False)].get()),
                float(self.entries[(2, False)].get())
            ])
            roll = float(self.entries[(0, True)].get())
            pitch = float(self.entries[(1, True)].get())
            yaw = float(self.entries[(2, True)].get())

            orientation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
            return

        self.parent.reposition_assembly_in_sim(pos, orientation)
        self.destroy()

class TravelTimeDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Adjust Travel Times")

    def create_widgets(self):
        self.traveltime_from_assembly_to_dropoff_entry = self.create_entry("Traveltime from Assembly to dropoff", "15.0") #default value can be adjusted here
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        try:
            travel_time = float(self.traveltime_from_assembly_to_dropoff_entry.get())
            self.parent.add_output_line(f"Travel time set to: {travel_time} seconds")
            
            try:
                # Try to load existing settings
                with open("cell_settings.json", "r") as json_file:
                    cell_data = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a new settings file if it doesn't exist or is invalid
                cell_data = {}

            cell_data["traveltime_assembly_to_dropoff"] = travel_time

            with open("cell_settings.json", "w") as json_file:
                json.dump(cell_data, json_file, indent=4)
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a valid number.")

class ToolChangeTimeDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Adjust Tool Change Times")

    def create_widgets(self):
        self.time_from_start_to_toolstorage_entry = self.create_entry("Time from Robot Start to Tool-Storage", "10.0")  # default value can be adjusted here
        self.time_from_dropoff_to_toolstorage_entry = self.create_entry("Time from Dropoff to Tool-Storage", "10.0")  # default value can be adjusted here
        self.time_from_assembly_to_toolstorage_entry = self.create_entry("Time from Assembly to Tool-Storage", "10.0")  # default value can be adjusted here
        self.time_for_tool_change_at_toolstorage_entry = self.create_entry("Time for Tool Change at Tool-Storage", "60.0")  # default value can be adjusted here
        
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        try:
            start_to_storage_time = float(self.time_from_start_to_toolstorage_entry.get())
            dropoff_to_storage_time = float(self.time_from_dropoff_to_toolstorage_entry.get())
            assembly_to_storage_time = float(self.time_from_assembly_to_toolstorage_entry.get())
            tool_change_time = float(self.time_for_tool_change_at_toolstorage_entry.get())
            
            self.parent.add_output_line(f"Time from Robot Start to Tool-Storage: {start_to_storage_time} s")
            self.parent.add_output_line(f"Time from Dropoff to Tool-Storage: {dropoff_to_storage_time} s")
            self.parent.add_output_line(f"Time from Assembly to Tool-Storage: {assembly_to_storage_time} s")
            self.parent.add_output_line(f"Time for Tool Change at Tool-Storage: {tool_change_time} s")
            
            try:
                # Try to load existing settings
                with open("cell_settings.json", "r") as json_file:
                    cell_data = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a new settings file if it doesn't exist or is invalid
                cell_data = {}

            cell_data["toolchange_time_start_to_toolstorage"] = start_to_storage_time
            cell_data["toolchange_time_dropoff_to_storage"] = dropoff_to_storage_time
            cell_data["toolchange_time_assembly_to_storage"] = assembly_to_storage_time
            cell_data["toolchange_time_at_toolstorage"] = tool_change_time

            with open("cell_settings.json", "w") as json_file:
                json.dump(cell_data, json_file, indent=4)
            
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")

class FeedRateDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Adjust Feed Rates")

    def create_widgets(self):
        # Add all required feed rate fields with default values - store the entries directly
        self.feed_rate_milling_entry = self.create_entry("Feed rate milling (mm/min)", "1000.0")
        self.feed_rate_gripping_approaching_entry = self.create_entry("Feed rate gripping approaching Assembly (mm/min)", "2000.0")
        self.feed_rate_gripping_leaving_entry = self.create_entry("Feed rate gripping leaving (mm/min)", "1000.0")
        self.feed_rate_screwing_approaching_entry = self.create_entry("Feed rate screwing approaching (mm/min)", "2000.0")
        self.feed_rate_screwing_leaving_entry = self.create_entry("Feed rate screwing leaving (mm/min)", "1000.0")
        self.milling_penalty = self.create_entry("Milling penalty (s)", "100.0")
        
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.center_window()

    def submit(self):
        try:
            # Get values using our explicit entry references
            feed_rate_milling = float(self.feed_rate_milling_entry.get())
            feed_rate_gripping_approaching = float(self.feed_rate_gripping_approaching_entry.get())
            feed_rate_gripping_leaving = float(self.feed_rate_gripping_leaving_entry.get())
            feed_rate_screwing_approaching = float(self.feed_rate_screwing_approaching_entry.get())
            feed_rate_screwing_leaving = float(self.feed_rate_screwing_leaving_entry.get())
            milling_penalty = float(self.milling_penalty.get())
            
            # Log the values
            self.parent.add_output_line(f"Feed rate milling set to: {feed_rate_milling} mm/min")
            self.parent.add_output_line(f"Feed rate gripping approaching set to: {feed_rate_gripping_approaching} mm/min")
            self.parent.add_output_line(f"Feed rate gripping leaving set to: {feed_rate_gripping_leaving} mm/min")
            self.parent.add_output_line(f"Feed rate screwing approaching set to: {feed_rate_screwing_approaching} mm/min")
            self.parent.add_output_line(f"Feed rate screwing leaving set to: {feed_rate_screwing_leaving} mm/min")
            self.parent.add_output_line(f"Extra topping for milling set to: {milling_penalty} s")
            
            try:
                # Try to load existing settings
                with open("cell_settings.json", "r") as json_file:
                    cell_data = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a new settings file if it doesn't exist or is invalid
                cell_data = {}

            cell_data["feed_rate_milling"] = feed_rate_milling
            cell_data["feed_rate_gripping_approaching"] = feed_rate_gripping_approaching
            cell_data["feed_rate_gripping_leaving"] = feed_rate_gripping_leaving
            cell_data["feed_rate_screwing_approaching"] = feed_rate_screwing_approaching
            cell_data["feed_rate_screwing_leaving"] = feed_rate_screwing_leaving     
            cell_data["milling_penalty"] = milling_penalty

            with open("cell_settings.json", "w") as json_file:
                json.dump(cell_data, json_file, indent=4)
            
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")

class ApproachingDistancesDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Set Approaching Distances")

    def create_widgets(self):
        # Create entry fields for each parameter with default values
        self.screwing_start_offset = self.create_entry("Start offset to part in mm (screwing)", "100.0")
        self.screwing_pullout_factor = self.create_entry("Pullout distance factor regarding the part length (screwing)", "1.1")
        self.screwing_minimal_pullout_value = self.create_entry("Minimal pullout value in mm (screwing)", "100.0")
        self.gripping_start_offset = self.create_entry("Start offset to part in mm (gripping)", "100.0")
        self.gripping_pullout_factor = self.create_entry("Pullout distance factor regarding the part length (gripping)", "2.1")
        self.gripping_minimal_pullout_value = self.create_entry("Minimal pullout value in mm (gripping)", "100.0")
        
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)
        self.center_window()

    def submit(self):
        try:
            # Get values from all entry fields
            screwing_offset = float(self.screwing_start_offset.get())
            screwing_pullout_factor = float(self.screwing_pullout_factor.get())
            screwing_minimal_pullout_value = float(self.gripping_start_offset.get())
            gripping_offset = float(self.gripping_start_offset.get())
            gripping_pullout_factor = float(self.gripping_pullout_factor.get())
            gripping_minimal_pullout_value = float(self.gripping_start_offset.get())
            
            # Log the values
            self.parent.add_output_line(f"Screwing start offset: {screwing_offset} mm")
            self.parent.add_output_line(f"Screwing pullout factor: {screwing_pullout_factor}")
            self.parent.add_output_line(f"Screwing minimal pullout value: {screwing_minimal_pullout_value} mm")
            self.parent.add_output_line(f"Gripping start offset: {gripping_offset} mm")
            self.parent.add_output_line(f"Gripping pullout factor: {gripping_pullout_factor}")
            self.parent.add_output_line(f"Gripping minimal pullout value: {gripping_minimal_pullout_value} mm")
            
            try:
                # Try to load existing settings
                with open("cell_settings.json", "r") as json_file:
                    cell_data = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a new settings file if it doesn't exist or is invalid
                cell_data = {}

            cell_data["screwing_start_offset"] = screwing_offset
            cell_data["screwing_pullout_factor"] = screwing_pullout_factor
            cell_data["screwing_minimal_pullout_value"] = screwing_minimal_pullout_value
            cell_data["gripping_start_offset"] = gripping_offset
            cell_data["gripping_pullout_factor"] = gripping_pullout_factor
            cell_data["gripping_minimal_pullout_value"] = gripping_minimal_pullout_value
            
            with open("cell_settings.json", "w") as json_file:
                json.dump(cell_data, json_file, indent=4)
            
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")

class ActionDurationDialog(BaseDialog):
    def __init__(self, parent):
        self.json_path = os.path.join(os.path.dirname(__file__), "action_duration.json")
        super().__init__(parent, "Adjust Action durations (just the time of red line)")
        self.parent = parent

    def create_widgets(self):
        # Create a frame that will contain all the entries
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar for many entries
        self.canvas = tk.Canvas(self.main_frame)
        scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Load data from JSON file
        #print(f" json_path: {self.json_path}")
        try:
            with open(self.json_path, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
            messagebox.showerror("Error", "No json file found.")
        
        # Create entry fields for each probability
        self.entries = {}
        row = 0
        
        # Add header
        header_label = tk.Label(self.scrollable_frame, text="Action duration times in s", font=self.parent.medium_font)
        header_label.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        # Create entries in a grid layout
        for key, value in self.data.items():
            # Initialize variables
            label_text = key
            path_duration = 0.0
            
            # Extract action_type and part_name from the key and value if available
            if isinstance(value, dict) and "action_type" in value and "part_name" in value:
                action_type = value["action_type"]
                part_name = value["part_name"]
                path_duration = value.get("path_duration", 0.0)
                
                # Format label text as requested
                label_text = f"{action_type} {part_name}"
            
            label = tk.Label(self.scrollable_frame, text=f"{label_text}:", anchor="w")
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            entry = tk.Entry(self.scrollable_frame)
            entry.insert(0, str(path_duration))
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            self.entries[key] = entry
            row += 1
        
        # Make the value column expandable
        self.scrollable_frame.columnconfigure(1, weight=1)
        
        # Create the submit button in a separate frame
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.submit_button = tk.Button(button_frame, text="Save Duration Times", command=self.submit)
        self.submit_button.pack(side="right")
        
        # Center the window after creating all widgets
        self.center_window()

    def submit(self):
        try:
            # Load existing data first to preserve structure
            with open(self.json_path, "r") as f:
                existing_data = json.load(f)

            # Update only the path_duration field for each entry
            for key, entry in self.entries.items():
                value = float(entry.get())
                if not (value > 0):
                    messagebox.showerror("Error", f"Value for {key} must be larger than 0")
                    return
                
                # Keep all existing data but update only the path_duration
                if key in existing_data:
                    existing_data[key]["path_duration"] = value
                else:
                    # If key doesn't exist in the original file, create a new entry
                    existing_data[key] = {"path_duration": value}

            # Save the updated JSON
            with open(self.json_path, "w") as f:
                json.dump(existing_data, f, indent=4)
            
            self.parent.add_output_line(f"Action durations saved to {self.json_path}")
            self.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number format: {str(e)}")

class ActionsucceedDialog(BaseDialog):
    def __init__(self, parent):
        self.json_path = os.path.join(os.path.dirname(__file__), "..", "GraphMDPAnalyser_main", "GraphMDPAnalyser", "action_succeed_probs.json")
        super().__init__(parent, "Set probabilities of non-destructive disassembly actions")
        self.parent = parent

    def create_widgets(self):
        # Create a frame that will contain all the entries
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar for many entries
        self.canvas = tk.Canvas(self.main_frame)
        scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Load data from JSON file
        print(f" json_path: {self.json_path}")
        try:
            with open(self.json_path, "r") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
            messagebox.showerror("Error", "No json file found.")
        
        # Create entry fields for each probability
        self.entries = {}
        row = 0
        
        # Add header
        header_label = tk.Label(self.scrollable_frame, text="probabilities of non-destructive disassembly actions", font=self.parent.medium_font)
        header_label.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        # Create entries in a grid layout
        for key, value in self.data.items():
            label_text = key
            if "__" in key:
                text_parts = key.split("__")
                if len(text_parts) >= 3:
                    label_text = f"{text_parts[2]} {text_parts[1]}"  # Format as "action_type part_name"
            
            label = tk.Label(self.scrollable_frame, text=f"{label_text}:", anchor="w")
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            entry = tk.Entry(self.scrollable_frame)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            self.entries[key] = entry
            row += 1
        
        # Make the value column expandable
        self.scrollable_frame.columnconfigure(1, weight=1)
        
        # Create the submit button in a separate frame
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.submit_button = tk.Button(button_frame, text="Save Probabilities", command=self.submit)
        self.submit_button.pack(side="right")
        
        # Center the window after creating all widgets
        self.center_window()

    def submit(self):
        new_data = {}
        try:
            for key, entry in self.entries.items():
                value = float(entry.get())
                if not (0.0 <= value <= 1.0):
                    messagebox.showerror("Error", f"Value for {key} must be between 0.0 and 1.0")
                    return
                new_data[key] = value

            # Save the updated JSON
            with open(self.json_path, "w") as f:
                json.dump(new_data, f, indent=4)
            
            self.parent.add_output_line(f"probabilities of non-destructive disassembly actions saved to {self.json_path}")
            self.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number format: {str(e)}")

class SelectRLTargetPartDialog(BaseDialog):
    def __init__(self, parent):
        self.parent = parent
        self.target_part = None  # Initialize target_part attribute
        
        # Get parts list from assembly
        if hasattr(parent, 'assembly') and parent.assembly is not None:
            self.parts_list = [part.name for part in parent.assembly.parts]
        else:
            self.parts_list = []
            messagebox.showwarning("Warning", "No assembly loaded or no parts found.")
        
        # Try to load existing target part from JSON file
        try:
            with open("q_learning_parameters.json", "r") as json_file:
                q_parameter = json.load(json_file)
                self.default_target = q_parameter.get("target_part", [""])[0]
        except (FileNotFoundError, json.JSONDecodeError):
            self.default_target = ""
            
        super().__init__(parent, "Select Target Part for RL Agent")

    def create_widgets(self):
        # Create title label
        self.title_label = tk.Label(
            self, 
            text="Select the target part for the RL agent:", 
            font=self.parent.medium_font
        )
        self.title_label.pack(pady=(20, 10))
        
        # Create dropdown menu for part selection
        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(fill="x", padx=20, pady=10)
        
        self.selected_part = tk.StringVar(self)
        # Set default value from JSON if it exists in parts list
        if self.default_target and self.default_target in self.parts_list:
            self.selected_part.set(self.default_target)
        elif self.parts_list:
            self.selected_part.set(self.parts_list[0])  # Default to first part
        else:
            self.selected_part.set("")
        
        self.part_dropdown = tk.OptionMenu(self.dropdown_frame, self.selected_part, *self.parts_list)
        self.part_dropdown.config(font=self.parent.medium_font)
        self.part_dropdown.pack(fill="x")
        
        # Create hint label
        self.hint_label = tk.Label(
            self, 
            text="This part will be used as the target for RL agent training.",
            font=self.parent.medium_font,
            fg="gray50",
            justify=tk.CENTER
        )
        self.hint_label.pack(pady=10)
        
        # Submit button
        self.submit_button = tk.Button(self, text="Set Target Part", command=self.submit)
        self.submit_button.pack(pady=20)
        
        self.center_window()

    def submit(self):
        # Get selected part name
        self.target_part = self.selected_part.get()
        
        if not self.target_part:
            messagebox.showerror("Error", "Please select a target part")
            return
        
        try:
            # Try to load existing settings
            with open("q_learning_parameters.json", "r") as json_file:
                q_parameter = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create a new settings file if it doesn't exist or is invalid
            q_parameter = {}
        
        # Add target part to parameters
        q_parameter["target_part"] = [self.target_part]
        
        # Save updated parameters
        with open("q_learning_parameters.json", "w") as json_file:
            json.dump(q_parameter, json_file, indent=4)
        
        self.parent.add_output_line(f"Target part for RL agent set to: {self.target_part}")
        self.parent.target_part = self.target_part  # Set it in the parent
                   
        self.destroy()
    
    def show(self):
        """Shows the dialog and waits for it to close, then returns the selected target part"""
        self.wait_window()  # Wait for the dialog to close
        return self.target_part

class QLearningParametersDialog(BaseDialog):
    def __init__(self, parent):
        # Import matplotlib here to avoid dependency issues
        import matplotlib.pyplot as plt
        
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        super().__init__(parent, "Adjust Q-Learning Parameters")
        
    def create_widgets(self):
        # Create main frame for parameters
        self.params_frame = tk.Frame(self)
        self.params_frame.pack(fill="x", padx=10, pady=10)

        # Try to load existing parameters from JSON
        try:
            with open("q_learning_parameters.json", "r") as json_file:
                q_parameter = json.load(json_file)
            
            # Set default values from loaded parameters or fallback to defaults
            alpha_default = str(q_parameter.get("alpha", 0.7))
            gamma_default = str(q_parameter.get("gamma", 0.99))
            episodes_default = str(q_parameter.get("episodes", 2000))
            epsilon_start_default = str(q_parameter.get("epsilon_start", 0.8))
            epsilon_min_default = str(q_parameter.get("epsilon_min", 0.1))
            decay_rate_default = str(q_parameter.get("decay_rate", 0.0015))
            show_optimal_path_default = q_parameter.get("show_optimal_path", False)
        except (FileNotFoundError, json.JSONDecodeError):
            # Use defaults if file doesn't exist or is invalid
            alpha_default = "0.7"
            gamma_default = "0.99"
            episodes_default = "2000"
            epsilon_start_default = "0.8"
            epsilon_min_default = "0.1"
            decay_rate_default = "0.0015"
            show_optimal_path_default = False

        # Create StringVars for tracking changes with loaded defaults
        self.alpha_var = tk.StringVar(value=alpha_default)
        self.gamma_var = tk.StringVar(value=gamma_default)
        self.episodes_var = tk.StringVar(value=episodes_default)
        self.epsilon_start_var = tk.StringVar(value=epsilon_start_default)
        self.epsilon_min_var = tk.StringVar(value=epsilon_min_default)
        self.decay_rate_var = tk.StringVar(value=decay_rate_default)
        self.show_optimal_path_var = tk.BooleanVar(value=show_optimal_path_default)
        
        # Add trace to variables to update plot
        self.epsilon_start_var.trace_add("write", self.update_plot)
        self.epsilon_min_var.trace_add("write", self.update_plot)
        self.decay_rate_var.trace_add("write", self.update_plot)
        self.episodes_var.trace_add("write", self.update_plot)
        
        # Create entry fields for each parameter with default values
        self.alpha = self.create_entry_with_var("Alpha - Learning rate", self.alpha_var, parent_frame=self.params_frame)
        self.gamma = self.create_entry_with_var("Gamma - discount factor", self.gamma_var, parent_frame=self.params_frame)
        self.episodes = self.create_entry_with_var("Number of Training Episodes", self.episodes_var, parent_frame=self.params_frame)
        self.epsilon_start = self.create_entry_with_var("Epsilon start", self.epsilon_start_var, parent_frame=self.params_frame)
        self.epsilon_min = self.create_entry_with_var("Epsilon minimum", self.epsilon_min_var, parent_frame=self.params_frame)
        self.decay_rate = self.create_entry_with_var("Epsilon decay rate (exponential decay)", self.decay_rate_var, parent_frame=self.params_frame)
        
        # Add checkbox for showing optimal path
        self.show_optimal_path_chk = tk.Checkbutton(
            self.params_frame, 
            text="Show optimal path in MDP-Graph (plotly)", 
            variable=self.show_optimal_path_var,
            font=self.parent.medium_font
        )
        self.show_optimal_path_chk.pack(pady=10, anchor="w")
        
        # Create graph frame
        self.graph_frame = tk.Frame(self)
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        self.fig = self.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = self.FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initial plot
        self.update_plot()
        
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)
        self.center_window()
    
    def create_entry_with_var(self, label_text, variable, parent_frame=None):
        if parent_frame is None:
            parent_frame = self
            
        label = tk.Label(parent_frame, text=label_text)
        label.pack()
        entry = tk.Entry(parent_frame, textvariable=variable)
        entry.pack()
        setattr(self, label_text.replace(" ", "_").lower() + "_entry", entry)
        return entry
    
    def update_plot(self, *args):
        try:
            # Get current values
            epsilon_start = float(self.epsilon_start_var.get())
            epsilon_min = float(self.epsilon_min_var.get())
            decay_rate = float(self.decay_rate_var.get())
            episodes = int(float(self.episodes_var.get()))
            
            # Create data for plot
            # Use actual episode count but sample points for better performance with large numbers
            if episodes > 10000:
                step = episodes // 1000
                episodes_range = range(0, episodes + 1, step)
            else:
                episodes_range = range(0, episodes + 1)
            
            # Calculate epsilon values using exponential decay
            #epsilon_values = [max(epsilon_min, epsilon_start * (decay_rate ** i)) for i in episodes_range]
            epsilon_values = [max(epsilon_min, epsilon_start * np.exp(-decay_rate * i)) for i in episodes_range]
             
            
            # Clear previous plot
            self.ax.clear()
            
            # Create new plot
            self.ax.plot(episodes_range, epsilon_values)
            self.ax.set_xlabel('Episodes')
            self.ax.set_ylabel('Epsilon Value')
            self.ax.set_title('Epsilon Decay Over Episodes')
            self.ax.set_ylim(0, 0.5)  # Fixed y-axis scale from 0 to 0.5
            self.ax.set_xlim(0, episodes)  # Set x-axis to match the number of episodes
            self.ax.grid(True)
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
        except (ValueError, ZeroDivisionError):
            # Ignore errors during input
            pass

    def submit(self):
        try:
            # Get values from all entry fields
            alpha = float(self.alpha_var.get())
            gamma = float(self.gamma_var.get())
            episodes = int(self.episodes_var.get())
            epsilon_start = float(self.epsilon_start_var.get())
            epsilon_min = float(self.epsilon_min_var.get())
            decay_rate = float(self.decay_rate_var.get())
            show_optimal_path = self.show_optimal_path_var.get()
            self.parent.show_optimal_path = show_optimal_path
            
            # Log the values
            self.parent.add_output_line(f"alpha: {alpha}")
            self.parent.add_output_line(f"gamma: {gamma}")
            self.parent.add_output_line(f"episodes: {episodes}")
            self.parent.add_output_line(f"epsilon_start: {epsilon_start}")
            self.parent.add_output_line(f"epsilon_min: {epsilon_min}")
            self.parent.add_output_line(f"decay_rate: {decay_rate}")
            self.parent.add_output_line(f"show_optimal_path: {show_optimal_path}")
            
            try:
                # Try to load existing settings
                with open("q_learning_parameters.json", "r") as json_file:
                    q_parameter = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # Create a new settings file if it doesn't exist or is invalid
                q_parameter = {}

            q_parameter["alpha"] = alpha
            q_parameter["gamma"] = gamma
            q_parameter["episodes"] = episodes
            q_parameter["epsilon_start"] = epsilon_start
            q_parameter["epsilon_min"] = epsilon_min
            q_parameter["decay_rate"] = decay_rate
            q_parameter["show_optimal_path"] = show_optimal_path
            
            with open("q_learning_parameters.json", "w") as json_file:
                json.dump(q_parameter, json_file, indent=4)
            
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")

class ActionResultDialog(BaseDialog):
    def __init__(self, parent, g_markov, current_node, show_save_checkbox=False, last_save_flag=True):
        self.result = None  # Will store the result (True for success, False for failure)
        self.g_markov = g_markov  # Store the graph for accessing successors
        self.current_node = current_node
        self.show_save_checkbox = show_save_checkbox
        self.save_data_to_send_flag = last_save_flag if show_save_checkbox else None
        super().__init__(parent, "Action Result")

    def create_widgets(self):
        # Title label
        tk.Label(self, text="Was the action successful?", font=self.parent.medium_font).pack(pady=(20, 10))

        # Frame to hold the radio buttons
        self.radio_frame = tk.Frame(self)
        self.radio_frame.pack(pady=10)

        # Use a single IntVar to track which option is selected (1 for Yes, 0 for No)
        self.selected_option = tk.IntVar(value=-1)  # Default to no selection

        # Create the radio buttons
        self.yes_radio = tk.Radiobutton(self.radio_frame, text="Yes", variable=self.selected_option,
                                       value=1, font=self.parent.medium_font)
        self.yes_radio.pack(side=tk.LEFT, padx=20)

        self.no_radio = tk.Radiobutton(self.radio_frame, text="No", variable=self.selected_option,
                                      value=0, font=self.parent.medium_font)
        self.no_radio.pack(side=tk.LEFT, padx=20)

        # Optional: Checkbox für "data_to_send speichern?"
        if self.show_save_checkbox:
            self.save_var = tk.BooleanVar(value=self.save_data_to_send_flag)
            self.save_checkbox = tk.Checkbutton(self, text="data_to_send speichern?", variable=self.save_var)
            self.save_checkbox.pack(pady=(10, 0))

        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=20)

        self.center_window()

    def submit(self):
        if self.selected_option.get() == -1:
            messagebox.showwarning("Warning", "Please select an option")
            return
        self.result = bool(self.selected_option.get())
        self.parent.action_result = self.result  # Store result in parent object
        if self.show_save_checkbox:
            self.save_data_to_send_flag = self.save_var.get()
        self.destroy()

class MountingConfigDialog(BaseDialog):
    def __init__(self, parent):
        self.parent = parent
        self.current_pos, self.current_orient = self.parent.request_spawn_pose()
        super().__init__(parent, "Define Mounting Configuration")

    def create_widgets(self):
        if self.current_pos is None or self.current_orient is None:
            messagebox.showerror("Error", "Failed to retrieve current spawn position.")
            self.destroy()
            return

        euler_orient = R.from_quat(self.current_orient).as_euler('xyz', degrees=True)

        # Name field
        self.name_entry = self.create_entry("Configuration Name", "NewConfig")

        # Position fields
        self.pos_x_entry = self.create_entry("Position X", str(self.current_pos[0]))
        self.pos_y_entry = self.create_entry("Position Y", str(self.current_pos[1]))
        self.pos_z_entry = self.create_entry("Position Z", str(self.current_pos[2]))

        # Orientation fields
        self.roll_entry = self.create_entry("Roll", str(euler_orient[0]))
        self.pitch_entry = self.create_entry("Pitch", str(euler_orient[1]))
        self.yaw_entry = self.create_entry("Yaw", str(euler_orient[2]))

        # Submit button
        self.submit_button = tk.Button(self, text="Save Configuration", command=self.submit)
        self.submit_button.pack(pady=10)

        self.center_window()

    def submit(self):
        name = self.configuration_name_entry.get().strip()

        try:
            pos = np.array([
                float(self.position_x_entry.get()),
                float(self.position_y_entry.get()),
                float(self.position_z_entry.get())
            ])
            roll = float(self.roll_entry.get())
            pitch = float(self.pitch_entry.get())
            yaw = float(self.yaw_entry.get())
            orient = np.array([roll, pitch, yaw])  # Keep as degrees

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
            return
        # Update in-memory dictionary
        self.parent.mounting_configurations[name] = {
            'position': pos,
            'orientation': orient
        }
        self.parent.save_mounting_configurations()
        self.destroy()


class InputNodeDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "Enter Current Node")
        self.result = None

    def create_widgets(self):
        # Add instruction label
        instruction_label = tk.Label(self, text="Please enter the current Node/state in the disassembly graph")
        instruction_label.pack(pady=5)
        
        self.node_entry = self.create_entry("Node number:", "0")
        
        # Submit button
        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack(pady=5)
        self.center_window()

    def submit(self):
        try:
            node_value = int(self.node_entry.get())
            self.result = node_value
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter an integer.")


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.assembly = None  # Initialize assembly as None

        self.planner = None  # Initialize planner as None
        self.path = None # Initialize path as None
        self.currently_showing = {"type": None, "number": None, "index": None}
        self.current_state_number = None
        self.disassembly_target_part_names = []
        self.show_optimal_path = False  # Initialize show_optimal_path
        self.parts_list = []  # Initialize parts list
        self.graph_folder = None

        self.node_history = ""  # Initialize node history
        self.disassembly_done = False

        # Initialize the configparser
        self.config = configparser.ConfigParser()

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the path to the config file
        config_path = os.path.join(script_dir, 'config.ini')

        # Check if the configuration file exists
        if os.path.isfile(config_path):
            self.config.read(config_path)
        else:
            messagebox.showerror("Error", f"Configuration file 'config.ini' not found in {script_dir}")
            sys.exit(1)  # Exit the application if config file is missing
        self.available_screwdrivers = self.config['Settings']['available_screwdrivers'].split(', ')
        print(f"available screwdrivers: {self.available_screwdrivers}")
        self.data_directory = resolve_path(self.config['Settings']['data_directory'])
        print(f"data directory: {self.data_directory}")
        server_url = self.config['Settings']['opc_server_url']
        self.view_distance = self.config['View_Parameters'].getfloat('view_distance')
        self.view_angle_X = self.config['View_Parameters'].getfloat('view_angle_X')
        self.view_angle_Y = self.config['View_Parameters'].getfloat('view_angle_Y')
        self.view_angle_Z = self.config['View_Parameters'].getfloat('view_angle_Z')

        self.contact_tollerance = self.config['Assembly'].getfloat('contact_tollerance')
        self.dot_product_threshold = -math.cos(math.radians(self.config['Assembly'].getfloat('contact_tollerance_angle')))
        print(f"contact tolerance: {self.contact_tollerance}, dot product threshold: {self.dot_product_threshold}")


        # Creating an instance of the Client class
        self.client = Client(server_url)

        # Setting the username and password
        self.client.set_user(self.config['Settings']['opc_user_name'])
        self.client.set_password(self.config['Settings']['opc_password'])
        self.robot_C_status_node_id = "ns=2;s=/Channel/Parameter/R[u1,0]"
        self.robot_D_status_node_id = "ns=2;s=/Channel/Parameter/R[u2,0]"


        # Create empty Zivid Object
        self.zivid_settings = resolve_path(self.config['Settings']['zivid_settings'])
        self.zivid_wd = resolve_path(self.config['Settings']['zivid_workingdirectory'])
        self.zivid_camera = None
        self.zivid_frame_buffer = []

        # create datastructures for simulated camera
        self.last_frame_from_sim = {}
        self.sim_frame_buffer = []
        self.transformation = np.eye(4)

        #initialize perceptionModule
        self.perception_module = PerceptionModule()


        #set server parameters
        self.sim_host = self.config['Settings']['sim_host']
        self.sim_port = self.config['Settings'].getint('sim_port')
        self.sim_wd = resolve_path(self.config['Settings']['sim_workingdirectory'])

        #load mounting position and orientation
        self.mounting_pos = np.array(self.config['Settings']['mounting_position'].split(',')).astype(float)
        self.mounting_orn = np.array(self.config['Settings']['mounting_orientation'].split(',')).astype(float) #TODO orientation change currently not supported in sim
        print(f"mounting position: {self.mounting_pos}, mounting orientation: {self.mounting_orn}")
        self.operations_folder = resolve_path(self.config['Settings']['operations_folder'])
        self.mounting_xml_path = resolve_path(self.config['Settings']['mounting_xml_path'])
        self.mounting_configurations = {}
        #self.load_mounting_configurations() #auskommentiert weil kp was es macht


        self.pack(fill=tk.BOTH, expand=True)
        self.create_menu()


        # Create frames
        self.assembly_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.planning_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.cell_design_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.mdp_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.agent_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.sim_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.execution_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.perception_frame = tk.Frame(self, borderwidth=2, relief="groove")
        self.status_frame = tk.Frame(self, borderwidth=2, relief="groove")

        # Configure grid column weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Adjust row for vertical spacing
        self.grid_rowconfigure(1, weight=1)  # Adjust row for vertical spacing
        self.grid_rowconfigure(2, weight=1)  # Adjust row for vertical spacing
        self.grid_rowconfigure(3, weight=1)  # Adjust row for vertical spacing

        # Position the frames sorted by column
        self.assembly_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="new")
        self.status_frame.grid(row=2, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.cell_design_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.planning_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky="new")
        self.mdp_frame.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")
        
        self.sim_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="new")
        self.execution_frame.grid(row=0, column=3, rowspan=2, padx=5, pady=5, sticky="new")
        self.perception_frame.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.agent_frame.grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky="nsew") #sollte perception ueberschreiben. TODO @david: noch schoener machen

        # Load the initial image
        os.chdir(script_dir_path) #war nötig um dateien zu finden
        original_image = Image.open("WSK_Logo_full_scale.png")
        self.image_wsk = ImageTk.PhotoImage(original_image)
        # Create a label and add the image
        self.image_wsk_label = tk.Label(self, image=self.image_wsk)
        #self.image_wsk_label.grid(row=2, column=0, padx=5, pady=5, sticky="sew")

        # Load the initial image
        original_image = Image.open("WBK_4C_D_scale.png")
        self.image_wbk = ImageTk.PhotoImage(original_image)
        # Create a label and add the image
        self.image_wbk_label = tk.Label(self, image=self.image_wbk)
       # self.image_wbk_label.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky="sew")

        # Defining fonts
        self.large_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        self.medium_font = tkfont.Font(family="Helvetica", size=12)

        #Assembly section
        tk.Label(self.assembly_frame, text="Assembly", font=self.large_font).pack(anchor='n', fill=tk.X)
        # Create a label to display the current assembly name
        self.assembly_label = tk.Label(self.assembly_frame, text="Currently loaded: None", font=self.medium_font)
        self.assembly_label.pack(side='top', anchor='w')
        self.image_label_asm = tk.Label(self.assembly_frame)
        self.image_label_asm.pack(side='top', anchor='center')
        img = Image.new('RGB', (800, 800), "white")
        self.img_asm_size = (300,300)
        self.display_image_on_lable(img, self.image_label_asm, self.img_asm_size)
        # Create a sub-frame within the assembly frame
        self.asm_button_frame = tk.Frame(self.assembly_frame)
        self.asm_button_frame.pack(fill=tk.BOTH, expand=True, side='top', anchor='center')
        self.asm_button_frame.grid_columnconfigure(0, weight=1)
        self.asm_button_frame.grid_columnconfigure(1, weight=1)
        self.asm_button_frame.grid_columnconfigure(2, weight=1)
        self.asm_button_frame.grid_columnconfigure(3, weight=1)
        tk.Button(self.asm_button_frame, text="Load Parts", font=self.medium_font, command=self.load_parts_dialog).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.asm_button_frame, text="Add Relation", font=self.medium_font, command=self.add_relation).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.asm_button_frame, text="Load XML", font=self.medium_font, command=self.load_xml).grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.asm_button_frame, text="Save XML", font=self.medium_font, command=self.save_as_xml).grid(row=0, column=3, padx=5, pady=5, sticky="nsew")



        #Planning section
        tk.Label(self.planning_frame, text="Planning", font=self.large_font).pack(anchor='n', fill=tk.X)
        # Create a label to display the planner parameters
        self.planner_parameters_label = tk.Label(self.planning_frame, text="Planner parameters: None", font=self.medium_font)
        self.planner_parameters_label.pack(side='top', anchor='w')

        self.image_label_plan = tk.Label(self.planning_frame)
        self.image_label_plan.pack(side='top', anchor='center')
        img = Image.new('RGB', (800, 800), "white")
        self.img_plan_size = (400,400)
        self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)

        # Create sub-frames within the planning frame
        self.plan_button_frame_top = tk.Frame(self.planning_frame)
        self.plan_button_frame_top.pack(fill=tk.BOTH, expand=True, side='top', anchor='center')
        self.plan_button_frame_top.grid_columnconfigure(0, weight=1)
        self.plan_button_frame_top.grid_columnconfigure(1, weight=1)
        self.plan_button_frame_top.grid_columnconfigure(2, weight=1)
        self.plan_button_frame_top.grid_columnconfigure(3, weight=1)
        tk.Button(self.plan_button_frame_top, text="Initialize Planner", font=self.medium_font, command=self.create_planner_dialog)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_top, text="Load Planner", font=self.medium_font, command=self.load_from_pkl)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_top, text="Save Planner", font=self.medium_font, command=self.save_as_pkl)\
            .grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_top, text="Save Graph(infos)", font=self.medium_font, command=self.save_reduced_graph)\
            .grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_top, text="Plan Sequence To Target", font=self.medium_font, command=self.plan_sequence_to_target_dialog)\
            .grid(row=0, column=4, padx=5, pady=5, sticky="nsew")

        self.plan_button_frame_bottom = tk.Frame(self.planning_frame)
        self.plan_button_frame_bottom.pack(fill=tk.BOTH, expand=True, side='top', anchor='center')
        self.plan_button_frame_bottom.grid_columnconfigure(0, weight=2)
        self.plan_button_frame_bottom.grid_columnconfigure(1, weight=2)
        self.plan_button_frame_bottom.grid_columnconfigure(2, weight=1)
        self.plan_button_frame_bottom.grid_columnconfigure(3, weight=15)
        self.plan_button_frame_bottom.grid_columnconfigure(4, weight=1)
        self.plan_button_frame_bottom.grid_columnconfigure(5, weight=2)
        tk.Button(self.plan_button_frame_bottom, text="Show Graph", font=self.medium_font, command=self.show_graph)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_bottom, text="Show Start", font=self.medium_font, command= self.show_start)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_bottom, text="<", font=self.medium_font, command=self.prev)\
            .grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.show_lable = tk.Label(self.plan_button_frame_bottom, text="Nothing to show.", font=self.medium_font)
        self.show_lable.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_bottom, text=">", font=self.medium_font, command=self.next)\
            .grid(row=0, column=4, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_bottom, text="Show End", font=self.medium_font, command= self.show_result)\
            .grid(row=0, column=5, padx=5, pady=5, sticky="nsew")
        tk.Button(self.plan_button_frame_bottom, text="Save Sequence", font=self.medium_font, command= self.export_sequence)\
            .grid(row=0, column=6, padx=5, pady=5, sticky="nsew")


        # Create a label to display the currently planned sequence
        self.planned_sequence_lable = tk.Label(self.planning_frame, text="Current sequence: None", font=self.medium_font)
        self.planned_sequence_lable.pack(side='bottom', anchor='w')

        #Cell design section
        tk.Label(self.cell_design_frame, text="Cell Design", font=self.large_font).pack(anchor='n', fill=tk.X)
        self.cell_design_button_frame = tk.Frame(self.cell_design_frame)
        self.cell_design_button_frame.pack(fill=tk.BOTH, expand=True)  # Add this line to place the frame in the parent
        self.cell_design_button_frame.grid_columnconfigure(0, weight=1)
        self.cell_design_button_frame.grid_columnconfigure(1, weight=1)
        
        tk.Button(self.cell_design_button_frame, text="Adjust travel times", font=self.medium_font, command=self.adjust_travel_times)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.cell_design_button_frame, text="Adjust Toolchange times", font=self.medium_font, command=self.adjust_toolchange_times)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.cell_design_button_frame, text="Adjust feed rate", font=self.medium_font, command=self.adjust_feed_rate)\
            .grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.cell_design_button_frame, text="Adjust approach distance to assembly", font=self.medium_font, command=self.adjust_distances_to_assembly)\
            .grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        #MDP section
        tk.Label(self.mdp_frame, text="Graph & MDP-Section", font=self.large_font).pack(anchor='n', fill=tk.X)
        self.mdp_button_frame = tk.Frame(self.mdp_frame)
        self.mdp_button_frame.pack(fill=tk.BOTH, expand=True)  # Add this line to place the frame in the parent
        self.mdp_button_frame.grid_columnconfigure(0, weight=1)
        self.mdp_button_frame.grid_columnconfigure(1, weight=1)
        #self.mdp_button_frame.grid_columnconfigure(2, weight=1)
        
        tk.Button(self.mdp_button_frame, text="Load Graph", font=self.medium_font, command=self.load_graph)\
            .grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.mdp_button_frame, text="Show Graph (plotly)", font=self.medium_font, command=self.show_graph_dash)\
            .grid(row=3, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.mdp_button_frame, text="Convert graph to MDP", font=self.medium_font, command=self.convert_graph_to_mdp)\
            .grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.mdp_button_frame, text="Load MPD", font=self.medium_font, command=self.load_mdp_graph)\
            .grid(row=4, column=1, padx=5, pady=5, sticky="nsew") 
        tk.Button(self.mdp_button_frame, text="Save MPD-Graph", font=self.medium_font, command=self.save_mdp_graph_pre)\
            .grid(row=5, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.mdp_button_frame, text="Show MPD", font=self.medium_font, command=self.show_mdp_graph_in_dash)\
            .grid(row=5, column=1, padx=5, pady=5, sticky="nsew") 
        
        #RL-Agent-Section 
        tk.Label(self.agent_frame, text="RL-Section", font=self.large_font).pack(anchor='n', fill=tk.X)
        self.agent_button_frame = tk.Frame(self.agent_frame)
        self.agent_button_frame.pack(fill=tk.BOTH, expand=True)  # Add this line to place the frame in the parent
        self.agent_button_frame.grid_columnconfigure(0, weight=1)
        self.agent_button_frame.grid_columnconfigure(1, weight=1)
        
        tk.Button(self.agent_button_frame, text="Select target Part", font=self.medium_font, command=self.select_target_part)\
            .grid(row=6, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Agent Settings", font=self.medium_font, command=self.adjust_q_learning_parameters)\
            .grid(row=6, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Reset Agent", font=self.medium_font, command=self.delete_q_table)\
            .grid(row=6, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Train Agent", font=self.medium_font, command=self.train_agent)\
            .grid(row=7, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Load Agent", font=self.medium_font, command=self.load_agent)\
            .grid(row=7, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Reset RL-sequence", font=self.medium_font, command=self.reset_RL)\
            .grid(row=7, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="scan -> node", font=self.medium_font, command=self.scan_to_node)\
            .grid(row=8, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="get next action", font=self.medium_font, command=self.get_next_action)\
            .grid(row=8, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.agent_button_frame, text="Action result?", font=self.medium_font, command=self.action_result)\
            .grid(row=8, column=2, padx=5, pady=5, sticky="nsew")
        # Simulation section
        tk.Label(self.sim_frame, text="Simulation", font=self.large_font).pack(anchor='n', fill=tk.X)
        tk.Button(self.sim_frame, text="Check Connection", font=self.medium_font, command=self.check_connection)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="Instanciate Simulation", font=self.medium_font, command=self.instanciate_simulation)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="Load Assembly", font=self.medium_font, command=self.load_assembly_in_sim)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="Reposition Assembly", font=self.medium_font, command=self.open_manual_reposition_dialog) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="Look at part", font=self.medium_font, command=self.look_at_part_dialog)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="save gcode", font=self.medium_font, command=self.save_g_code) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="execute action", font=self.medium_font, command=self.execute_action) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.sim_frame, text="capture image of cell", font=self.medium_font, command=self.capture_image_of_cell) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        """tk.Button(self.sim_frame, text="rotate assembly", font=self.medium_font, command=self.rotate_assembly) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)"""
        tk.Button(self.sim_frame, text="execute RL-action", font=self.medium_font, command=self.execute_optimal_action) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)

        """tk.Button(self.sim_frame, text="Simulate Sequence", font=self.medium_font, command=self.simulate_sequence) \
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)"""

        #Execution section
        tk.Label(self.execution_frame, text="Execution", font=self.large_font).pack(anchor='n', fill=tk.X)
        tk.Button(self.execution_frame, text="Transmit to Robot", font=self.medium_font, command=self.send_nc_file)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Label(self.execution_frame, text="Automatic mode:", font=self.medium_font).pack(anchor='n', fill=tk.X)
        tk.Button(self.execution_frame, text="Move Assembly", font=self.medium_font, command=self.auto_move_assembly)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.execution_frame, text="Auto look at part", font=self.medium_font, command=self.look_at_part_auto)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.execution_frame, text="Auto look at assembly", font=self.medium_font, command=self.look_at_assembly_auto)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.execution_frame, text="Auto execute alignment action", font=self.medium_font, command=self.run_initial_allignment)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.execution_frame, text="Auto execute action", font=self.medium_font, command=self.execute_action_auto)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        tk.Button(self.execution_frame, text="Auto execute sequence", font=self.medium_font, command=self.execute_sequence_auto)\
            .pack(anchor="n", fill=tk.X, padx=5, pady=5)
        #Perception section
        tk.Label(self.perception_frame, text="Perception", font=self.large_font).pack(anchor='n', fill=tk.X)

        self.image_label_perception = tk.Label(self.perception_frame)
        self.image_label_perception.pack(side='top', anchor='center')
        img = Image.new('RGB', (200, 200), "white")
        self.img_perception_size = (200,200)
        self.display_image_on_lable(img, self.image_label_perception, self.img_perception_size)
        self.perception_button_frame = tk.Frame(self.perception_frame)
        self.perception_button_frame.pack(fill=tk.BOTH, expand=True, side='bottom', anchor='center')

        tk.Button(self.perception_button_frame, text="Capture frame", font=self.medium_font, command=self.take_zivid_image)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Button(self.perception_button_frame, text="Save frame", font=self.medium_font, command=self.save_zivid_file)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Button(self.perception_button_frame, text="Load frame", font=self.medium_font, command=self.load_zivid_file) \
            .grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        tk.Button(self.perception_button_frame, text="Show Overlay", font=self.medium_font, command=self.show_perception_result)\
            .grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        tk.Button(self.perception_button_frame, text="Allign", font=self.medium_font, command=self.perform_allignment)\
            .grid(row=0, column=4, padx=5, pady=5, sticky="nsew")
        # tk.Button(self.perception_button_frame, text="Auto", font=self.medium_font, command=self.execute_measure_program)\
        #     .grid(row=0, column=5, padx=5, pady=5, sticky="nsew")
        tk.Button(self.perception_button_frame, text="Clear", font=self.medium_font, command=self.clear_pointcloud_wds)\
            .grid(row=0, column=5, padx=5, pady=5, sticky="nsew")

        #status section
        tk.Label(self.status_frame, text="Status", font=self.large_font).pack(anchor='n', fill=tk.X)
        self.text_area = scrolledtext.ScrolledText(self.status_frame, wrap=tk.WORD, width=50, height=27)
        self.text_area.pack(side='bottom', anchor='center',  fill=tk.X)

        # Disable the text area for editing
        self.text_area.config(state=tk.DISABLED)

    def create_menu(self):
        # Create a menu bar
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        # Create an 'Assembly' menu
        self.assembly_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Assembly", menu=self.assembly_menu)

        # Add 'Load Parts' and 'Save as XML' options to the 'Assembly' menu
        self.assembly_menu.add_command(label="Load Parts", command=self.load_parts_dialog)
        self.assembly_menu.add_command(label="Load XML", command=self.load_xml)
        self.assembly_menu.add_command(label="Save as XML", command=self.save_as_xml)
        self.assembly_menu.add_command(label="Rename Parts", command=self.rename_parts)
        self.assembly_menu.add_command(label="Add Relation", command=self.add_relation)
        self.assembly_menu.add_command(label="Add Screw in Pointcloud", command=self.add_screw_in_pointcloud)

        # Create a 'Planning' menu
        self.planning_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Planning", menu=self.planning_menu)
        self.planning_menu.add_command(label="Create Planner", command=self.create_planner_dialog)
        self.planning_menu.add_command(label="Plan Sequence to Target", command=self.plan_sequence_to_target_dialog)
        self.planning_menu.add_command(label="Plan multiple sequences", command=self.plan_multiple_sequences_dialog)
        self.planning_menu.add_command(label="Generate full graph", command=self.plan_full_graph_dialog)
        self.planning_menu.add_command(label="export sequence", command=self.export_sequence)
        self.planning_menu.add_command(label="export all states", command=self.export_states)
        self.planning_menu.add_command(label="export graph as pkl", command=self.export_graph)
        self.planning_menu.add_command(label="save planner as pkl", command=self.save_as_pkl)
        self.planning_menu.add_command(label="load planner from pkl", command=self.load_from_pkl)
        self.planning_menu.add_command(label="Add Handlingpoint in Pointcloud", command=self.add_handlingpoint)
        self.planning_menu.add_command(label="Save defined gripping operations", command=self.export_gripping_ops)
        self.planning_menu.add_command(label="Load gripping operations", command=self.import_gripping_ops)
        self.planning_menu.add_command(label="Remove all gripping operations of part", command=self.select_part_to_remove_handlingOPs)

        # Create a 'View' menu
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Set View Parameters", command=self.set_view_parameters_dialog)
        self.view_menu.add_command(label="Show current graph", command=self.show_graph)
        self.view_menu.add_command(label="Save current graph", command=self.save_graph_png)
        self.view_menu.add_command(label="Show state", command=self.show_state_dialog)
        self.view_menu.add_command(label="Show action", command=self.show_action_dialog)
        self.view_menu.add_command(label="Show DOF round", command=self.show_DOF_round)
        self.view_menu.add_command(label="Show DOF square", command=self.show_DOF_square)
        self.view_menu.add_command(label="Show disassemblability", command=self.show_disassemblability)
        self.view_menu.add_command(label="Show geometric connection", command=self.show_connection_dialog)
        self.view_menu.add_command(label="Print end nodes", command=self.print_end_nodes)
        self.view_menu.add_command(label="Get sequence to node", command=self.get_sequence_to_target_dialog)
        self.view_menu.add_command(label="Scene 3D", command=self.scene_3D)


        # Create a 'Simulation' menu
        self.sim_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Simulation", menu=self.sim_menu)
        self.sim_menu.add_command(label="Add Mounting Configuration", command=self.add_mounting_configuration)
        self.sim_menu.add_command(label="Lift Assembly", command=self.lift_assembly)
        self.sim_menu.add_command(label="Load Assembly - other pose", command=self.load_assembly_in_sim_op) # @TODO schauen ob nötig ist oder über repositioning reicht
        self.sim_menu.add_command(label="Update robot positions", command=self.updat_robots_in_sim)
        self.sim_menu.add_command(label="Take zivid image", command=self.take_zivid_image_sim)
        #self.sim_menu.add_command(label="Simulate Sequence", command=self.simulate_sequence)

        # Create a 'Demo' menu
        self.demo_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Demo", menu=self.demo_menu)
        self.demo_menu.add_command(label="Cable Disassembly Demo - Planner", command=self.run_CMP_demo)
        self.demo_menu.add_command(label="Cable Disassembly Demo - Machine Vision (capture)", command=self.run_CMV_demo_capture)
        self.demo_menu.add_command(label="Cable Disassembly Demo - Machine Vision (execution)", command=self.run_CMV_demo)
        self.demo_menu.add_command(label="Plug Disassembly Demo - Machine Vision (capture)", command=self.run_plug_demo_capture)
        self.demo_menu.add_command(label="Plug Disassembly Demo - Machine Vision (execution)", command=self.run_plug_demo)

        # Create a 'Perception' menu
        self.perception_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Perception", menu=self.perception_menu)
        self.perception_menu.add_command(label="Run Perception", command=self.run_perception)
        self.perception_menu.add_command(label="Scan Assembly", command=self.scan_assembly_to_pointcloud)
        self.perception_menu.add_command(label="TEMPORARY", command=self.capture_and_align)

        
        #Creae a Graph&MDP-Section menu
        self.graph_mdp_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Graph & MDP", menu=self.graph_mdp_menu)
        self.graph_mdp_menu.add_command(label="Save Node/Graph Images", command=self.save_pics)
        self.graph_mdp_menu.add_command(label="Save MDP Graph", command=self.save_mdp_graph_pre)


        # Create a 'RL-Menu' menu
        self.rl_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="RL-Menü", menu=self.rl_menu)
        self.rl_menu.add_command(label="Manipulation action durations", command=self.adjust_action_durations)
        self.rl_menu.add_command(label="Adjust probabilities of non-destructive disassembly actions", command=self.adjust_actionsucceed_porbs)

    def scene_3D(self):
        if self.currently_showing["type"] == "action":
            from_state = self.path[self.currently_showing["index"]]
            to_state = self.path[self.currently_showing["index"] + 1]
            operation = self.planner.graph.get_edge_data(from_state, to_state)["operation"]
            operation_scene(operation, self.planner.assembly_states[from_state]).show()
        else:
            if self.path is not None:
                assembly_to_trimesh_scene(self.planner.assembly_states[self.path[self.currently_showing["index"]]]).show()
            else:
                assembly_to_trimesh_scene(self.planner.assembly_states[self.currently_showing["index"]]).show()
    def show_state(self, state):
        if self.planner is not None and state < len(self.planner.assembly_states):
            self.currently_showing["type"] = "state"
            self.currently_showing["number"] = f'{state}'
            self.update_show_lable()
            scene = assembly_to_trimesh_scene(self.planner.assembly_states[state])
            img = self.generate_image_pyrender(scene)  # Generate the image from the assembly
            self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)

    def show_result(self):
        if self.path is not None:
            self.currently_showing["index"]=len(self.path)
            self.show_state(self.path[-1])
        else:
            self.currently_showing["index"] = len(self.planner.assembly_states)-1
            self.show_state(self.currently_showing["index"])
            
    def show_start(self):
        if self.path is not None:
            self.currently_showing["index"]=0
            self.show_state(self.path[0])
        else:
            self.currently_showing["index"] = 0
            self.show_state(self.currently_showing["index"])

 
    def save_pics(self, path=None):
        if self.planner is None and not hasattr(self, "graph_folder"):
            messagebox.showerror("Error", "No planner initialized. Plan a sequence to target first.")
            return
        
        if hasattr(self, "graph_folder"): # if graph is loaded from file, then save images in folder
            saving_path = os.path.join(self.graph_folder, "assets")
        elif hasattr(self, "planner"): #if graph is planned, then save images in dasg_gym/assets folder -> its where plotly dash need the pics
            self.assets_folder_gym = os.path.join(self.data_directory, "..", "dasg_gym", "envs", "assets")
            saving_path = self.assets_folder_gym
        else:
            raise ValueError("Planner or graph folder not found. Please initialize the planner or set the graph folder.")
        
        # Create directory if it doesn't exist
        os.makedirs(saving_path, exist_ok=True)
        
        self.add_output_line(f"Saving visualization to {saving_path}")
            
        # Save all states # end nodes: {104, 168, 169, 170, 171, 143, 113, 119}
        self.add_output_line(f"Saving images of all states...")
        self.add_output_line(f"{len(self.planner.assembly_states)} states")
        
        for state_idx in range(len(self.planner.assembly_states)):
            #print(f"len(self.planner.assembly_states[state_idx].parts): {len(self.planner.assembly_states[state_idx].parts)}")
            #print(f"state {state_idx}")
            if len(self.planner.assembly_states[state_idx].parts) != 0:
                #print(f" len ungleich 0")
                scene = assembly_to_trimesh_scene(self.planner.assembly_states[state_idx])
            
            if scene is None:
                print(f"Scene for state {state_idx} is None")
                continue
            #print(f"Scene for state {state_idx} is {scene}")
            img = self.generate_image_pyrender(scene)
            img.save(os.path.join(saving_path, f"state_{state_idx}.png"))
            print(f"Saved state {state_idx}")
        
        # Save all actions
        self.add_output_line(f"Saving images of all actions...")
        
        for from_state, to_state, edge_data in self.planner.graph.edges(data=True):
            action = str(edge_data["operation"]).split(" ")[0]
            part = str(edge_data["operation"]).split(" ")[1]
            operation = edge_data["operation"]
            scene = operation_scene(operation, self.planner.assembly_states[from_state])
            img = self.generate_image_pyrender(scene)
            img.save(os.path.join(saving_path, f"{from_state}r_act_{action}_{part}.png"))
            print(f"Saved action from state {from_state} to state {to_state}")

        self.add_output_line(f"All images are saved.")
        self.add_output_line(f"{len(self.planner.graph.edges(data=True))} states")
        messagebox.showinfo("Success", f"All images saved to {saving_path}")

    def show_action(self, from_state, to_state):
        if from_state < len(self.planner.assembly_states) and \
                to_state < len(self.planner.assembly_states):
            self.currently_showing["type"] = "action"
            self.currently_showing["number"] = f'{from_state}_{to_state}'
            self.update_show_lable()
            operation = self.planner.graph.get_edge_data(from_state, to_state)["operation"]
            #print("test")
            scene = operation_scene(operation, self.planner.assembly_states[from_state])
            img = self.generate_image_pyrender(scene)  # Generate the image from the assembly
            self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)
    def next(self):
        if self.path is not None:
            if self.currently_showing["index"] is not None and self.currently_showing["index"]<len(self.path)-1:
                if self.currently_showing["type"] == "action":
                    self.currently_showing["index"] += 1
                    self.show_state(self.path[self.currently_showing["index"]])
                else:
                    self.show_action(self.path[self.currently_showing["index"]], self.path[self.currently_showing["index"]+1])
        else:
            if self.currently_showing["index"] is not None and self.currently_showing["index"] < len(self.planner.assembly_states) - 1:
                self.currently_showing["index"] +=1
                self.show_state(self.currently_showing["index"])

    def prev(self):
        if self.path is not None:
            if self.currently_showing["index"] is not None and self.currently_showing["index"]>0:
                if self.currently_showing["type"] == "action":
                    self.show_state(self.path[self.currently_showing["index"]])
                else:
                    self.currently_showing["index"] -= 1
                    self.show_action(self.path[self.currently_showing["index"]], self.path[self.currently_showing["index"]+1])
            else:
                self.show_state(self.path[self.currently_showing["index"]])
        else:
            if self.currently_showing["index"] is not None and self.currently_showing["index"] > 0:
                self.currently_showing["index"] -=1
                self.show_state(self.currently_showing["index"])
    def update_show_lable(self):
        self.show_lable.config(text=f'{self.currently_showing["type"]} {self.currently_showing["number"]}')

    def run_CMP_demo(self):
        file_name = r"C:\MaH\Data\LimpDissKit\Assemblies\Cable_Dummy.xml"
        self.assembly = Assembly.load_from_xml(file_name)  # Assuming Assembly class has this method
        self.update_assembly_label()
        self.show_image()
        self.add_output_line(f"assembly loaded from xml {file_name}")
        self.load_assembly_in_sim()
        self.reposition_assembly_in_sim(np.array([-0.008619006723165512, 0.020156865939497948, 0.48033544421195984]),
                                        R.from_euler('xyz',
                                                     [8.952208770566008e-05, 0.020835791885540654, 1.8460169871731784],
                                                     degrees=False).as_quat())
        self.look_at_assembly_auto()
        self.perform_allignment()
        self.run_CMP()
        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])
        if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
            self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            # messagebox.showinfo("Waiting for Robot", f"Please execute on Robot")
            self.add_output_line(f"Waiting for Robot, please execute on Robot C/D")
            self.client.connect()
            node_C = self.client.get_node(self.robot_C_status_node_id)
            node_D = self.client.get_node(self.robot_D_status_node_id)
            value_C = node_C.get_value()
            value_D = node_D.get_value()
            while value_C < 1 and value_D < 1:
                time.sleep(0.1)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
            while value_C > 0 or value_D > 0:
                time.sleep(0.1)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
            self.client.disconnect()

    def run_CMP(self):
        # Path to the virtual environment's 'activate.bat' script
        venv_path = r'C:\MaH\Python\venv_collection\LimpDissKit\Scripts\python.exe'

        # Command to run another Python script
        script_path = r"C:\MaH\Python\LimpDissKit\CableMovementPlanner.py"


        # Run the script using the virtual environment’s Python
        result = subprocess.run([venv_path, script_path], capture_output=True, text=True)

        # Print the output (optional)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Return the exit code
        print(f"Script exited with code {result.returncode}")
    def run_CMV_demo_capture(self):
        self.look_at_assembly_auto(distance=0.5,theta=0.0, phi=0.0)
    def run_CMV_demo(self):
        file_frame_sim = filedialog.askopenfilename(filetypes=[("Sim Frame", "*.pkl")], initialdir=self.sim_wd)
        file_name_json = filedialog.askopenfilename(filetypes=[("Gripping File", "*.json")], initialdir=self.data_directory)
        #load sim_frame
        with open(file_frame_sim, 'rb') as f:
            frame_sim = pickle.load(f)
        frame_sim["points"] *= 1000
        view_matrix_transform = np.linalg.inv(np.array(frame_sim["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:, 3] *= 1000
        view_matrix_transform[3, 3] = 1.0
        #load json
        with open(file_name_json, 'r') as f:
            gripping_info = json.load(f)

        grip_point_h = np.append(np.array(gripping_info["grip_point_xyz"]),1.0)
        grip_point_h[1] *= -1 #necessary to transform from the "real" zivid coordinate Frame to the "Sim" zivid coordinate frame
        grip_point_h[2] *= -1
        grip_point = np.dot(view_matrix_transform, grip_point_h)/1000.0
        print(f"grip_point: {grip_point}")
        move_point_1_h = np.append(np.array(gripping_info["move_point_1"]),1.0)
        move_point_1_h[1] *= -1
        move_point_1_h[2] *= -1
        move_point_1 = np.dot(view_matrix_transform, move_point_1_h)/1000.0
        print(f"move_point_1: {move_point_1}")
        move_point_2_h = np.append(np.array(gripping_info["move_point_2"]),1.0)
        move_point_2_h[1] *= -1
        move_point_2_h[2] *= -1
        move_point_2 = np.dot(view_matrix_transform, move_point_2_h)/1000.0
        print(f"move_point_2: {move_point_2}")
        path = [grip_point[:3]+np.array([0,0,0.1]), grip_point[:3], move_point_2[:3], move_point_2[:3]+np.array([0,0,0.1])]
        base_orn = rotation_matrix_from_zy(np.array([0,0,1]), grip_point[:3]-move_point_2[:3])
        orientations = [base_orn for _ in range(len(path))]
        data_to_send = {"request": "execute_action", "type": "cable_gripping_MV", "path": path, "orientations": orientations}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                messagebox.showinfo("Connection successfull", f"Received: {answer}")
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
        #self.perform_allignment()
        """self.run_CMV()
        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])
        self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf"))
        self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf"))
        # messagebox.showinfo("Waiting for Robot", f"Please execute on Robot")
        self.add_output_line(f"Waiting for Robot, please execute on Robot C")
        self.client.connect()
        node_C = self.client.get_node(self.robot_C_status_node_id)
        node_D = self.client.get_node(self.robot_D_status_node_id)
        value_C = node_C.get_value()
        value_D = node_D.get_value()
        while value_C < 1 and value_D < 1:
            time.sleep(0.1)
            value_C = node_C.get_value()
            value_D = node_D.get_value()
        while value_C > 0 or value_D > 0:
            time.sleep(0.1)
            value_C = node_C.get_value()
            value_D = node_D.get_value()
        self.client.disconnect()"""

    def run_plug_demo_capture(self):
        self.look_at_assembly_auto(distance=0.5,theta=np.pi/4, phi=-np.pi/2)
    def run_plug_demo(self):
        file_frame_sim = filedialog.askopenfilename(filetypes=[("Sim Frame", "*.pkl")], initialdir=self.sim_wd)
        file_name_json = filedialog.askopenfilename(filetypes=[("Gripping File", "*.json")], initialdir=self.data_directory)
        #load sim_frame
        with open(file_frame_sim, 'rb') as f:
            frame_sim = pickle.load(f)
        frame_sim["points"] *= 1000
        view_matrix_transform = np.linalg.inv(np.array(frame_sim["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:, 3] *= 1000
        view_matrix_transform[3, 3] = 1.0
        #load json
        with open(file_name_json, 'r') as f:
            gripping_info = json.load(f)

        grip_point_h = np.append(np.array(gripping_info["grip_point_xyz"]),1.0)
        grip_point_h[1] *= -1 #necessary to transform from the "real" zivid coordinate Frame to the "Sim" zivid coordinate frame
        grip_point_h[2] *= -1
        grip_point = np.dot(view_matrix_transform, grip_point_h)[:3]/1000.0
        print(f"grip_point: {grip_point}")
        unplug_direction = np.array(gripping_info["unplug_direction"])
        unplug_direction[1] *= -1
        unplug_direction[2] *= -1
        unplug_direction = np.dot(view_matrix_transform[:3,:3], unplug_direction)
        unplug_direction = unplug_direction/np.linalg.norm(unplug_direction)
        print(f"unplug_direction: {unplug_direction}")


        path = [grip_point+0.1*unplug_direction,
                grip_point,
                grip_point+0.1*unplug_direction,
                grip_point+0.2*unplug_direction]

        target_orn = R.from_euler('xyz',[0,0,-180], degrees=True).as_matrix()
        base_orn = np.dot(rotation_matrix_to_align_z(unplug_direction, target_orn), target_orn)
        orientations = [base_orn for _ in range(len(path))]
        data_to_send = {"request": "execute_action", "type": "cable_gripping_MV", "path": path, "orientations": orientations}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                messagebox.showinfo("Connection successfull", f"Received: {answer}")
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()

    def run_CMV(self):
        # Path to the virtual environment's 'activate.bat' script
        venv_path = r'C:\MaH\Python\venv_collection\LimpDissKit\Scripts\python.exe'

        # Command to run another Python script
        script_path = r"C:\MaH\Python\LimpDissKit\CableMovementVisualizer.py"

        # Run the script using the virtual environment’s Python
        result = subprocess.run([venv_path, script_path], capture_output=True, text=True)

        # Print the output (optional)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Return the exit code
        print(f"Script exited with code {result.returncode}")

    def instanciate_simulation(self):
        # Path to the virtual environment's 'activate.bat' script
        venv_activate = str(resolve_path(".venv/Scripts/activate.bat"))
        script_to_run = str(resolve_path("crc_disassembly_sim/DisassemblySim.py"))

        # Construct the command to activate venv and run the script with arguments
        #command = f'{venv_activate} && python {script_to_run}'

        # Command to open a new Command Prompt window and execute the Python script
        cmd_command = f'start cmd.exe /k "{venv_activate} && python {script_to_run} --server_active=True"'

        # Use subprocess.Popen to run the command
        subprocess.Popen(cmd_command, shell=True)

        # Use subprocess.run to execute the command
        #self.sim_process = subprocess.Popen(command)
        #subprocess.run(command, shell=True)
    def receive_data(self, conn):
        # Receive the size of the pickled data (16 bytes)
        size_data = conn.recv(16)
        size = int.from_bytes(size_data, 'big')
        conn.sendall(b'ACK')
        # Receive the actual data based on the size
        pickled_data = conn.recv(size)
        data = pickle.loads(pickled_data)
        return data

    def check_connection(self):
        data_to_send = {"request": "get_state"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                messagebox.showinfo("Connection successfull", f"Received: {answer}")
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
    def load_assembly_in_sim(self):
        if self.assembly is None:
            self.load_xml()
        print("load_assembly_in_sim")
        if self.assembly.xml_path is None:
            print("load_assembly_in_sim: no xml path")
            self.save_as_xml()
        print(f"load_assembly_in_sim: xml path: {self.assembly.xml_path}")
        """file_types = [('XML files', '*.xml')]
        file_name = filedialog.askopenfilename(filetypes=file_types, initialdir=self.data_directory)"""
        data_to_send = {"request": "load_assembly", "file": self.assembly.xml_path, "use_loading_point": False}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
        #self.capture_image_of_cell()
    def load_assembly_in_sim_op(self):
        if self.assembly is None:
            self.load_xml()
        if self.assembly.xml_path is None:
            self.save_as_xml()
        """file_types = [('XML files', '*.xml')]
        file_name = filedialog.askopenfilename(filetypes=file_types, initialdir=self.data_directory)"""
        data_to_send = {"request": "load_assembly_op", "file": self.assembly.xml_path, "use_loading_point": False}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer}")
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
        #self.capture_image_of_cell()

    def load_state_in_sim(self, state):
        #print(f"state--:  {state}")
        #print(f"self. data_directory: {self.data_directory}")
        
        # Create a Working_dir directory if it doesn't exist
        if hasattr(self, 'graph_folder'):
            working_dir = os.path.join(self.graph_folder)
        else:
            working_dir = os.path.join(self.data_directory, "Working_dir")
        os.makedirs(working_dir, exist_ok=True)
        
        # First check if the file exists directly in the working directory
        direct_path = os.path.join(working_dir, f"state_{state}.xml")
        if os.path.exists(direct_path):
            path = direct_path
        else:
            # Check if there's an assembly_states subdirectory
            assembly_states_dir = os.path.join(working_dir, "assembly_states")
            if os.path.exists(assembly_states_dir):
                # Check if the file exists in the assembly_states subdirectory
                subdir_path = os.path.join(assembly_states_dir, f"state_{state}.xml")
                if os.path.exists(subdir_path):
                    path = subdir_path
                else:
                    raise ValueError(f"Could not find state_{state}.xml in {working_dir} or {assembly_states_dir}")
            else:
                raise ValueError(f"Could not find state_{state}.xml in {working_dir} and assembly_states directory does not exist")
        # Now create the path
        path = os.path.join(working_dir, "assembly_states", f"state_{state}.xml")

        print(f" pfad: {path}")
        
        # Check if state exists
        if hasattr(self, 'graph_folder'):
            self.assembly= Assembly.load_from_xml(path)
            self.update_assembly_label()
            self.show_image()
            self.add_output_line(f"Assembly-state loaded from {path}")
            self.load_assembly_in_sim()
        elif hasattr(self.planner, 'assembly_states'):# or self.planner.assembly_states is not None:
            if state in self.planner.assembly_states.keys():
                self.planner.assembly_states[int(state)].export_full(path)
            else:
                print(f"State '{state}' not found in assembly_states. Available states: {list(self.planner.assembly_states.keys())}")
                # Get the keys of the dictionary
                keys = list(self.planner.assembly_states.keys())
                if keys:  # Make sure there's at least one key
                    # Print the type of the first key
                    print(f"Type of keys in self.planner.assembly_states: {type(keys[0])}")
                    # Also print the actual keys for reference
                    print(f"Available keys: {keys[:10]}")  # Show first 10 keys
                else:
                    print("self.planner.assembly_states is empty")
                # Handle the missing state appropriately
        elif os.path.exists(path):
            print("State will be loaded from file")
        else:
            raise ValueError(f"State {state} not found in planner.assembly_states and file {path} does not exist.")

        
        data_to_send = {"request": "load_assembly", "file": path, "use_loading_point": False}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer}")
                self.add_output_line(f"Connection successfull. Received: {answer}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
        #self.capture_image_of_cell()

    def request_spawn_pose(self):
        data_to_send = {"request": "get_spawn_pose"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                spawn_point, spawn_orient = answer["spawn_point"], answer["spawn_orient"]
                self.add_output_line(f"Spawn point received: {spawn_point}, Orientation: {spawn_orient}")
                return spawn_point, spawn_orient
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to retrieve spawn point: {e}")
                return None, None
            finally:
                s.close()

    def reposition_assembly_in_sim(self, translation, quaternion):
        data_to_send = {"request": "reposition_base", "pos": translation, "orn": quaternion}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Connection sucessfull, Received: {answer['state']}")
                self.perception_module.global_transformation = np.eye(4)
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()

    def allign_assembly_in_sim(self, translation, quaternion):
        data_to_send = {"request": "allign", "pos": translation, "orn": quaternion}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Connection sucessfull, Received: {answer['state']}")
                self.perception_module.global_transformation = np.eye(4)
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()
    def reposition_part_in_sim(self, part_name, translation, quaternion):
        data_to_send = {"request": "reposition_part", "part_name": part_name, "pos": translation, "orn": quaternion}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"{part_name} xyz: {translation}, rpy: {R.from_quat(quaternion).as_euler('xyz', True)} ")
                #self.perception_module.global_transformation = np.eye(4)
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()

    def capture_image_of_cell(self):
        data_to_send = {"request": "capture_image_of_cell"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection successfull", f"Received: {answer['state']}")
                self.add_output_line(f"Connection successfull.Received: {answer['state']}")
                img = Image.open(answer['buffer'])
                self.display_image_on_lable(img,  self.image_label_plan, self.img_plan_size)
            except Exception as e:
                print(f"Something went wrong connecting to the simulation server:\n{e}")
                messagebox.showwarning("Warning", f"Something went wrong connecting to the simulation server:\n{e}")
            finally:
                s.close()

    def rotate_assembly(self):
        data_to_send = {"request": "reposition_asm", "pos_offset": [-1.2, -1.2, 1.5], "orn_offset": [-2*np.pi/3, 0, np.pi/3]}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Connection sucessfull, Received: {answer['state']}")
                self.perception_module.global_transformation = np.eye(4)
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()

        self.add_output_line("Assembly rotation requested")

    def save_data_to_send_to_pkl(self, data_to_send):
        #save the data_to_send to a file in the working directory, build it like a list, if i call this function multiple times, it will append the data to the file
        working_dir = os.path.join(self.data_directory, "Data_to_send")
        os.makedirs(working_dir, exist_ok=True)
        file_path = os.path.join(working_dir, "data_to_send.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = []
        
        # Append new data to the existing list
        existing_data.append(data_to_send)
        
        # Save the updated list back to the file
        with open(file_path, 'wb') as f:
            pickle.dump(existing_data, f)


    def execute_RL_action(self, input_operation):
        action = input_operation.split(" ")[0]
        part = input_operation.split(" ")[1]
        #print(f"self.currentnode: {self.node_history}")

        #print(f"self.g_markov: {self.g_markov}")
        
        # Check if the node exists in the graph
        node_key = self.node_history
        if node_key not in self.g_markov and isinstance(node_key, str) and node_key.isdigit():
            # Try as integer if it's a string representation of a number
            node_key = int(node_key)
            if node_key not in self.g_markov:
                print(f"Node {self.node_history} not found in graph (tried both string and int)")
                return None
        elif node_key not in self.g_markov:
            print(f"Node {self.node_history} not found in graph")
            return None
            
        #print(f"self.g_markov[currentnode]: {self.g_markov[node_key]}")
        key = f"{node_key}_act_{action}_{part}"
        

        if key in self.g_markov:
            
            #print(f"type(self.g_markov[key]): {type(self.g_markov[key])} ---- und len(self.g_markov[key].keys()): {len(self.g_markov[key].keys())}  ")
            if len(self.g_markov[key].keys()) > 1:
                print(f" Multiple actions found for key '{key}'. Selecting first action.")
                first_key, *_ = self.g_markov[key].keys()
                operation = self.g_markov[key][first_key]
            else:
                operation = self.g_markov[key]
            #print(f"action ist {operation}")
            
            
        else: # this happens if we are milling
            # Suche nur im Attribut-Dict des aktuellen Knotens (also self.g_markov[self.node_history])
            node_dict = self.g_markov[self.node_history]
            matching_keys = [k for k in node_dict if k.startswith(key)]
            if matching_keys:
                print(f"Found matching keys: {matching_keys}")
                operation = node_dict[matching_keys[0]]
            else:
                print(f"No matching action found for '{action} {part}' in available keys")
                raise ValueError(f"No matching action found for '{action} {part}' in available keys")
        #print(f"original operation: {operation}, und keys: {operation.keys()}")
        #print(f"original operation['operation']: {operation['operation']},")
        #print(f"und data: {operation['operation'].data}")
    
        print(f"Operation - normal execute_action_RL: {operation}")
    
        data_to_send = {"request": "execute_action", "type": action, "path": operation['operation'].data["path"],
                        "direction": operation['operation'].data["direction"]}
        if action == "gripping":
            data_to_send["part_name"] = part
        elif action == "SafetyRing":
            data_to_send["part_name"] = part
            data_to_send["orientations"] = operation["orientations"]
            data_to_send["grip_widths"] = operation["grip_widths"]
        elif action == "screwing":
            data_to_send["part_name"] = operation['operation'].data['connection'].part_name1
        elif action == "milling":
            data_to_send["part_name"] = operation['operation'].data['connection'].part_name1
        elif action == "general_gripping":
            data_to_send["part_name"] = part
            data_to_send["path"] = operation["path"]
            data_to_send["direction"] = operation["direction"]
            data_to_send["orientations"] = operation["orientations"]
            data_to_send["grip_widths"] = operation["grip_widths"]

        #self.save_data_to_send_to_pkl(data_to_send)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                    #data_to_send = {"request": "get_state"}
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
                return None
            finally:
                s.close()

        return data_to_send
    
    def look_at_part(self, part_name, distance = 0.50, theta = 30.0/180.0*np.pi, phi = -90.0/180.0*np.pi, buffer_frames=False):
        start_time = time.time()
        data_to_send = {"request": "look_at_part",
                        "part_name": part_name,
                        "distance": distance,
                        "theta": theta,
                        "phi": phi}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                print(f"Answering time: {time.time()-start_time:.2f} seconds")
                self.last_frame_from_sim = answer["frame"]
                del self.last_frame_from_sim["img"]
                del self.last_frame_from_sim["depth_img"]
                #del self.last_frame_from_sim["part_array"]
                if buffer_frames:
                    self.sim_frame_buffer.append(copy.deepcopy(self.last_frame_from_sim))
                else:
                    if not os.path.exists(self.sim_wd):
                        os.mkdir(self.sim_wd)
                    num_files = len(os.listdir(self.sim_wd))
                    new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                    with open(new_filename, 'wb') as file:
                        pickle.dump(self.last_frame_from_sim, file)
                print(f"Saving/Buffering time: {time.time()-start_time:.2f} seconds")
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except:
                messagebox.showwarning(f"Warning", "Something went wrong connecting to the simulation server. - looka t part")
            finally:
                s.close()

            #self.last_frame_from_sim["points"][:, 2] *= -1
            #self.last_frame_from_sim["points"][:, 1] *= -1

    def look_at_assembly(self, distance = 0.50, theta = 30.0/180.0*np.pi, phi = -90.0/180.0*np.pi, buffer_frames=False):
        start_time = time.time()
        data_to_send = {"request": "look_at_assembly",
                        "distance": distance,
                        "theta": theta,
                        "phi": phi}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                print(f"Answering time: {time.time()-start_time:.2f} seconds")
                self.last_frame_from_sim = answer["frame"]
                del self.last_frame_from_sim["img"]
                del self.last_frame_from_sim["depth_img"]
                #del self.last_frame_from_sim["part_array"]
                if buffer_frames:
                    self.sim_frame_buffer.append(copy.deepcopy(self.last_frame_from_sim))
                else:
                    if not os.path.exists(self.sim_wd):
                        os.mkdir(self.sim_wd)
                    num_files = len(os.listdir(self.sim_wd))
                    new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                    with open(new_filename, 'wb') as file:
                        pickle.dump(self.last_frame_from_sim, file)
                print(f"Saving/Buffering time: {time.time()-start_time:.2f} seconds")
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except Exception as e:
                messagebox.showwarning(f"Warning", f"Something went wrong connecting to the simulation server. - look at assemlby:\n{e}")
            finally:
                s.close()
    def run_initial_allignment(self):
        self.look_at_assembly_auto(distance = 0.5, theta = 30.0/180.0*np.pi, phi = -90.0/180.0*np.pi, buffer_frames=True)
        self.perception_module.initialize_current_data_objects()
        for frame_sim, frame_zivid in zip(self.sim_frame_buffer, self.zivid_frame_buffer):
            self.perception_module.add_view(frame_sim, frame_zivid)
        self.perception_module.calculate_global_transform()
        if not np.array_equal(self.perception_module.global_transformation, np.eye(4)):
            #display the result
            o3d_pointclouds = [self.perception_module.current_point_cloud_sim.paint_uniform_color(wbk_color_dict["yellow"] / 255),
                               self.perception_module.current_point_cloud_real.paint_uniform_color(wbk_color_dict["blue"] / 255)]
            temp_pcd = copy.deepcopy(self.perception_module.current_point_cloud_sim)
            temp_pcd.transform(self.perception_module.global_transformation)
            temp_pcd.paint_uniform_color(wbk_color_dict["green"] / 255)
            o3d_pointclouds.append(temp_pcd)
            self.display_image_on_lable(self.generate_image_from_pointclouds(o3d_pointclouds, True),
                                        self.image_label_perception, self.img_perception_size)
            # Extract the translation vector
            translation = np.array(self.perception_module.global_transformation[:3, 3])

            # Extract the rotation matrix
            rotation_matrix = np.array(self.perception_module.global_transformation[:3, :3])

            # Convert the rotation matrix to a quaternion
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # Returns (x, y, z, w)
            self.allign_assembly_in_sim(translation, quaternion)


    def run_perception(self, part_name=None, distance = 0.5, theta =  np.radians(30.0), phi =  np.radians(-90.0),
                       delta_theta= np.radians(18.0), delta_phi= np.radians(30.0), num_theta=1, num_phi=7):
    #def run_perception(self, part_name=None, distance = 0.75, theta = 30.0/180.0*np.pi, phi = -90.0/180.0*np.pi, delta_theta=18.0, delta_phi=60.0, num_theta=1, num_phi=4):
        if self.planner is None:
            self.add_output_line("No planner active")
            messagebox.showerror("Error", "No planner active")
            return
        start_time = time.time()
        """if part_name is None:
            part_name = self.planner.assembly_states[self.current_state_number].parts[0].name"""
        #clear data objects
        self.perception_module.initialize_current_data_objects()
        self.perception_module.global_transformation = None
        self.sim_frame_buffer = []
        
        viewpoints = [[0.4, np.radians(90), np.radians(-45)],
                      [0.4, np.radians(90), np.radians(-30)],
                      [0.4, np.radians(120), np.radians(-90)],
                      [0.4, np.radians(120), np.radians(-180)],
                      [0.4, np.radians(60), np.radians(-180)],
                      [0.4, np.radians(60), np.radians(-90)],
                      ]                      

        if not any("Getriebegehaeuse_400_V1_01p" in part.name for part in self.assembly.parts): #if its not the angle grinder, use the viewpoints from the assembly-but may need to be adapted for other assemblies
            for vp in viewpoints:
                self.look_at_assembly(vp[0], vp[1], vp[2], buffer_frames=True)
        else:        
            for vp in viewpoints:
                self.look_at_part("Getriebegehaeuse_400_V1_01p", vp[0], vp[1], vp[2], buffer_frames=True) #specific for angle grinder

        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])
        if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
            self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:

            self.add_output_line(f"Waiting for Robot, please execute on Robot D -runperception")
            self.execute_measure_program(buffer_frames=True)
            for frame_sim, frame_zivid in zip(self.sim_frame_buffer, self.zivid_frame_buffer):
                #save the frames
                num_files = len(os.listdir(self.zivid_wd))
                frame_zivid.save(f"{self.zivid_wd}/{num_files:03d}.zdf")
                num_files = len(os.listdir(self.sim_wd))
                new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                with open(new_filename, 'wb') as file:
                    pickle.dump(frame_sim, file)
                self.perception_module.add_view(frame_sim, frame_zivid) #IMPORTANT TO DO AFTER SAVING DUE TO SCALING --> see github issue



            self.perception_module.current_state = copy.deepcopy(self.planner.assembly_states[self.current_state_number]) #self.current_state_number ist the active node 
            connections_to_remove = []
            for con in self.perception_module.current_state.connections:
                if con.connection_type == "screw":
                    connections_to_remove.append(con)
            for con in connections_to_remove:
                self.perception_module.current_state.remove_connection(con)
            self.add_output_line(f"Starting perception processing...")
            self.perception_module.process_current_state()
            self.add_output_line(f"perception processed in {time.time()-start_time:.2f} seconds")
            print(f"perception processed in {time.time()-start_time:.2f} seconds")
            self.perception_module.show_result

            o3d_pointclouds = self.perception_module.result_o3d(self.perception_module.last_step["result"],
                                                      self.perception_module.last_step["segments_sim"],
                                                      self.perception_module.last_step["point_cloud_real"])
            self.display_image_on_lable(self.generate_image_from_pointclouds(o3d_pointclouds, False),
                                        self.image_label_perception, self.img_perception_size)


            #visualize Result TODO display the result in the GUI
            self.perception_module.show_result(self.perception_module.last_step["result"],
                             self.perception_module.last_step["segments_sim"],
                             self.perception_module.last_step["point_cloud_real"],
                             self.perception_module.calculate_bboxes_for_state(self.perception_module.last_step["segments_sim"],
                                                                               self.perception_module.last_step["state"],
                                                                               self.perception_module.last_step["global_transformation"],
                                                                               self.perception_module.last_step["assembly_base_rotation_matrix"],
                                                                               ),
                             individual_view=False)

            #apply global transformation in case of initial allignment
            if not np.array_equal(self.perception_module.global_transformation, np.eye(4)):
                # Extract the translation vector
                translation = np.array(self.perception_module.global_transformation[:3, 3])

                # Extract the rotation matrix
                rotation_matrix = np.array(self.perception_module.global_transformation[:3, :3])

                # Convert the rotation matrix to a quaternion
                rotation = R.from_matrix(rotation_matrix)
                quaternion = rotation.as_quat()  # Returns (x, y, z, w)
                self.allign_assembly_in_sim(translation, quaternion)
            #determine actual disassembly state by first checking for missing parts
            actual_state = copy.deepcopy(self.planner.assembly_states[self.current_state_number])
            state_changed = False
            missing_parts = []
            for part, res in self.perception_module.last_step["result"].items():
                self.add_output_line(f"Part: {part}, found part: {res['found_part']}, part moved: {res['part_moved']}")
                if not res['found_part']:
                    actual_state.remove_part(actual_state.get_part_by_name(part))
                    missing_parts.append(part)
                    state_changed = True
            #update disassembly target to include missing parts
            for p in missing_parts:
                if p not in self.disassembly_target_part_names:
                    self.disassembly_target_part_names.append(p)
            #reexplore the graph to make sure all possible states are included
            part_list = []
            for part_name in self.disassembly_target_part_names:
                part_list.append(self.assembly.get_part_by_name(part_name))


            
            print(f"self.parts_list: {self.parts_list}")
            # Berechne die Differenz zwischen den beiden Listen (Elemente in self.parts_list, die nicht in part_list sind)
            real_part_list = list(set(self.parts_list) - set([p.name for p in part_list]))
            # Wenn "BG_Gehaeuse_05p" nicht in real_part_list ist, dann füge es hinzu
            if "BG_Gehaeuse_05p" not in real_part_list:
                real_part_list.append("BG_Gehaeuse_05p")


            if hasattr(self, 'dis_graph'):
                # Finde heraus, aus welchem Ordner die dis_graph geladen wurde
                if hasattr(self, "graph_folder") and self.graph_folder is not None:
                    # wenn folder "assembly_states" in folder, then load state_0.xml from assembly_states directory
                    if os.path.exists(os.path.join(self.graph_folder, "assembly_states")):
                        #blalba gehe denordner durch und schaue die xml file durch, bis du eine hast, in der nur die parts sind, die in self.parts_list sind. beachte dabei nicht das part "BG_Gehaeuse_05p"
                        assembly_states_dir = os.path.join(self.graph_folder, "assembly_states")
                        found_state = None
                        for fname in os.listdir(assembly_states_dir):
                            if fname.endswith(".xml"):
                                xml_path = os.path.join(assembly_states_dir, fname)
                                try:
                                    # Efficiently parse the XML to get part names without loading full meshes
                                    tree = ET.parse(xml_path)
                                    root = tree.getroot()
                                    asm_part_names = set()
                                    for part_elem in root.findall(".//part"):
                                        name = part_elem.get("name")
                                        if name:
                                            asm_part_names.add(name)
                                    # Es dürfen keine missing parts in der xml vorkommen (außer das Gehäuse)
                                    if set(asm_part_names) == set(real_part_list):
                                        # Prüfe, ob wirklich keine extra Teile in asm_part_names sind (außer Gehäuse)
                                        # (eigentlich schon durch == abgedeckt, aber zur Sicherheit)
                                        #if all(p in real_part_list for p in asm_part_names):
                                        found_state = fname.replace("state_", "").replace(".xml", "")
                                        self.current_state_number = int(found_state)
                                        print(f"Found matching state: {found_state} in {xml_path}")
                                        self.scan_to_node_automated()
                                        break
                                except Exception as e:
                                    print(f"Error loading {xml_path}: {e}")
                    else:
                        print(f"Assembly states directory not found in graph folder: {self.graph_folder}")
                else:
                    print(f"could not find graph folder.")
            else:
                print(f" no graph loaded. Please load a graph first to explore the assembly states.")
                
                

            """self.planner.explore_all_nodes_with_parts(part_list)

            #if state has changed, try to find state in graph #TODO implement logic if state is not in graph
            if state_changed:
                for i in range(len(self.planner.assembly_states)):
                    if actual_state == self.planner.assembly_states[i]:
                        self.current_state_number = i
                        print(f"found new state(node): {self.current_state_number}")
                        break

            sequences, end_states = self.planner.find_all_sequences_to_target(part_list, self.current_state_number)
            costs = []
            for seq in sequences:
                costs.append(self.planner.calculate_costs_for_sequence(seq))
            print(f'Found {len(sequences)} possible sequences for {len(end_states)} end states:\n {end_states}')
            self.add_output_line(f'Found {len(sequences)} possible sequences for {len(end_states)} end states:\n'
                                 f'Endstates: {end_states}\nSequences: {sequences}\nCosts: {costs}')
            if len(costs) > 0:
                min_index = costs.index(min(costs))
                self.path = copy.deepcopy(sequences[min_index])
                self.show_graph()
                self.update_sequence_lable()"""
            
            # Update Simulation:
            #if state has changed update assembly in sim
            if state_changed:
                self.load_state_in_sim(self.current_state_number)  # TODO logic for updating the current state in the simulation if state is not in graph!
            #update positions of parts in simulation
            for part, res in self.perception_module.last_step["result"].items():
                self.add_output_line(f"Part: {part}, found part: {res['found_part']}, part moved: {res['part_moved']}")
                if res['found_part'] and res['part_moved']:

                    # Extract the translation vector
                    translation = np.array(res['transformation'][:3, 3])

                    # Extract the rotation matrix
                    rotation_matrix = np.array(res['transformation'][:3, :3])

                    # Convert the rotation matrix to a quaternion
                    rotation = R.from_matrix(rotation_matrix)
                    quaternion = rotation.as_quat()  # Returns (x, y, z, w)
                    self.reposition_part_in_sim(part, translation, quaternion)
            return state_changed
    def capture_and_align(self):
        self.take_zivid_image_sim()
        self.take_zivid_image()
        self.perform_allignment()
    def scan_assembly_to_pointcloud(self):
        #ask user for .ply filename
        file_types = [('PLY files', '*.ply')]
        file_name = filedialog.asksaveasfilename(defaultextension=".ply",
                                                    filetypes=file_types,
                                                    initialdir=self.data_directory)

        self.sim_frame_buffer = []
        
        viewpoints = [[0.6, np.radians(40), np.radians(-90)], #adjusted manually for one working pose and assembly
                      [0.6, np.radians(40), np.radians(-30)],
                      [0.6, np.radians(40), np.radians(30)],
                      [0.6, np.radians(40), np.radians(90)],
                      [0.7, np.radians(20), np.radians(90)],
                      [0.7, np.radians(20), np.radians(-90)]]
                      
        for vp in viewpoints:
            self.look_at_assembly(vp[0], vp[1], vp[2], buffer_frames=True)

        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])
        if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
            self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            
            self.add_output_line(f"Waiting for Robot, please execute on Robot D -scan assembly to pointcloud")
            self.execute_measure_program(buffer_frames=True)
            for frame_sim, frame_zivid in zip(self.sim_frame_buffer, self.zivid_frame_buffer):
                # save the frames
                num_files = len(os.listdir(self.zivid_wd))
                frame_zivid.save(f"{self.zivid_wd}/{num_files:03d}.zdf")
                num_files = len(os.listdir(self.sim_wd))
                new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                with open(new_filename, 'wb') as file:
                    pickle.dump(frame_sim, file)
                self.perception_module.add_view(frame_sim,
                                                frame_zivid)  # IMPORTANT TO DO AFTER SAVING DUE TO SCALING --> see github issue
            self.perception_module.save_real_point_cloud(file_name)
            mesh = self.perception_module.generate_mesh()
            o3d.visualization.draw_geometries([mesh])
            # Save it as STL
            o3d.io.write_triangle_mesh(self.data_directory + "/scan/base.stl", mesh)
            base_part = Part("base", self.data_directory + "/scan/base.stl")
            self.assembly = Assembly("scan_assembly", [base_part])
            self.update_assembly_label()
            self.show_image()
            self.perception_module.initialize_current_data_objects()


    def save_g_code(self, folder = None):
        if folder is None:
            folder = resolve_path(self.config['Settings']['gcode_path'])
            os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
        file_name_C = f"{folder}/001_Robot_C.mpf"
        file_name_D = f'{folder}/002_Robot_D.mpf'
        """file_types = [('gcode files', '*.mpf')]
        file_name_C = filedialog.asksaveasfilename(defaultextension=".mpf",
                                                   filetypes=file_types,
                                                   initialdir=r'C:/MaH/Data/gcode')
        file_name_D = filedialog.asksaveasfilename(defaultextension=".mpf",
                                                   filetypes=file_types,
                                                   initialdir=r'C:/MaH/Data/gcode')"""
        data_to_send = {"request": "save_gcode", "file_C": file_name_C, "file_D": file_name_D}
        print(f"sending: {data_to_send}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                print(f"connecting")
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except:
                messagebox.showwarning(f"Warning", "Something went wrong connecting to the simulation server. - save gcode")
            finally:
                s.close()
    def execute_action(self):
        if self.currently_showing["index"] is not None and self.currently_showing["type"] == "action":
            operation = self.planner.graph.get_edge_data(
                self.path[self.currently_showing["index"]],
                self.path[self.currently_showing["index"]+ 1] )["operation"]
            data_to_send = {"request": "execute_action", "type": operation.data["type"], "path": operation.data["path"],
                            "direction": operation.data["direction"]}
            if operation.data["type"] == "gripping":
                data_to_send["part_name"] = operation.data["part_name"]
            elif operation.data["type"] == "SafetyRing":
                data_to_send["part_name"] = operation.data["part_name"]
                data_to_send["orientations"] = operation.data["orientations"]
                data_to_send["grip_widths"] = operation.data["grip_widths"]
            elif operation.data["type"] == "screwing":
                data_to_send["part_name"] = operation.data['connection'].part_name1 #here its the part that is being screwed
            elif operation.data["type"] == "milling":
                data_to_send["part_name"] = operation.data["connection"].part_name1 #here its the part which should be free after milling, not the milled one
            elif operation.data["type"] == "general_gripping":
                data_to_send["part_name"] = operation.data["part_name"]
                data_to_send["path"] = operation.data["path"]
                data_to_send["direction"] = operation.data["direction"]
                data_to_send["orientations"] = operation.data["orientations"]
                data_to_send["grip_widths"] = operation.data["grip_widths"]
                data_to_send["gripper_name"] = operation.data["gripper_name"]
                target_config_name = operation.data["part_name"] + "_storage"
                if target_config_name not in self.mounting_configurations:
                    position = None
                    orientation = None
                else:
                    position = self.mounting_configurations[target_config_name]["position"]
                    orientation = R.from_euler('xyz',
                                               self.mounting_configurations[target_config_name]["orientation"],
                                               degrees=True).as_quat()
                data_to_send["position"] = position
                data_to_send["orientation"] = orientation
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((self.sim_host, self.sim_port))
                     #data_to_send = {"request": "get_state"}
                    s.sendall(pickle.dumps(data_to_send))
                    answer = self.receive_data(s)
                    #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                    self.add_output_line(f"Received: {answer['state']}")
                except:
                    messagebox.showwarning(f"Warning", "Something went wrong connecting to the simulation server. - execute aciton")
                    return None
                finally:
                    s.close()
            if operation.data["type"] == "milling":
                temp = self.path[self.currently_showing["index"]+ 1]
                self.load_state_in_sim(self.path[self.currently_showing["index"]+ 1])
            self.current_state_number = self.path[self.currently_showing["index"]+ 1]
            return data_to_send
        return None

    def place_assembly_on_fixture(self):
        if self.assembly is not None:
            self.look_at_part_auto(self.assembly.parts[0].name)
            self.perform_allignment()
            data_to_send = {"request": "execute_action", "type": "move_assembly", "path": None,
                            "direction": None, "pos": self.mounting_pos, "orn": self.mounting_orn}
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect((self.sim_host, self.sim_port))
                     #data_to_send = {"request": "get_state"}
                    s.sendall(pickle.dumps(data_to_send))
                    answer = self.receive_data(s)
                    #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                    self.add_output_line(f"Received: {answer['state']}")
                except:
                    messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
                    return None
                finally:
                    s.close()
            print("trying to save gcode")
            self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
            gcode_path = resolve_path(self.config['Settings']['gcode_path'])
            if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
                    self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
                # messagebox.showinfo("Waiting for Robot", f"Please execute on Robot")
                self.add_output_line(f"Waiting for Robot, please execute on Robot C/D")
                self.client.connect()
                node_C = self.client.get_node(self.robot_C_status_node_id)
                node_D = self.client.get_node(self.robot_D_status_node_id)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
                while value_C < 1 and value_D < 1:
                    time.sleep(0.1)
                    value_C = node_C.get_value()
                    value_D = node_D.get_value()
                while value_C > 0 or value_D > 0:
                    time.sleep(0.1)
                    value_C = node_C.get_value()
                    value_D = node_D.get_value()
                self.client.disconnect()

        else:
            messagebox.showwarning("Warning", "No assembly loaded.")

    def open_manual_reposition_dialog(self):
        if self.assembly is not None:
            ManualRepositionDialog(self)
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")

    def get_mdp_node(self, node):
        """ input node ist the node in the dasg_graph, which was specified by the zivid scan. then here we should compare the node with the nodes in the MDP graph and return the corresponding MDP node
        we assume to get an int and then we check al the nodes in the mdp which end on "_{node}r" and return the first one that matches
        """
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", "No MDP graph available. Please load or convert a graph first.")
            return
        
        if str(node) == "0" or node ==0: 
            return 0
        
        if node is not None:
            # Find all matching nodes that end with _{node}r or are exactly {node}
            candidates = []
            for mdp_node in self.g_markov.nodes():
                s = str(mdp_node)
                if s.endswith(f"_{str(node)}r"):
                    candidates.append(mdp_node)
                elif s == f"{node}":
                    candidates.append(mdp_node)


            #print(f"Candidates for MDP node for node {node}: {candidates}")
            if candidates:
            # Return the shortest string representation (prefer e.g. 0_3r over 0_8f_3r)
                shortest = min(candidates, key=lambda n: len(str(n)))
            print(f"Found MDP node: {shortest} for node: {node}")
            self.node_history = shortest
            return shortest
        return None

    def look_at_part_dialog(self):
        if self.assembly is not None:
            self.look_at_dialog = SelectPartToLookAtDialog(self)
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")
        print("done")

    def look_at_part_auto(self, part_name = None):
        if self.assembly is not None:
            if part_name is None:
                self.look_at_dialog = SelectPartToLookAtDialog(self)
                self.wait_window(self.look_at_dialog)
            else:
                self.look_at_part(part_name)
            
            self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
            gcode_path = resolve_path(self.config['Settings']['gcode_path'])
            if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
                    self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            
                #messagebox.showinfo("Waiting for Robot", f"Please execute on Robot C")
                self.add_output_line(f"Waiting for Robot, please execute on Robot D - lookt at part auto")
                self.execute_measure_program()
                """self.client.connect()
                node = self.client.get_node(self.robot_C_status_node_id)
                value = node.get_value()
                while value < 2:
                    time.sleep(0.1)
                    value = node.get_value()
                self.client.disconnect()
                self.take_zivid_image()
                #node.set_value(ua.DataValue(ua.Variant(1.0, ua.VariantType.Double)))
                self.add_output_line(f"Image captured")"""
                #self.show_overlay()
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")

    def look_at_assembly_auto(self, distance=0.50, theta=30.0/180.0*np.pi, phi=-90.0/180.0*np.pi, buffer_frames=False):
        if self.assembly is not None:
            if buffer_frames:
                self.sim_frame_buffer = []
            self.look_at_assembly(distance=distance,
                                  theta=theta,
                                  phi=phi,
                                  buffer_frames=buffer_frames)
            self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
            gcode_path = resolve_path(self.config['Settings']['gcode_path'])
            if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
                self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            
                #messagebox.showinfo("Waiting for Robot", f"Please execute on Robot C")
                self.add_output_line(f"Waiting for Robot, please execute on Robot D - look at assembly auto")
                self.execute_measure_program(buffer_frames=buffer_frames)
                """self.client.connect()
                node = self.client.get_node(self.robot_C_status_node_id)
                value = node.get_value()
                while value < 2:
                    time.sleep(0.1)
                    value = node.get_value()
                self.client.disconnect()
                self.take_zivid_image()
                #node.set_value(ua.DataValue(ua.Variant(1.0, ua.VariantType.Double)))
                self.add_output_line(f"Image captured")"""
                #self.show_overlay()
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")
    def execute_action_auto(self):
        #time.sleep(3.0)
        sent_data = self.execute_action() 
        if sent_data is None:
            return
        
        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])

        self.client.connect()
        node_C = self.client.get_node(self.robot_C_status_node_id)
        node_D = self.client.get_node(self.robot_D_status_node_id)
        value_C = node_C.get_value()
        value_D = node_D.get_value()
        while value_C > 0 or value_D > 0:  # wait for both robots to reset to idle state
            time.sleep(0.1)
            value_C = node_C.get_value()
            value_D = node_D.get_value()
            print(f"Waiting for robots to idle; Robot C: {value_C}, Robot D: {value_D}")
        self.client.disconnect()
        if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
            self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            #messagebox.showinfo("Waiting for Robot", f"Please execute on Robot")
            self.add_output_line(f"Waiting for Robot, please execute on Robot D - execute action auto")
            self.client.connect()
            node_C = self.client.get_node(self.robot_C_status_node_id)
            node_D = self.client.get_node(self.robot_D_status_node_id)
            value_C = node_C.get_value()
            value_D = node_D.get_value()
            while value_C < 1 and value_D < 1: #wait for at least one robots to start the action
                time.sleep(0.1)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
            while value_C > 0 or value_D > 0: #wait for both robots to finish the action
                time.sleep(0.1)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
            self.client.disconnect()
            #time.sleep(2.0)
            #self.look_at_part_auto(sent_data["part_name"])
            self.currently_showing["index"] += 1
            self.show_state(self.path[self.currently_showing["index"]])
            self.add_output_line(f"Action executed")


    def execute_sequence_auto(self):
        if len(self.path) > 1:
            self.add_output_line(f"Executing sequence with {len(self.path)} states.")
            self.currently_showing["index"] = 0
            #activate if just initial allignment is to be performed
            #self.run_initial_allignment()

            #LEGACY CODE
            #self.look_at_part_auto(self.assembly.parts[0].name)
            #self.look_at_assembly_auto()
            #self.perform_allignment()--> legacy funciton
            #if self.run_perception(num_phi=1):

            #activate to run full perception
            if self.run_perception(num_phi=7):
                #create message box to inform, that state has changed
                messagebox.showinfo("State changed", "Deviation detected. State has changed, plan was adapted.")
            for i in range(len(self.path)-1):
                self.currently_showing["index"] = i
                self.show_action(self.path[i], self.path[i + 1]) 
                #self.execute_action()
                self.execute_action_auto()
                """if not messagebox.askyesno("DEMO-MODE", "Disassembly successful?"):
                    break"""

    def execute_measure_program(self, buffer_frames=False):
        if buffer_frames:
            self.zivid_frame_buffer=[]
        value = 0
        self.client.connect()
        node = self.client.get_node(self.robot_D_status_node_id)
        value = node.get_value()
        self.add_output_line(f"Waiting for program start")
        while value < 1:
            time.sleep(0.1)
            value = node.get_value()
        self.client.disconnect()
        while value > 0:
            self.client.connect()
            node = self.client.get_node(self.robot_D_status_node_id)
            value = node.get_value()
            self.add_output_line(f"Waiting for robot")
            self.add_output_line(f"value: {value}")
            while 2 > value > 0:
                time.sleep(0.1)
                value = node.get_value()
            self.client.disconnect()
            if value > 1:
                self.take_zivid_image(buffer_frames=buffer_frames)
                self.add_output_line(f"Image captured")
            else:
                break
        self.add_output_line(f"Auto capture finished")

    #### hier dann die ganzen knopf-funktionen hinterlegen also show graph in dash oder convert graph to MDP und sowas... die ganze neue sektion
    def adjust_travel_times(self):
        self.dialog = TravelTimeDialog(self)
    
    def adjust_toolchange_times(self):
        self.dialog = ToolChangeTimeDialog(self)
    
    def adjust_feed_rate(self):
        self.dialog = FeedRateDialog(self)
    
    def adjust_distances_to_assembly(self):
        self.dialog = ApproachingDistancesDialog(self)

    def load_graph(self):
        print(f"starting to load graph from file")
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for loading
        file_name = filedialog.askopenfilename(filetypes=file_types,
                                               initialdir=self.data_directory)
        
        with open(file_name, 'rb') as file:
            self.dis_graph = pickle.load(file)
        print(f"graph: {self.dis_graph}")
        # Print edge information for all edges
        
        self.add_output_line(f"dis_graph {file_name} loaded successfully.")
        
        #gets the assembly from the graph folder
        if file_name is not None:
            self.graph_folder = os.path.dirname(file_name)
            #print(f"folder: {folder}")
            # wenn folder "assembly_states" in folder, then load state_0.xml from assembly_states directory
            if os.path.exists(os.path.join(self.graph_folder, "assembly_states")):
                xml_file = os.path.join(self.graph_folder, "assembly_states", "state_0.xml")
                print(f" file xml: {xml_file}")
                if os.path.exists(xml_file):
                    self.assembly= Assembly.load_from_xml(xml_file)
                    self.update_assembly_label()
                    self.show_image()
                    self.add_output_line(f"Assembly loaded from {xml_file}")
                    self.auto_update_prob_variables()
                    self.parts_list = []  # List of possible parts to be disassembled
                    for part in self.assembly.parts:
                        self.parts_list.append(part.name)
                    print(f"self.parts_list: {self.parts_list}")
                    
                    #self.update_action_durations(self.dis_graph)
                    self.save_action_durations(self.dis_graph)
                    
                    with open("parts_in_asm_raw.json", "w") as json_file:
                        json.dump(self.parts_list, json_file, indent=4)
                    print(f"parts_in_asm_raw saved to json file: {self.parts_list}")
                else:
                    self.add_output_line(f"state_0.xml not found in assembly_states directory")
            else:
                self.add_output_line(f"No assembly_states directory found in {self.graph_folder}")
        
        if os.path.exists(os.path.join(self.graph_folder, "assets")): #copys images to asset folder
                self.asset_folder_graph = os.path.join(self.graph_folder, "assets")
                print(f"asset folder-graph: {self.asset_folder_graph}")
                self.copy_images_to_asset_folder()
        
        genereated_mdp_name = os.path.basename(file_name).replace(".pkl", "_mdp.pkl")
        #print(f"genereated_mdp_name: {genereated_mdp_name}")
        if os.path.exists(os.path.join(self.graph_folder, genereated_mdp_name)): #checks if mdp is also available, then load it as well
            mdp_g = os.path.join(self.graph_folder, genereated_mdp_name)
            print(f"mdp_g: {mdp_g}")
            with open(mdp_g, 'rb') as file:
                self.g_markov = pickle.load(file)
                self.add_output_line(f"MDP-Graph was found and loaded as well: {genereated_mdp_name}.")

            print(f"self.g_markov: {self.g_markov}")
            """# Print edge information for all edges
            print("\nEdge information:")
            for u, v, data in self.g_markov.edges(data=True):
                print(f"Edge from {u} to {v}: {data}")
                print(f" --> data[edge_data][operation]: {data['edge_data']['operation'].data}") # {data['operation'].data} -> da ist z.b. 'operation': <operations.ScrewingOperation object at 0x000002DB45C44FE0> nur als str gespeichert, demnach keine daten dahinter"""
        else:
            print(f"No MDP graph found for this graph. If MDP Graph is needed, please convert the graph.")

    def show_graph_dash(self):
        if (not hasattr(self, 'planner') or self.planner is None) and (not hasattr(self, 'dis_graph') or self.dis_graph is None):
            messagebox.showerror("Error", "No planner initialized. Plan a sequence to target first. Or load Graph")
            return
        print("starting to visualize graph in plotly dash")

        if hasattr(self, 'dis_graph') and self.dis_graph is not None:
            print(f"self.dis_graph: {self.dis_graph.graph}")
            show_in_dash(self.dis_graph, edge_annotation = "y", title="DASG-Graph") # DASG-Graph
        elif hasattr(self, 'planner') and self.planner is not None:
            print(f"self.planner.graph: {self.planner.graph}")
            show_in_dash(self.planner.graph) #MDP-Graph
        
    
    def convert_graph_to_mdp(self):
        if (not hasattr(self, 'dis_graph') or self.dis_graph is None) and (not hasattr(self, 'planner') or self.planner is None):
            messagebox.showerror("Error", "No MDP graph available. Please load or convert a graph first.")
            return
        
        print(f"starting to convert graph into MDP graph")

        _ = self.check_action_succeed_probs()

        if hasattr(self, 'dis_graph') and self.dis_graph is not None: #graph was loaded
            self.g_markov, node_infos, edge_infos = graph_zu_markov_graph(self.dis_graph)
            self.save_mdp_graph_pre()

        else:                       
            print(f"converting generated graph to MDP graph: {self.planner.graph}")
            self.g_markov, node_infos, edge_infos = graph_zu_markov_graph(self.planner.graph)
        print(f"Graph converted to MDP with {len(self.g_markov.nodes())} nodes and {len(self.g_markov.edges())} edges")
    
    def update_action_durations(self, graph):
        """
        Lädt die action_duration.json und speichert die Dauer für alle im MDP-Graphen vorkommenden Aktionen/Teile.
        """
        # Standardpfad zur action_duration.json im Graph-Ordner
        action_duration_path = "action_duration.json"
        if not os.path.exists(action_duration_path):
            self.action_duration = {}
            self.add_output_line(f"action_duration.json nicht gefunden in {self.graph_folder}.")
            return
        with open(action_duration_path, "r") as f:
            duration_data = json.load(f)

        # Sammle alle relevanten Aktionen/Teile aus dem MDP-Graphen
        # Neue Logik: action_duration_dict für alle Operationen im Graphen aufbauen
        action_duration_dict = {}
        if hasattr(graph, 'edges'):
            for edge in graph.edges(data=True):
                op = edge[2].get('operation')
                print(f"operation: {op}")
                if op is not None and not isinstance(op, str):
                    # Prüfe, ob path_duration existiert
                    if hasattr(op, "data") and isinstance(op.data, dict) and "path_duration" in op.data:
                        action_duration_dict[str(op)] = op.data["path_duration"]
                        print(f"Path duration --> {op.data['path_duration']} für Operation {op}")
        self.action_duration = action_duration_dict
        self.add_output_line(f"action_duration für {len(action_duration_dict)} Aktionen geladen.")
        

    def load_mdp_graph(self):
        print(f"starting to load MDP graph from file")
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for loading
        file_name = filedialog.askopenfilename(filetypes=file_types,
                                               initialdir=self.data_directory)
        
        with open(file_name, 'rb') as file:
            self.g_markov = pickle.load(file)
        #print(f"DASG-Graph Knoteninfos: {self.g_markov.nodes.data()}")
        #print(f"DASG-Graph Kanteninfos: {self.g_markov.edges.data()}")
        self.add_output_line(f"MDP Graph {file_name} loaded successfully.")

        #gets the assembly from the graph folder
        if file_name is not None:
            self.graph_folder = os.path.dirname(file_name)
            #print(f"self.graph_folder: {self.graph_folder}")
            # wenn self.graph_folder "assembly_states" in self.graph_folder, then load state_0.xml from assembly_states directory
            if os.path.exists(os.path.join(self.graph_folder, "assembly_states")):
                xml_file = os.path.join(self.graph_folder, "assembly_states", "state_0.xml")
                print(f" file xml: {xml_file}")
                if os.path.exists(xml_file):
                    self.assembly= Assembly.load_from_xml(xml_file)
                    self.update_assembly_label()
                    self.show_image()
                    self.add_output_line(f"Assembly loaded from {xml_file}")
                    
                    self.auto_update_prob_variables()
                    self.check_action_succeed_probs()
                    
                    self.parts_list = []  # List of possible parts to be disassembled
                    for part in self.assembly.parts:
                        self.parts_list.append(part.name)
                    
                    with open("parts_in_asm_raw.json", "w") as json_file:
                        json.dump(self.parts_list, json_file, indent=4)

                    print(f"parts_in_asm_raw saved to json file: {self.parts_list}")
                else:
                    self.add_output_line(f"state_0.xml not found in assembly_states directory")
            else:
                self.add_output_line(f"No assembly_states directory found in {self.graph_folder}")
            
            if os.path.exists(os.path.join(self.graph_folder, "assets")):
                self.asset_folder_graph = os.path.join(self.graph_folder, "assets")
                print(f"asset folder-graph-mdp: {self.asset_folder_graph}")
                self.copy_images_to_asset_folder()
                
            # Check if there's a corresponding DASG graph file (without "_mdp" suffix)
            base_filename = os.path.basename(file_name)
            if base_filename.endswith("_mdp.pkl"):
                original_filename = base_filename.replace("_mdp.pkl", ".pkl")
                original_filepath = os.path.join(self.graph_folder, original_filename)
                
                if os.path.exists(original_filepath):
                    # Load the original DASG graph
                    with open(original_filepath, 'rb') as file:
                        self.dis_graph = pickle.load(file)
                        #self.update_action_durations(self.dis_graph)
                        self.save_action_durations(self.dis_graph)
                    self.add_output_line(f"Original DASG graph loaded from {original_filepath}")
                else:
                    #self.update_action_durations(self.g_markov)
                    self.save_action_durations(self.g_markov)

                

    def save_mdp_graph_pre(self):
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", "No MDP graph available. Please load or convert a graph first.")
            return
        print(f"starting to save MDP graph")
        if hasattr(self, "graph_folder") and self.graph_folder is not None:
            parent_dir_name = os.path.basename(self.graph_folder)
            mdp_graph_filename = f"{parent_dir_name}_mdp.pkl"
            save_graph_as_pkl(self.g_markov, file_dir = self.graph_folder, file_name = mdp_graph_filename)
            self.add_output_line(f"MDP-Graph saved, {self.g_markov}")
            return
        else:
            self.save_reduced_graph()
        
        self.save_mdp_graph_pre()
        
        
    def show_mdp_graph_in_dash(self):
        print(f"starting to visualize MDP graph in plotly dash")
        if hasattr(self, 'g_markov') and self.g_markov is not None:
            show_in_dash(self.g_markov, edge_annotation = "y", title="MDP-Graph")  # #muss noch so gestaltet werden, das hier nicht das fenster blockiert.
        else:
            messagebox.showerror("Error", "No MDP graph available. Please load or convert a graph first.")
        
    #RL Section functions
    def select_target_part(self):
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", "No MDP graph available. Please load or convert a graph first.")
            return
        if not hasattr(self, 'assembly') or self.assembly is None:
            messagebox.showerror("Error", "No assembly loaded. Please load an assembly first.")
            return
        # Create a dialog to select a target part for RL training
        self.target_part = SelectRLTargetPartDialog(self)

    def adjust_action_durations(self):
        self.dialog = ActionDurationDialog(self)
    def adjust_actionsucceed_porbs(self):
        self.dialog = ActionsucceedDialog(self)

    def delete_q_table(self):
        #print(f"deleting q_table: self data directory: {self.data_directory}")
        filepath = os.path.join(self.data_directory, "..", "dasg_gym", "envs", "q_table.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"deleted q_table: {filepath}")
            self.q_table_path = filepath

    def adjust_q_learning_parameters(self):
        self.dialog = QLearningParametersDialog(self)

    def copy_images_to_asset_folder(self):
        #copy all images from the asset_folder to probabilistic_dissasembly_planner\dasg_gym\envs\assets 
        if not os.path.exists(self.asset_folder_graph):
            messagebox.showerror("Error", f"Asset folder {self.asset_folder_graph} does not exist. Please create it first.")
            return
        # Get the path to the assets folder in dasg_gym
        self.assets_folder_gym = os.path.join(self.data_directory, "..", "dasg_gym", "envs", "assets")
        
        #print(f"assets path: {os.path.normpath(self.assets_folder_gym)}")
        if not os.path.exists(self.assets_folder_gym):
            os.makedirs(self.assets_folder_gym)
            return
        # Copy all images from the asset_folder to the self.assets_folder_gym
        for filename in os.listdir(self.asset_folder_graph):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                src_path = os.path.join(self.asset_folder_graph, filename)
                dest_path = os.path.join(self.assets_folder_gym, filename)
                if not os.path.exists(dest_path):
                    shutil.copy(src_path, dest_path)
                    #print(f"Copied {filename} to {os.path.normpath(self.assets_folder_gym)}")
        print("Success", f"Images copied to {os.path.normpath(self.assets_folder_gym)} successfully.")

    def check_action_succeed_probs(self):
        # --- Compare keys in action_succeed_probs.json to values in action_succeed_probabilities_var.json ---
        json_path_var = "action_succeed_probabilities_var.json"
        json_path_probs = os.path.join(os.path.dirname(__file__), "..", "GraphMDPAnalyser_main", "GraphMDPAnalyser", "action_succeed_probs.json")
        try:
            with open(json_path_var, "r") as f:
                action_succeed_var = json.load(f)
            with open(json_path_probs, "r") as f:
                action_succeed_probs = json.load(f)
            # Flatten all values from action_succeed_var
            expected_keys = set()
            for part_dict in action_succeed_var.values():
                for v in part_dict.values():
                    expected_keys.add(v)

            print(f"Expected keys: {expected_keys}")

            actual_keys = set(action_succeed_probs.keys())
            print(f"Actual keys in action_succeed_probs.json: {actual_keys}")
            
            if expected_keys != actual_keys:
                missing_in_probs = expected_keys - actual_keys
                extra_in_probs = actual_keys - expected_keys
                msg = "Mismatch between expected and actual probabilities of non-destructive disassembly actions keys.\n"
                if missing_in_probs:
                    msg += f"Missing in action_succeed_probs.json: {sorted(missing_in_probs)}\n"
                if extra_in_probs:
                    msg += f"Extra in action_succeed_probs.json: {sorted(extra_in_probs)}\n"
                messagebox.showwarning("probabilities of non-destructive disassembly actions Mismatch", msg)
                self.add_output_line(msg)
                # Remove extra keys
                for key in extra_in_probs:
                    del action_succeed_probs[key]
                # Add missing keys with default value 1.0
                for key in missing_in_probs:
                    action_succeed_probs[key] = 1.0
                # Save updated action_succeed_probs.json
                with open(json_path_probs, "w") as f:
                    json.dump(action_succeed_probs, f, indent=4)
                self.add_output_line("action_succeed_probs.json was updated: extras removed, missing added with default value 1.0.")
                return False
            else:
                self.add_output_line("probabilities of non-destructive disassembly actions keys match expected values.")
                return True
        except Exception as e:
            self.add_output_line(f"Error comparing probabilities of non-destructive disassembly actions keys: {e}")
            return False
        # Check if all assembly parts have success probability entries
        

    def train_agent(self):
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", f"No MDP graph available. Please load or convert a graph first.")
            return
        if not hasattr(self, 'target_part') or self.target_part is None:
            messagebox.showerror("Error", f"No target part selected for training. Please select a target part first.")
            return
        
        if not self.check_action_succeed_probs():
            self.add_output_line("probabilities of non-destructive disassembly actions are updated. Click on train agent again.")
            return
        
        print(f"starting to train Q-learning agent for target part: {self.target_part}")
        
        if (hasattr(self, 'planner') or self.planner is not None) and not hasattr(self, 'dis_graph'):
            # Ask user if they want to save state and action images
            save_images = messagebox.askyesno(
                "Save Images", 
                "Do you want to save images of all states and actions?\n\nWarning: This process can take a very long time for large graphs.",
                icon="warning"
            )

            # Save images only if user confirmed
            if save_images:
                self.add_output_line("Saving images of all states and actions. This may take a while...")
                self.save_pics(self.graph_folder)  # saves all images from states and actions
                self.add_output_line("Images saved successfully.")
            else:
                self.add_output_line("Skipped saving images of states and actions.")
 
            #if we generate everything on demand then we need to save the images in the assets folder for the visualization
            #other paths its not required bc the images are already copied there
            
        #messagebox.showinfo("Info", f"Agent will be trained. The hmi will freeze if you selected to show the training in the dash. To unfreeze press  crtl+c in the terminal of the hmi.") @ TODO: uncomment this line
        self.agent_results = train_q_agent(target_part=self.target_part, save_results=True, q_table_path=None, excel_output_dir=None, graph=self.g_markov, show_in_dash=self.show_optimal_path)
        self.q_table_path = self.agent_results["q_table_path"]
        
        self.add_output_line(f"Agent has been trained.")
        self.initialize_RL_infos()
        
    def load_agent(self):
        # Initialize agent_results if it doesn't exist yet
        if not hasattr(self, 'agent_results') or self.agent_results is None:
            self.agent_results = {}
        messagebox.showinfo("Info", f"Select the q_table you want to apply. Make sure, that there are .pkl files 'action_space.pkl' and 'state_node_pairs.pkl' in the same directory as the q_table.pkl file.")
        # Specify the file type for Pickle files
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for loading
        agent_dir = os.path.join(self.data_directory, "..", "dasg_gym", "envs")
        file_name = filedialog.askopenfilename(title="Select Q-Table",
                                               defaultextension=".pkl",
                                               filetypes=file_types,
                                               initialdir=agent_dir)
        # Extract the directory path without the filename
        agent_dir = os.path.dirname(file_name)
        print(f"Directory path: {agent_dir}")

        if file_name:
            self.q_table_path = file_name
            with open(file_name, 'rb') as file:
                self.agent_results['q_table'] = pickle.load(file)
            
            track_state_node_pairs_file = os.path.join(agent_dir, "state_node_pairs.pkl")
            with open(track_state_node_pairs_file, 'rb') as file:
                self.agent_results['track_state_node_pairs'] = pickle.load(file)
            
            action_space_file = os.path.join(agent_dir, "action_space.pkl")
            with open(action_space_file, 'rb') as file:
                self.agent_results['action_space'] = pickle.load(file)

            target_part_file = os.path.join(agent_dir, "target_part.pkl")
            with open(target_part_file, 'rb') as file:
                self.target_part = pickle.load(file)
                #print(f"Target part loaded: {self.target_part}")
            
        #print(f"Q-Table: {self.agent_results['q_table']}")
        #print(f"State-Node-Pairs: {self.agent_results['track_state_node_pairs']}")
        self.create_node_state_pairs(self.agent_results["track_state_node_pairs"])
        self.initialize_RL_infos()
        print(f"------------------------- Agent loaded successfully -------------------------")

    def create_node_state_pairs(self, state_node_pairs):
        self.reverse_lookup = {}
        for key, values in state_node_pairs.items():
            for v in values:
                self.reverse_lookup[v] = key
        #print(f"Created reverse lookup for node-state pairs: {self.reverse_lookup}")
        
    def reset_RL(self):
        self.disassembly_done = False
        self.initialize_RL_infos()

    def get_action_from_id(self, action_id):
        if not hasattr(self, 'agent_results') or self.agent_results is None:
            messagebox.showerror("Error", f"Agent has not been trained properly. Please train/load the agent first.")
            return
        if "action_space" in self.agent_results and action_id < len(self.agent_results["action_space"]):
            action = self.agent_results["action_space"][action_id]
            print(f"Action for ID {action_id} is {action}")
            return action
        else:
            print(f"No action found for ID {action_id}")
            return None
   
    def initialize_RL_infos(self):
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", f"No MDP graph available. Please load or convert a graph first.")
            return
        
        if not hasattr(self, 'agent_results') or self.agent_results is None:
            messagebox.showerror("Error", f"Agent has not been trained properly. Please train/load the agent first.")
            return
        
        self.scan_to_node_automated()
        self.current_node = self.node_history
 
        
        
    def scan_to_node(self):
        print(f"-------------------- Simulate Scanning ------------------")
        
        dialog = InputNodeDialog(self)
        self.wait_window(dialog)
        input_node = dialog.result
        
        if input_node is None:
            print("User cancelled input")
            return
            
        print(f"Input Node (from disassembly graph): {input_node}")
        mdp_node = self.get_mdp_node(input_node)
        print(f"MDP-Node: {mdp_node}")
        self.node_history = mdp_node  
        self.current_node = mdp_node

    def scan_to_node_automated(self):
        if self.current_state_number == None:
            self.current_state_number = 0

        print(f"Input Node (from disassembly graph): {self.current_state_number}")
        mdp_node = self.get_mdp_node(self.current_state_number)
        print(f"MDP-Node: {mdp_node}")
        self.node_history = mdp_node  
        self.current_node = mdp_node



    def get_next_action(self):
        print(f"-------------------- Getting best action for node {self.node_history} ------------------")
        
        if not self.disassembly_done:
            self.optimal_operation, action_id = get_best_action(self.q_table_path, self.node_history, self.agent_results['q_table']) #this imported function needs to be changed if different agent is implemented.
            print(f"Best action for node {self.node_history} is {self.optimal_operation}")
            self.add_output_line(f"Best action for node {self.node_history} is {self.optimal_operation}")

            self.optimal_op_action = self.optimal_operation.split(" ")[0]
            self.optimal_op_part = self.optimal_operation.split(" ")[1]
        else:
            messagebox.showinfo("Done", f"We reched a final state. Restart the hmi for a new disassembly sequence.")


    def run_demo_perception(self, part_name=None, distance = 0.5, theta =  np.radians(30.0), phi =  np.radians(-90.0),
                       delta_theta= np.radians(18.0), delta_phi= np.radians(30.0), num_theta=1, num_phi=7):
        if self.planner is None:
            self.add_output_line("No planner active")
            messagebox.showerror("Error", "No planner active")
            return
        
        start_time = time.time()
        """if part_name is None:
            part_name = self.planner.assembly_states[self.current_state_number].parts[0].name"""
        #clear data objects
        self.perception_module.initialize_current_data_objects()
        self.perception_module.global_transformation = None
        self.sim_frame_buffer = []
        
        viewpoints = [[0.55, np.radians(90), np.radians(-45)],
                      [0.55, np.radians(90), np.radians(-30)],
                      [0.55, np.radians(60), np.radians(-90)],
                      [0.55, np.radians(60), np.radians(-180)],
                      [0.55, np.radians(120), np.radians(-180)],
                      [0.55, np.radians(120), np.radians(-90)]]                      

        if not any("Getriebegehaeuse_400_V1_01p" in part.name for part in self.assembly.parts): #if its not the angle grinder, use the viewpoints from the assembly-but may need to be adapted for other assemblies
            for vp in viewpoints:
                self.look_at_assembly(vp[0], vp[1], vp[2], buffer_frames=True)
        else:        
            for vp in viewpoints:
                self.look_at_part("Getriebegehaeuse_400_V1_01p", vp[0], vp[1], vp[2], buffer_frames=True) #specific for angle grinder

        self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
        gcode_path = resolve_path(self.config['Settings']['gcode_path'])
        if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
            self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:

            self.add_output_line(f"Waiting for Robot, please execute on Robot D -runperception")
            self.execute_measure_program(buffer_frames=True)
            for frame_sim, frame_zivid in zip(self.sim_frame_buffer, self.zivid_frame_buffer):
                #save the frames
                num_files = len(os.listdir(self.zivid_wd))
                frame_zivid.save(f"{self.zivid_wd}/{num_files:03d}.zdf")
                num_files = len(os.listdir(self.sim_wd))
                new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                with open(new_filename, 'wb') as file:
                    pickle.dump(frame_sim, file)
                self.perception_module.add_view(frame_sim, frame_zivid) #IMPORTANT TO DO AFTER SAVING DUE TO SCALING --> see github issue



            #self.perception_module.current_state = copy.deepcopy(self.planner.assembly_states[self.current_state_number]) #self.current_state_number ist the active node 
            #connections_to_remove = []
            """for con in self.perception_module.current_state.connections:
                if con.connection_type == "screw":
                    connections_to_remove.append(con)
            for con in connections_to_remove:
                self.perception_module.current_state.remove_connection(con)
            self.add_output_line(f"Starting perception processing...")
            self.perception_module.process_current_state()
            self.add_output_line(f"perception processed in {time.time()-start_time:.2f} seconds")
            print(f"perception processed in {time.time()-start_time:.2f} seconds")
            self.perception_module.show_result

            o3d_pointclouds = self.perception_module.result_o3d(self.perception_module.last_step["result"],
                                                      self.perception_module.last_step["segments_sim"],
                                                      self.perception_module.last_step["point_cloud_real"])
            self.display_image_on_lable(self.generate_image_from_pointclouds(o3d_pointclouds, False),
                                        self.image_label_perception, self.img_perception_size)


            #visualize Result TODO display the result in the GUI
            self.perception_module.show_result(self.perception_module.last_step["result"],
                             self.perception_module.last_step["segments_sim"],
                             self.perception_module.last_step["point_cloud_real"],
                             self.perception_module.calculate_bboxes_for_state(self.perception_module.last_step["segments_sim"],
                                                                               self.perception_module.last_step["state"],
                                                                               self.perception_module.last_step["global_transformation"],
                                                                               self.perception_module.last_step["assembly_base_rotation_matrix"],
                                                                               ),
                             individual_view=False)"""



    def get_next_action_scan(self):
        self.run_demo_perception()
        self.get_next_action()

    def save_data_to_send(self, input_operation):
        action = input_operation.split(" ")[0]
        part = input_operation.split(" ")[1]
        #print(f"self.currentnode: {self.node_history}")

        #print(f"self.g_markov: {self.g_markov}")
        
        # Check if the node exists in the graph
        node_key = self.node_history
        if node_key not in self.g_markov and isinstance(node_key, str) and node_key.isdigit():
            # Try as integer if it's a string representation of a number
            node_key = int(node_key)
            if node_key not in self.g_markov:
                print(f"Node {self.node_history} not found in graph (tried both string and int)")
                return None
        elif node_key not in self.g_markov:
            print(f"Node {self.node_history} not found in graph")
            return None
            
        #print(f"self.g_markov[currentnode]: {self.g_markov[node_key]}")
        key = f"{node_key}_act_{action}_{part}"
        

        if key in self.g_markov:
            
            #print(f"type(self.g_markov[key]): {type(self.g_markov[key])} ---- und len(self.g_markov[key].keys()): {len(self.g_markov[key].keys())}  ")
            if len(self.g_markov[key].keys()) > 1:
                print(f" Multiple actions found for key '{key}'. Selecting first action.")
                first_key, *_ = self.g_markov[key].keys()
                operation = self.g_markov[key][first_key]
            else:
                operation = self.g_markov[key]
            #print(f"action ist {operation}")
            
            
        else: # this happens if we are milling
            # Suche nur im Attribut-Dict des aktuellen Knotens (also self.g_markov[self.node_history])
            node_dict = self.g_markov[self.node_history]
            matching_keys = [k for k in node_dict if k.startswith(key)]
            if matching_keys:
                print(f"Found matching keys: {matching_keys}")
                operation = node_dict[matching_keys[0]]
            else:
                print(f"No matching action found for '{key}' in available keys")
                raise ValueError(f"No matching action found for '{key}' in available keys")
            
        print(f"Operation, saved to pkl: {operation}")
           
        data_to_send = {"request": "execute_action", "type": action, "path": operation['operation'].data["path"],
                        "direction": operation['operation'].data["direction"]}
        if action == "gripping":
            data_to_send["part_name"] = operation['operation'].data["part_name"] # part
        elif action == "SafetyRing":
            data_to_send["part_name"] = operation['operation'].data["part_name"] # part
            data_to_send["orientations"] = operation["orientations"]
            data_to_send["grip_widths"] = operation["grip_widths"]
        elif action == "screwing":
            data_to_send["part_name"] = operation['operation'].data['connection'].part_name1
        elif action == "milling":
            data_to_send["part_name"] = operation['operation'].data['connection'].part_name1
        elif action == "general_gripping":
            data_to_send["part_name"] = part
            data_to_send["path"] = operation["path"]
            data_to_send["direction"] = operation["direction"]
            data_to_send["orientations"] = operation["orientations"]
            data_to_send["grip_widths"] = operation["grip_widths"]

        self.save_data_to_send_to_pkl(data_to_send)



    def action_result(self):
        self.operation_result = None
        dialog = ActionResultDialog(self, self.g_markov, self.current_node, show_save_checkbox=True, last_save_flag=getattr(self, "save_data_to_send_flag", True))
        self.wait_window(dialog)
        self.operation_result = dialog.result
        self.save_data_to_send_flag = dialog.save_data_to_send_flag  # Merke letzte Einstellung
        print(f"Operation result: {self.operation_result}, save_data_to_send: {self.save_data_to_send_flag}")

        self.add_output_line(f"Operation result: {self.operation_result}")

        if self.save_data_to_send_flag:
            self.save_data_to_send(self.optimal_operation)

        if self.optimal_op_part == self.target_part and self.operation_result:
            print(f"Target part {self.target_part} has been successfully removed.")
            messagebox.showinfo("Success", f"Target part {self.target_part} has been successfully removed.")
            self.disassembly_done = True
        self.get_next_node()
        
        if self.optimal_op_action == "milling" and self.operation_result: # when milling a new assembly is loaded, since new parts are involved
            print(f"loading new assembly in simulation. for node: {self.sim_node}; bc we milled a part")
            #self.load_state_in_sim(self.sim_node) # @TODO: activate this, if all states are loaded-for simulation... deactivated bc of huge graphs
            #self.current_node = self.sim_node
        
        self.node_history = self.current_node
        # Das eigentliche Speichern von data_to_send erfolgt an der Stelle, wo die Daten erzeugt werden (execute_RL_action),
        # dort kann dann self.save_data_to_send_flag abgefragt werden.

        """if self.operation_result:
            self.node_history += f"_{self.current_node}r"  # Append result node with 'r'
        else:
            self.node_history += f"_{self.current_node}f" # Append failure node with 'f'"""
        
        




    def get_next_node(self):
        # Get all possible next nodes
        if "_act_" in str(self.current_node):
            if not hasattr(self.g_markov.graph, 'successors'):
                next_nodes = list(self.g_markov.successors(self.current_node))  # Use the current node directly
            else:
                next_nodes = list(self.g_markov.graph.successors(self.current_node))
        else:
            if not hasattr(self.g_markov.graph, 'successors'):
                if self.node_history == "0":  # Check if it's the string "0" for initial node
                    next_nodes = list(self.g_markov.successors(0))  # Use integer 0 for the initial node
                else:
                    next_nodes = list(self.g_markov.successors(self.node_history))  # Use the string directly
            else:
                if self.node_history == "0":
                    next_nodes = list(self.g_markov.graph.successors(0))  # Use integer 0 for the initial node   
                else:
                    next_nodes = list(self.g_markov.graph.successors(self.node_history))
            
        if next_nodes:
            #print(f"Next nodes available: {next_nodes}")

            # Parse node information to find matching action
            for next_node in next_nodes:
                if "_act_" in str(next_node):
                    action_part = next_node.split("_act_")[1]
                    #print(f"Checking action part: {action_part}")
                    action = action_part.split('_')[0]  # Get the action part before the first underscore
                    #print(f"Action found: {action} ")
                    if action == self.optimal_op_action:
                        part = action_part.split('_', 1)[1]  # Get everything after the first underscore
                        #print(f"Part found: {part}")
                        if part == self.optimal_op_part:
                            #print(f"Found matching action: {action} for part: {part} in node {next_node}")
                            self.current_node = next_node
                            self.get_next_node()  # Recursively call to find the next node
                            break
                        elif part.startswith(self.optimal_op_part):
                            #print(f"Found matching action: {action} for part with similar prefix: {part} in node {next_node}")
                            print(f"Actionnode: {next_node}")
                            self.current_node = next_node
                            self.get_next_node()  # Recursively call to find the next node
                            break
                else:
                    #print(f" next node: {next_node} -- is not an action node, checking for result or failure nodes")
                    next_node = str(next_node)
                    if self.operation_result and next_node.endswith("r"):
                        #print(f"Found result node: {next_node}")
                        
                        self.sim_node = next_node.rsplit("_", 1)[1][:-1]
                        self.current_node = next_node
                        break  
                    
                    elif not self.operation_result and str(next_node) == "-1":
                        print(f"Found failure node: {next_node}")
                        
                        self.sim_node = "-1"
                        self.current_node = next_node
                        print(f"Advanced to failure node {self.current_node}")
                        messagebox.showerror("Error", f"Due to unexpected Action-outcomes we reached an globalre failure node. Further Actions can't be planned or executed automated.")
                        self.disassembly_done = True
                        break
                        
                    elif not self.operation_result and next_node.endswith("f"):
                        print(f"Found failure node: {next_node}")
                        
                        self.sim_node = next_node.rsplit("_", 1)[1][:-1]
                        self.current_node = next_node
                        print(f"Advanced to failure node {self.current_node}")
                        break            
        else:
            print("No successor nodes found. Staying at current node.")

        
    def execute_optimal_action(self):
        if not hasattr(self, 'agent_results') or self.agent_results is None:
            messagebox.showerror("Error", f"Agent has not been trained properly. Please train the agent first.")
            return
        if not hasattr(self, 'current_node') or self.current_node is None:
            messagebox.showerror("Error", f"Current node is not set. Please initialize RL infos first.")
            return
        
        if not hasattr(self, 'g_markov') or self.g_markov is None:
            messagebox.showerror("Error", f"No MDP graph available. Please load or convert a graph first.")
            return
        
        if self.optimal_operation is None:
            self.get_next_action()

        if self.optimal_operation is not None:
            print(f"Executing optimal action: {self.optimal_operation}")
            self.execute_RL_action(self.optimal_operation)
            
        else:
            print(f"No optimal action found for node {self.current_node}")

    def clear_pointcloud_wds(self):
        for file in os.listdir(self.sim_wd):
            print(f"removing {self.sim_wd}/{file}")
            self.add_output_line(f"removing {self.sim_wd}/{file}")
            os.remove(f"{self.sim_wd}/{file}")
        for file in os.listdir(self.zivid_wd):
            print(f"removing {self.zivid_wd}/{file}")
            self.add_output_line(f"removing {self.zivid_wd}/{file}")
            os.remove(f"{self.zivid_wd}/{file}")

    def export_graph(self):
        # Specify the file type for Pickle files
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for saving
        file_name = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                 filetypes=file_types,
                                                 initialdir=self.data_directory)
        self.planner.save_graph_as_pkl(file_name)


    def plan_sequence_to_target_dialog(self):
        if self.planner is not None:
            self.sequence_dialog = PlanSequenceDialog(self)
        else:
            messagebox.showwarning("Warning", "No planner initialized. Create a planner first.")

    def plan_sequence_to_target(self, parts_to_be_disassembled, max_iterations, result_path):
        result = self.planner.plan_path_to_target(parts_to_be_disassembled, max_iterations=max_iterations, result_path=result_path)
        if result >0:
            self.path = copy.deepcopy(self.planner.graph.nodes[result]['cheapest_sequence_to_node'])
            self.path.append(result)
            #self.planner.export_sequence(self.path, "planning_results")
            self.show_graph()
            self.update_sequence_lable()
        else:
            messagebox.showwarning("Warning", "maximum iterations reached")
        # Implement any further actions with the path

    def plan_multiple_sequences_dialog(self):
        if self.planner is not None:
            self.multiple_sequence_dialog = PlanMultipleSequenceDialog(self)
        else:
            messagebox.showwarning("Warning", "No planner initialized. Create a planner first.")

    def visualize_sequence(self):
	
        folder = os.path.join(os.path.dirname(self.data_directory), "dasg_gym", "envs", "assets")
        # Handle directory creation/cleanup
        try:
            if os.path.exists(folder):
                try:
                    # Clean existing directory
                    for item in os.listdir(folder):
                        item_path = os.path.join(folder, item)
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                except Exception:
                    # Create timestamped folder if cleanup fails
                    folder = os.path.join(self.data_directory, 
                                         f"visualization_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            # Create directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)
            
            self.add_output_line(f"Saving visualization to {folder}")
            
            if not self.path:
                messagebox.showwarning("Warning", "No sequence planned. Plan a sequence first.")
                return
                
         
            
            # Function to save current image
            def save_current_image(filepath):
                img_pil = ImageTk.getimage(self.image_label_plan.image)
                img_pil.save(filepath)
            
            # Save all actions
            
            for u, v, data in self.planner.graph.edges(data=True):
                action = str(data["operation"]).split(" ")[0]
                part =  str(data["operation"]).split(" ")[1]
                self.show_action(u, v)
                if u != 0:
                    print(f"saving action from node {u} to {v} as {action}_{part}.png")
                    save_current_image(f"{folder}/{u}r_act_{action}_{part}.png")


			# Save all states
            for node in self.planner.graph.nodes():
                self.show_state(node)
                print(f"saving state {node} as state_{node}.png")
                save_current_image(f"{folder}/state_{node}.png")
           
            self.add_output_line(f"Saved {len(self.planner.graph.nodes())} states and {len(self.planner.graph.edges())} actions")
            
            # Restore to final state view
            self.show_start()
            print("Saving the graphs actions and states as png files-done")
            
        except TypeError as e:
            if "can't multiply sequence by non-int" in str(e):
                messagebox.showwarning("Error", "Error in visualization: check numeric values in your sequence.")
            else:
                raise

    def export_sequence(self):
        # Set the initial directory to the current file's directory
        #initial_dir = os.path.dirname(os.path.abspath(__file__))
        # Open file dialog with initial directory set to the data directory
        folder = filedialog.askdirectory(initialdir=self.data_directory)
        shutil.rmtree(folder)
        os.mkdir(folder)
        if self.path:
            self.planner.export_sequence(self.path, folder)
        else:
            messagebox.showwarning("Warning", "No sequence_planned. Plan a sequence first.")
    def export_states(self):
        # Open file dialog with initial directory set to the data directory
        folder = filedialog.askdirectory(initialdir=self.data_directory)
        shutil.rmtree(folder)
        os.mkdir(folder)
        if self.planner is not None:
            self.planner.export_states(folder)
        else:
            messagebox.showwarning("Warning", "No planner initialized. Create a planner first.")
    def save_as_pkl(self):
        # Specify the file type for Pickle files
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for saving
        file_name = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                 filetypes=file_types,
                                                 initialdir=self.data_directory)

        folder, _ = os.path.splitext(file_name)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        os.mkdir(f"{folder}/stl")
        for op in self.planner.calculated_milling_operations:
            filename = os.path.join(folder, "stl", op.data["part_not_milled"].name + '.stl')
            mesh = op.data["part_not_milled"].mesh
            op.data["part_not_milled"].mesh.export(filename)
            op.data["part_not_milled"].mesh_path = filename
            filename = os.path.join(folder, "stl", op.data["part_milled"].name + '_' + op.data["part_not_milled"].name + 'OP.stl')
            mesh = op.data["part_milled"].mesh
            op.data["part_milled"].mesh.export(filename)
            op.data["part_milled"].mesh_path = filename

        for node in self.planner.graph.nodes:
            self.planner.assembly_states[node].save_part_meshes(f"{folder}/stl")#export_full(f"{folder}/as{node}.xml")
            self.planner.assembly_states[node].delete_part_meshes() #ToDO: could be interesting to save space
        for edge in self.planner.graph.edges:
            print(f"saving edge {edge}: with direction: {self.planner.graph.edges[edge]['operation'].data['direction']}")
        with open(file_name, 'wb') as file:
            pickle.dump(self.planner, file)

        messagebox.showinfo("Success", f"Planner saved")
        self.add_output_line(f"Planner saved to {file_name}")

    def save_action_durations(self, graph):
        action_durations = {}
        for u, v, data in graph.edges(data=True):
            operation = data["operation"]
            
            # Get action type and part name from the operation string
            if hasattr(operation, "__str__"):
                op_string = str(operation)
                parts = op_string.split()
                action_type = parts[0] if len(parts) > 0 else "unknown"
                part_name = parts[1] if len(parts) > 1 else "unknown"
            else:
                action_type = "unknown"
                part_name = "unknown"
            
            # Get path duration if available
            path_duration = None
            if hasattr(operation, "data") and "path_duration" in operation.data:
                path_duration = operation.data["path_duration"]
            
            # Create a unique key for this action
            action_key = f"{action_type}_{part_name}"
            action_durations[action_key] = {
                "action_type": action_type, 
                "part_name": part_name,
                "path_duration": path_duration
            }
        #print(f"action_durations: {action_durations}")

        # Save action durations to JSON
        with open("action_duration.json", "w") as json_file:
            json.dump(action_durations, json_file, indent=4)

        self.add_output_line(f"Action durations saved to action_duration.json")

    def save_reduced_graph(self):
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for saving
        file_name = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                 filetypes=file_types,
                                                 initialdir=self.data_directory)
        
        #print(f"DASG-Graph Knoteninfos: {self.graph.nodes.data()}" )
        #print(f"DASG-Graph Kanteninfos gespeichert:{self.planner.graph.edges.data()}")

        # Create probabilities of non-destructive disassembly actions if they don't exist or update them
        if not hasattr(self.planner, 'action_succeed_probabilities'):
            self.planner.action_succeed_probabilities = {}

        # Generate probabilities of non-destructive disassembly actions for each part in the assembly
        for part in self.planner.assembly_states[0].parts:
            part_name = part.name
            if part_name not in self.planner.action_succeed_probabilities:
                if "screw" in part_name[:10] or "Screw" in part_name[:10] or "Schraube" in part_name[:10] or "schraube" in part_name[:10]:
                    self.planner.action_succeed_probabilities[part_name] = {
                        "screwing": f"sucprob__{part_name}__screwing", "milling": f"sucprob__{part_name}__milling"    # Default success probability for milling
                    }
                else: 
                    self.planner.action_succeed_probabilities[part_name] = {
                        "gripping": f"sucprob__{part_name}__gripping", "milling": f"sucprob__{part_name}__milling"    # Default success probability for milling
                    }
        self.save_action_durations(self.planner.graph)
                
        with open("action_succeed_probabilities_var.json", "w") as json_file:
            json.dump(self.planner.action_succeed_probabilities, json_file, indent=4)
        
        #for edge in self.planner.graph.edges:
            #print(f"saving edge {edge}: with direction: {self.planner.graph.edges[edge]['operation'].data['direction']}")
        
        
        # Strip .pkl extension if present
        base_filename, _ = os.path.splitext(file_name)
        self.planner.export_states(base_filename)
        
        # Create a subfolder with the base filename
        self.graph_folder = os.path.splitext(file_name)[0]
        if not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)

        # Create the new filepath inside the subfolder
        base_filename = os.path.basename(os.path.splitext(file_name)[0])
        new_file_path = os.path.join(self.graph_folder, f"{base_filename}.pkl")

        with open(new_file_path, 'wb') as file:
            pickle.dump(self.planner.graph, file)
        messagebox.showinfo("Success", f"Graph saved to {new_file_path}")

        # Ask user if they want to save state and action images
        save_images = messagebox.askyesno(
            "Save Images", 
            "Do you want to save images of all states and actions?\n\nWarning: This process can take a very long time for large graphs.",
            icon="warning"
        )

        # Save images only if user confirmed
        if save_images:
            self.add_output_line("Saving images of all states and actions. This may take a while...")
            self.save_pics(self.graph_folder)  # saves all images from states and actions
            self.add_output_line("Images saved successfully.")
        else:
            self.add_output_line("Skipped saving images of states and actions.")


    
    def load_from_pkl(self):
        # Specify the file type for Pickle files
        file_types = [('Pickle files', '*.pkl'), ('Pickle files', '*.pickle')]
        # Ask the user to select a file name for saving
        file_name = filedialog.askopenfilename(defaultextension=".pkl",
                                                 filetypes=file_types,
                                                 initialdir=self.data_directory)

        with open(file_name, 'rb') as file:
            self.planner = pickle.load(file)

        if not hasattr(self.planner, 'cost_safety_ring'):
            self.planner.cost_safety_ring = -1

        folder, _ = os.path.splitext(file_name)
        source_folder = f"{folder}/stl"
        for file in filter(os.path.isfile, (os.path.join(source_folder, f) for f in os.listdir(source_folder))):
            shutil.copy2(file, self.planner.working_dir)

        self.assembly = self.planner.assembly_states[0]
        self.update_assembly_label()
        self.current_state_number = 0
        self.update_planner_parameters_label()
        self.show_image()

    def plan_full_graph_dialog(self):
        if self.planner is not None:
            self.graph_dialog = PlanGraphDialog(self)
        else:
            messagebox.showwarning("Warning", "Sequence planned. Plan a sequenc to a specific target first.")

    def get_sequence_to_target_dialog(self):
        if self.planner is not None:
            self.sequence_to_target_dialog = GetSequenceToStateDialog(self)
        else:
            messagebox.showwarning("Warning", "Sequence planned. Plan a sequenc to a specific target first.")

    def print_end_nodes(self):
        if self.planner is not None:
            self.add_output_line(self.planner.end_nodes())
        else:
            messagebox.showwarning("Warning", "No planner initialized. Create a planner first.")



    def set_view_parameters_dialog(self):
        self.view_parameters_dialog = ViewParametersDialog(self)

    def set_view_parameters(self, angle_X, angle_Y, angle_Z,distance):
        self.view_angle_X = angle_X
        self.view_angle_Y = angle_Y
        self.view_angle_Z = angle_Z
        self.view_distance = distance
        self.show_image()



    def create_planner_dialog(self):
        if self.assembly is not None:
            if self.planner is not None:
                t_assembly = copy.deepcopy(self.assembly)
                del self.assembly
                del self.planner
                self.path = None
                self.assembly = t_assembly
            self.planner_dialog = PlannerDialog(self)
        else:
            messagebox.showwarning("Warning", "No assembly loaded. Load an assembly first.")

    def create_planner(self, cost_milling, cost_screwing, cost_gripping, cost_safety_ring, mill_diameter, path_to_working_dir):
        # Assuming the HierarchicalDisassemblySequencePlanner class and its constructor
        self.planner = HierarchicalDisassemblySequencePlanner(
            self.assembly, cost_milling=cost_milling, cost_screwing=cost_screwing,
            cost_gripping=cost_gripping, cost_safety_ring=cost_safety_ring, mill_diameter=mill_diameter, path_to_working_dir=path_to_working_dir,
            available_screwdrivers=self.available_screwdrivers)
        self.update_planner_parameters_label()
        self.current_state_number = 0
        self.add_output_line(f"planner initialized")

    def update_planner_parameters_label(self):
        # Update the label with the current planner parameters
        if self.planner:
            params_text = f"Cost Milling: {self.planner.cost_milling}, Cost Screwing: {self.planner.cost_screwing}, "
            params_text += f"Cost Gripping: {self.planner.cost_gripping}, Cost SR: {self.planner.cost_safety_ring}"
            params_text += f"Mill Diameter: {self.planner.mill_diameter}"
            self.planner_parameters_label.config(text=f"Planner Parameters: {params_text}")
        else:
            self.planner_parameters_label.config(text="Planner Parameters: None")

    def update_sequence_lable(self):
        # Update the label with the current sequence
        if self.path:
            params_text = f'{self.path}'
            self.planned_sequence_lable.config(text=f"Current sequence: {params_text}")
        else:
            self.planned_sequence_lable.config(text="Current sequence: None")
    def save_graph_png(self):
        fig = self.planner.plot_graph()
        file_types = [('PNG files', '*.png')]
        file_name = filedialog.asksaveasfilename(defaultextension=".png", filetypes=file_types,
                                                 initialdir=self.data_directory)
        fig.savefig(file_name)

    def show_graph(self):
        if self.path:
            fig = self.planner.plot_graph(self.path)
        else:
            fig = self.planner.plot_graph()

        # Create an in-memory file object for the figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Use PIL to open the image from the in-memory file object
        img = Image.open(buf)
        #img_tk = ImageTk.PhotoImage(img)

        # Use the existing display_image() function to display the image
        #self.display_image(img_tk)
        self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)

        # Close the buffer
        buf.close()

    def show_DOF_round(self):
        if self.path is not None:
            fig = self.planner.assembly_states[self.path[self.currently_showing["index"]]].generate_surface_plot_round()
        else:
            fig = self.planner.assembly_states[self.currently_showing["index"]].generate_surface_plot_round()

        # Create an in-memory file object for the figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Use PIL to open the image from the in-memory file object
        img = Image.open(buf)
        #img_tk = ImageTk.PhotoImage(img)

        # Use the existing display_image() function to display the image
        #self.display_image(img_tk)
        self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)

        # Close the buffer
        buf.close()

    def show_DOF_square(self):
        if self.path is not None:
            fig = self.planner.assembly_states[self.path[self.currently_showing["index"]]].generate_surface_plot()
        else:
            fig = self.planner.assembly_states[self.currently_showing["index"]].generate_surface_plot()

        # Create an in-memory file object for the figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Use PIL to open the image from the in-memory file object
        img = Image.open(buf)
        #img_tk = ImageTk.PhotoImage(img)

        # Use the existing display_image() function to display the image
        #self.display_image(img_tk)
        self.display_image_on_lable(img, self.image_label_plan, self.img_plan_size)

        # Close the buffer
        buf.close()



    def show_disassemblability(self):
        possible_operations = []
        if self.path is not None:
            index = self.path[self.currently_showing["index"]]
        else:
            index = self.currently_showing["index"]
        for succ in self.planner.graph.successors(index):
            possible_operations.append(self.planner.graph.get_edge_data(index, succ)["operation"])
        removable_parts = [po.__str__().split()[1] for po in possible_operations]
        red_length = 15
        yellow_length = 30
        green_length = 45
        for part in self.planner.assembly_states[index].parts:
            if part.name in removable_parts:
                dof = len(self.planner.assembly_states[index].get_part_dof(part))
                part.mesh.visual.vertex_colors = wbk_color_dict["green"]
                """for seg in segments:
                    length = seg.length
                    print(length)
                    length = min(length, green_length)
                    length = max(length, red_length)
                    if length <= yellow_length:
                        fact = (length - red_length) / (yellow_length - red_length)
                        color = wbk_color_dict["red"] + (wbk_color_dict["yellow"] - wbk_color_dict["red"]) * fact
                    else:
                        fact = (length - yellow_length) / (green_length - yellow_length)
                        color = wbk_color_dict["yellow"] + (wbk_color_dict["green"] - wbk_color_dict["yellow"]) * fact
                    if len(seg.entities) > 0:
                        print(color)
                        seg.colors = [color / 255.0 for _ in range(len(seg.entities))]"""
            else:
                part.mesh.visual.vertex_colors = wbk_color_dict["red"]
        self.planner.assembly_states[index].show()
        for part in self.planner.assembly_states[index].parts:
            part.mesh.visual.vertex_colors = wbk_color_dict["grey"]
        print("stop")


    def load_parts_dialog(self):
        self.dialog = LoadPartsDialog(self)

    def load_parts(self, name, depth):
        # Set the initial directory to the current file's directory
        initial_dir = self.data_directory
        #folder = filedialog.Directory(initialdir=initial_dir)
        # QApplication benötigt ein QApplication-Objekt
        app = QApplication([])

        # Ordnerauswahldialog
        folder = QFileDialog.getExistingDirectory(None, "Wähle einen Ordner")
        print("folder:", folder)
        #folder = tk.simpledialog.askstring("Ordnerpfad eingeben", "Pfad zum Ordner eingeben:")
        if folder and os.path.isdir(folder):
            print("Ordner gewählt:", folder)
        
        #folder = filedialog.askdirectory(parent=self.master, initialdir=initial_dir)
        if folder:  # Proceed only if a folder is selected
            self.parts = load_parts_from_folder(folder, depth, file_endings=('.stl', '.obj'))
            print(f"type of parts: {type(self.parts[0])}")
            self.assembly = create_assembly_object(self.parts, name, self.contact_tollerance, self.dot_product_threshold)
            self.update_assembly_label()
            self.show_image()

    def update_assembly_label(self):
        # Update the assembly label with the name of the current assembly
        if self.assembly:
            self.assembly_label.config(text=f"Current Assembly: {self.assembly.name}")
        else:
            self.assembly_label.config(text="Current Assembly: None")

    def auto_update_prob_variables(self):
        # Fill action_succeed_probabilities_var.json with expected keys for all parts
        json_path = "action_succeed_probabilities_var.json"
        action_suc_var = {}

        current_part_names = set()
        if self.assembly and hasattr(self.assembly, 'parts'):
            #print(f"self.assembly.parts: {self.assembly.parts}")
            for part in self.assembly.parts.values():
                part_name = part.name
                current_part_names.add(part_name)
                if ("screw" in part_name[:10]) or ("Screw" in part_name[:10]) or ("Schraube" in part_name[:10]) or ("schraube" in part_name[:10]):
                    expected = {"screwing": f"sucprob__{part_name}__screwing", "milling": f"sucprob__{part_name}__milling"}
                else:
                    expected = {"gripping": f"sucprob__{part_name}__gripping", "milling": f"sucprob__{part_name}__milling"}
                action_suc_var[part_name] = expected

        with open(json_path, "w") as json_file:
            json.dump(action_suc_var, json_file, indent=4)
        self.add_output_line("action_succeed_probabilities_var.json wurde neu aus der aktuellen Assembly erzeugt.")
            
    def load_xml(self):
        file_types = [('XML files', '*.xml')]
        file_name = filedialog.askopenfilename(filetypes=file_types, initialdir=self.data_directory)

        if file_name:
            try:
                self.assembly = Assembly.load_from_xml(file_name)  # Assuming Assembly class has this method
                self.update_assembly_label()
                self.show_image()
                self.add_output_line(f"assembly loaded from xml {file_name}")
                for part in self.assembly.parts.values():
                    self.parts_list.append(part.name)

                self.auto_update_prob_variables()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load assembly: {e}")


    def save_as_xml(self):
        if self.assembly:
            # Define file types and ask for a file name
            file_types = [('XML files', '*.xml')]
            file_name = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=file_types,
                                                     initialdir=self.data_directory)

            if file_name:
                self.assembly.export_full(file_name)
                messagebox.showinfo("Success", f"Assembly saved as {file_name}")
            else:
                messagebox.showwarning("Save Cancelled", "Save operation cancelled")
    def rename_parts(self):
        file_types = [('text files', '*.txt')]
        file_name = filedialog.askopenfilename(defaultextension=".txt", filetypes=file_types,
                                                 initialdir=self.data_directory)
        if file_name:
            self.assembly.rename_parts(file_name)
            messagebox.showinfo("Success", f"Parts renamed")
        else:
            messagebox.showwarning("Rename Cancelled", "No file selected.")

    def add_relation(self):
        if self.assembly:
            self.dialog = AddRelationDialog(self)
    def add_handlingpoint(self):
        if self.planner is None:
            messagebox.showwarning("Warning", "No planner initialized.")
            return
        self.dialog = SelectPartDialog(self, self.add_handlingpoint_in_pointcloud)

    def select_part_to_remove_handlingOPs(self):
        if self.planner is None:
            messagebox.showwarning("Warning", "No planner initialized.")
            return
        self.dialog = SelectPartDialog(self, self.remove_handlingpoints_of_part)

    def remove_handlingpoints_of_part(self, part_name):
        print(f"number of defined handling operations: {len(self.planner.defined_gripping_operations)}")
        for op in self.planner.defined_gripping_operations:
            if op.data["part_name"] == part_name:
                self.planner.defined_gripping_operations.remove(op)
        print(f"number of defined handling operations: {len(self.planner.defined_gripping_operations)}")

    def export_gripping_ops(self):
        if self.planner is None:
            messagebox.showwarning("Warning", "No planner initialized.")
            return
        file_path = os.path.join(self.operations_folder,self.planner.assembly_states[0].name + "_gripping_operations.pkl")
        self.planner.export_gripping_operations(file_path)
        self.add_output_line(f"Gripping operations exported to {file_path}")

    def import_gripping_ops(self):
        if self.planner is None:
            messagebox.showwarning("Warning", "No planner initialized.")
            return
        file_path = filedialog.askopenfilename(defaultextension=".pkl",
                                               filetypes=[("Pickle files", "*.pkl")],
                                               initialdir=f"{self.operations_folder}")
        self.planner.import_gripping_operations(file_path)
        self.add_output_line(f"Gripping operations imported from {file_path}")

    def add_screw_in_pointcloud(self):
        if self.assembly is None:
            messagebox.showwarning("Warning", "No assembly loaded.")
            return
        pcd = self.assembly.parts[0].mesh.as_open3d.sample_points_uniformly(number_of_points=1000000)
        pcd.estimate_normals()
        self.add_output_line("please pick screw heads")
        screw_points_idx = pick_points(pcd)
        screw_points = np.asarray(pcd.points)[screw_points_idx]
        self.add_output_line("please pick three points to define the direction of the screws")
        plane_points_idx = pick_points(pcd)
        plane_points = np.asarray(pcd.points)[plane_points_idx]
        normal = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
        normal = normal / np.linalg.norm(normal)
        for i in range(len(screw_points)):
            screw_mesh = create_screw_dummy(6.0, 30.0, screw_points[i], normal)
            screw_mesh.export(f"{self.data_directory}/scan/screw_{i}.stl")
            screw = Part(f"screw_{i}", f"{self.data_directory}/scan/screw_{i}.stl")
            self.assembly.add_part(screw)
            self.assembly.add_connection(Screw_Connection(part_name1=screw.name, part_name2=self.assembly.parts[0].name, screw_head="T6"))
            contact_normals = generate_orthogonal_vectors(normal)
            contact_normals = np.vstack([contact_normals, normal])
            contact_points = np.array([screw_points[i] for _ in range(len(contact_normals))])
            distances = np.zeros(len(contact_normals))
            self.assembly.add_connection(Geometric_Connection(part_name1=screw.name,
                                                              part_name2=self.assembly.parts[0].name,
                                                              points=contact_points,
                                                              normals=contact_normals,
                                                              distances=distances))
        scene = assembly_to_trimesh_scene(self.assembly)
        scene.show()

    def add_handlingpoint_in_pointcloud(self, part_name):
        #pcd = self.assembly.get_part_by_name(part_name).mesh.as_open3d.sample_points_uniformly(number_of_points=2500000)
        pcd = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(self.assembly.get_part_by_name(part_name).mesh.vertices), triangles = o3d.utility.Vector3iVector(self.assembly.get_part_by_name(part_name).mesh.faces)).sample_points_uniformly(number_of_points=2500000)


        pcd.estimate_normals()
        gripper_name = simpledialog.askstring("Input", "Please enter the gripper name:", parent=root)
        if not gripper_name:
            gripper_name = "default_gripper"
        self.add_output_line("please pick two grip points")
        grip_points_idx = pick_points(pcd)
        grip_points = np.asarray(pcd.points)[grip_points_idx]
        self.add_output_line("please pick three points to define the direction of movement")
        plane_points_idx = pick_points(pcd)
        plane_points = np.asarray(pcd.points)[plane_points_idx]
        normal = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
        normal = normal / np.linalg.norm(normal)
        print(normal)
        if gripper_name == "DLG":
            approach_direction_2 = grip_points[1] - grip_points[0]
            approach_direction_2 = approach_direction_2 / np.linalg.norm(approach_direction_2)
            point_approach_2 = grip_points[0] - approach_direction_2 * 5.0
            point_approach_1 = point_approach_2 + normal * 250
            point_lift = grip_points[0] + normal * 50.0
            path = [point_approach_1, point_approach_2, grip_points[0], grip_points[0], point_lift]
            orientation = rotation_matrix_from_zx(normal, approach_direction_2)
            orientations = [orientation for _ in range(len(path))]
            grip_width = np.linalg.norm(grip_points[0] - grip_points[1])
            grip_widths = [grip_width for _ in range(len(path))]
            grip_widths[0] = grip_width + 20.0
            grip_widths[1] = grip_width + 20.0
            grip_widths[2] = grip_width + 20.0
        elif gripper_name == "SRG":
            point_grip = (grip_points[0] + grip_points[1])/2
            point_approach = point_grip + normal * 200.0
            point_lift = point_grip + normal * 200.0
            path = [point_approach, point_grip, point_lift]
            orientation = rotation_matrix_from_zx(normal, grip_points[1] - grip_points[0])
            orientations = [orientation for _ in range(len(path))]
            grip_width = np.linalg.norm(grip_points[0] - grip_points[1])
            grip_widths = [grip_width for _ in range(len(path))]
            grip_widths[0] = grip_width + 5.0
        else:
            point_grip = (grip_points[0] + grip_points[1])/2
            point_approach = point_grip + normal * 500.0
            point_lift = point_grip + normal * 500.0
            path = [point_approach, point_grip, point_lift]
            if gripper_name == "SRG":
                orientation = rotation_matrix_from_zx(normal, grip_points[1] - grip_points[0])
            else:
                orientation = rotation_matrix_from_zx(normal, grip_points[1] - grip_points[0])
            orientations = [orientation for _ in range(len(path))]
            grip_width = np.linalg.norm(grip_points[0] - grip_points[1])
            grip_widths = [grip_width for _ in range(len(path))]
            grip_widths[0] = grip_width + 100.0
        op_data = {"part_name": part_name, "path": path, "orientations": orientations, "grip_widths": grip_widths,
                   "direction": normal, "gripper_name": gripper_name}
        handling_op = GeneralGripping(op_data)
        handling_op.scene(self.assembly).show()
        self.planner.defined_gripping_operations.append(handling_op)
    def updat_robots_in_sim(self):
        data_to_send = {"request": "update_robots"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                # data_to_send = {"request": "get_state"}
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
                return None
            finally:
                s.close()
    def take_zivid_image_sim(self):
        data_to_send = {"request": "get_zivid_image"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                # data_to_send = {"request": "get_state"}
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                self.last_frame_from_sim = answer["frame"]
                del self.last_frame_from_sim["img"]
                del self.last_frame_from_sim["depth_img"]
                #del self.last_frame_from_sim["part_array"]
                if not os.path.exists(self.sim_wd):
                    os.mkdir(self.sim_wd)
                num_files = len(os.listdir(self.sim_wd))
                new_filename = f"{self.sim_wd}/{num_files:03d}.pkl"
                with open(new_filename, 'wb') as file:
                    pickle.dump(self.last_frame_from_sim, file)
                self.add_output_line("Zivid image taken and saved to simulation working directory.")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
                return None
            finally:
                s.close()
    def lift_assembly(self):
        if self.planner is None:
            messagebox.showwarning("Warning", "No planner initialized.")
            return
        self.dialog = SelectPartDialog(self)
        selected = self.dialog.show()
        operation = None
        for op in self.planner.defined_gripping_operations:
            if op.data["part_name"] == selected:
                operation = op
                break
        if operation is None:
            messagebox.showwarning("Warning", "No fitting operation defined.")
            return
        self.dialog = SelectMountingConfigurationDialog(self)
        (selected_target_config, open_fixture, close_fixture) = self.dialog.show()
        if selected_target_config is None:
            position = None
            orientation = None
        else:
            position = self.mounting_configurations[selected_target_config]["position"]
            orientation = R.from_euler('xyz', self.mounting_configurations[selected_target_config]["orientation"], degrees=True).as_quat()

        data_to_send = {"request": "execute_action", "type": "assembly_handling",
                        "part_name": operation.data["part_name"],
                        "path": operation.data["path"],
                        "direction": operation.data["direction"],
                        "orientations" : operation.data["orientations"],
                        "grip_widths" : operation.data["grip_widths"],
                        "gripper_name": operation.data["gripper_name"],
                        "position": position,
                        "orientation": orientation,
                        "open_fixture": open_fixture,
                        "close_fixture": close_fixture,}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                # data_to_send = {"request": "get_state"}
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                # messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Received: {answer['state']}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
                return None
            finally:
                s.close()
        return data_to_send

    def auto_move_assembly(self):
        if self.lift_assembly() is not None:
            self.save_g_code(resolve_path(self.config['Settings']['gcode_path']))
            gcode_path = resolve_path(self.config['Settings']['gcode_path'])
            if self.send_nc_file(os.path.join(gcode_path, "001_Robot_C.mpf")) is not None and \
                self.send_nc_file(os.path.join(gcode_path, "002_Robot_D.mpf")) is not None:
            
                # messagebox.showinfo("Waiting for Robot", f"Please execute on Robot")
                self.add_output_line(f"Waiting for Robot, please execute on Robot C/D-auto move asm")
                self.client.connect()
                node_C = self.client.get_node(self.robot_C_status_node_id)
                node_D = self.client.get_node(self.robot_D_status_node_id)
                value_C = node_C.get_value()
                value_D = node_D.get_value()
                while value_C < 1 and value_D < 1:
                    time.sleep(0.1)
                    value_C = node_C.get_value()
                    value_D = node_D.get_value()
                while value_C > 0 or value_D > 0:
                    time.sleep(0.1)
                    value_C = node_C.get_value()
                    value_D = node_D.get_value()
                self.client.disconnect()
                self.add_output_line(f"Action executed")



    def add_mounting_configuration(self):
        if self.assembly is None:
            messagebox.showwarning("Warning", "No assembly loaded.")
            return
        self.dialog = MountingConfigDialog(self)

    def save_mounting_configurations(self):
        # Create XML root
        root = ET.Element("MountingConfigurations")

        # Rebuild XML from current state
        for config_name, values in self.mounting_configurations.items():
            config = ET.SubElement(root, "Configuration", name=config_name)
            ET.SubElement(config, "Position").text = ",".join(map(str, values['position']))
            ET.SubElement(config, "Orientation").text = ",".join(map(str, values['orientation']))

        # Pretty-print and save
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)  # Requires Python 3.9+
        tree.write(self.mounting_xml_path, encoding="utf-8")

    def load_mounting_configurations(self):
        """Load all mounting positions and orientations from an XML file."""
        if not os.path.exists(self.mounting_xml_path):
            messagebox.showwarning("Warning", f"Mounting configurations file not found: {self.mounting_xml_path}")
            return
        tree = ET.parse(self.mounting_xml_path)
        root = tree.getroot()
        configs = {}

        for config in root.findall("Configuration"):
            name = config.get("name")
            pos = np.fromstring(config.find("Position").text, sep=',')
            orn = np.fromstring(config.find("Orientation").text, sep=',')
            configs[name] = {'position': pos, 'orientation': orn}

        self.mounting_configurations = configs
        print("Loaded mounting configurations:", configs)


    def show_image(self):
        if self.assembly:
            scene = assembly_to_trimesh_scene(self.assembly)
            img = self.generate_image_pyrender(scene)  # Generate the image from the assembly
            self.display_image_on_lable(img, self.image_label_asm, self.img_asm_size)
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")

    def generate_image(self, scene):
        # Create the camera transformation
        translation_matrix = trimesh.transformations.translation_matrix(scene.centroid)
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_X, [1, 0, 0])
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_Y, [0, 1, 0]))
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_Z, [0, 0, 1]))
        print(rotation_matrix)
        translation_matrix2 = trimesh.transformations.translation_matrix([0, 0, self.view_distance])
        transform = np.dot(np.dot(translation_matrix, rotation_matrix), translation_matrix2)
        scene.camera_transform = transform
        #scene.show()
        #img = scene.save_image(resolution=(1440, 810), visible = True)
        img = scene.save_image(resolution=(800, 800), visible = True)
        # Convert the image to a format suitable for Tkinter
        img_data = Image.open(io.BytesIO(img))
        return img_data
        #return ImageTk.PhotoImage(img_data)
    """def generate_ball_plot(self, scene, angle_x, angle_y, angle_z, distance):
        # Convert trimesh scene to pyrender scene
        pr_scene = pyrender.Scene.from_trimesh_scene(scene)

        # Camera transformation
        translation_matrix = trimesh.transformations.translation_matrix(scene.centroid)
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 180.0 * angle_x, [1, 0, 0])
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * angle_y, [0, 1, 0]))
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * angle_z, [0, 0, 1]))
        print(rotation_matrix)
        translation_matrix2 = trimesh.transformations.translation_matrix([0, 0, distance])
        transform = np.dot(np.dot(translation_matrix, rotation_matrix), translation_matrix2)

        # Create a camera and add it to the scene
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 8.0)
        pr_scene.add(camera, pose=transform)

        # Lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        pr_scene.add(light, pose=transform)
        pr_scene.add(light, pose=np.dot(transform,
                                 trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])))
        pr_scene.add(light, pose=np.dot(transform,
                                 trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])))
        #pyrender.Viewer(pr_scene,flags=RenderFlags.FLAT | RenderFlags.SHADOWS_ALL)
        # Offscreen rendering
        renderer = pyrender.OffscreenRenderer(200, 200)
        #color, _ = renderer.render(pr_scene, flags=RenderFlags.FLAT )
        color, _ = renderer.render(pr_scene)# , flags=RenderFlags.FLAT| RenderFlags.ALL_WIREFRAME)

        # Convert numpy array (color image) to a format suitable for Tkinter
        img_data = Image.fromarray(color)

        # Cleanup resources
        renderer.delete()

        return img_data"""
    def generate_image_pyrender(self, scene):
        # Convert trimesh scene to pyrender scene
        pr_scene = pyrender.Scene.from_trimesh_scene(scene)

        # Camera transformation
        translation_matrix = trimesh.transformations.translation_matrix(scene.centroid)
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_X, [1, 0, 0])
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_Y, [0, 1, 0]))
        rotation_matrix = np.dot(rotation_matrix, trimesh.transformations.rotation_matrix(-np.pi / 180.0 * self.view_angle_Z, [0, 0, 1]))
        print(rotation_matrix)
        translation_matrix2 = trimesh.transformations.translation_matrix([0, 0, self.view_distance])
        transform = np.dot(np.dot(translation_matrix, rotation_matrix), translation_matrix2)

        # Create a camera and add it to the scene
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 8.0)
        pr_scene.add(camera, pose=transform)

        # Lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        pr_scene.add(light, pose=transform)
        pr_scene.add(light, pose=np.dot(transform,
                                 trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])))
        pr_scene.add(light, pose=np.dot(transform,
                                 trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])))
        #pyrender.Viewer(pr_scene,flags=RenderFlags.FLAT | RenderFlags.SHADOWS_ALL)
        # Offscreen rendering
        renderer = pyrender.OffscreenRenderer(800, 800)
        #color, _ = renderer.render(pr_scene, flags=RenderFlags.FLAT )
        color, _ = renderer.render(pr_scene)# , flags=RenderFlags.FLAT| RenderFlags.ALL_WIREFRAME)

        # Convert numpy array (color image) to a format suitable for Tkinter
        img_data = Image.fromarray(color)

        # Cleanup resources
        renderer.delete()

        return img_data
    """def display_image(self, img):
        if not hasattr(self, 'image_label'):
            self.image_label = tk.Label(self)
            self.image_label.place(relx=0.5, rely=0.5, anchor='center')
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference to avoid garbage collection"""
    def generate_image_from_pointclouds(self, o3d_pointclouds, show_3d=False):
        pointclouds_pyrender = [pyrender.Mesh.from_points(pc.points, colors=pc.colors) for pc in o3d_pointclouds]
        # Create a Pyrender scene and add the point cloud
        pr_scene = pyrender.Scene()
        for pc_r in pointclouds_pyrender:
            pr_scene.add(pc_r)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
        # Camera transformation
        camera_pose = np.eye(4)
        # camera_pose[:3, :3] = trimesh.transformations.euler_matrix(-np.pi / 180.0 * 180, -np.pi / 180.0 * 0,
        #                                                           -np.pi / 180.0 * 0)[:3, :3]
        with open("view_point.txt", "r") as file: #TODO change either to static, to mounting point or to regular config entry
            lines = file.readlines()

        camera_pose[:3, 3] = np.array([float(lines[0]), float(lines[1]), float(lines[2])])

        rot_mat = np.eye(4)
        rot_mat[:3,:3] = R.from_euler('xyz', [np.pi/4,0,0]).as_matrix()

        camera_pose = np.dot(camera_pose, rot_mat)

        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        pr_scene.add_node(camera_node)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
        pr_scene.add(light, pose=camera_pose)
        if show_3d:
            pyrender.Viewer(pr_scene)
        renderer = pyrender.OffscreenRenderer(800, 800)
        # color, _ = renderer.render(pr_scene, flags=RenderFlags.FLAT )
        color, _ = renderer.render(pr_scene)  # , flags=RenderFlags.FLAT| RenderFlags.ALL_WIREFRAME)

        # Convert numpy array (color image) to a format suitable for Tkinter
        img_data = Image.fromarray(color)

        # Cleanup resources
        renderer.delete()

        return img_data

    def display_image_on_lable(self, img, label, size = (600,600)):
        resized_image = img.resize(size)
        tk_image = ImageTk.PhotoImage(resized_image)
        label.config(image=tk_image)
        #label.config(image=img)
        label.image = tk_image  # Keep a reference to avoid garbage collection

    def show_state_dialog(self):
        self.dialog = ShowStateDialog(self)

    def show_action_dialog(self):
        self.dialog = ShowActionDialog(self)

    def show_connection_dialog(self):
        if self.assembly is not None:
            self.dialog = ShowConnectionDialog(self)
        else:
            messagebox.showwarning("Warning", "No assembly loaded.")
        print("done")
    def send_nc_file(self, path = None):
        if path is None:  # Check if a file was selected
            gcode_path = resolve_path(self.config['Settings']['gcode_path'])
            file_types = [('MPF files', '*.mpf'), ('TXT files', '*.txt')]  # Adjusted to include TXT files as an example
            path = filedialog.askopenfilename(defaultextension=".mpf",
                                              filetypes=file_types,
                                              initialdir=gcode_path)
        try:
            with open(path, 'r') as nc_file:
                nc_string = nc_file.read()
            print("Connecting to client")
            # Establishing a connection with the server
            self.client.connect()
            print("Client connected")
            file_name = os.path.basename(path)
            base_node = self.client.get_node("ns=2;s=Sinumerik")
            method_node = self.client.get_node("ns=2;s=/Methods/CopyFileToServer")
            arguments = [f'Sinumerik/FileSystem/Part Program/{file_name}', nc_string.encode('utf-8'), True]
            # Call the method
            result = base_node.call_method(method_node, *arguments)
            print("Method result:", result)

            # Display success message
            #messagebox.showinfo("Success", "File sent successfully.")
            self.add_output_line(f"File {file_name} sent successfully")
            return f"File {file_name} sent successfully"


        except Exception as e:
            print("Error:", e)
            # Display error message
            messagebox.showerror("Error", "Failed to send file.")
            return None

        finally:
            # Closing the connection
            self.client.disconnect()
            print("Client disconnected")

    def take_zivid_image(self, buffer_frames=False):
        if self.zivid_camera is None:
            self.zivid_camera=ZividCamera(self.zivid_settings)
        self.zivid_camera.capture()
        self.client.connect()
        node = self.client.get_node(self.robot_D_status_node_id)
        node.set_value(ua.DataValue(ua.Variant(1.0, ua.VariantType.Double)))
        self.client.disconnect()
        #self.show_overlay()
        if buffer_frames:
            self.zivid_frame_buffer.append(self.zivid_camera.frame)
        else:
            if not os.path.exists(self.zivid_wd):
                os.mkdir(self.zivid_wd)
            num_files = len(os.listdir(self.zivid_wd))
            self.zivid_camera.save_current_frame(f"{self.zivid_wd}/{num_files:03d}.zdf")
            self.display_image_on_lable(self.zivid_camera.create_image(), self.image_label_perception, self.img_perception_size)

    def save_zivid_file(self):
        if self.zivid_camera is None or self.zivid_camera.frame is None:
            messagebox.showwarning("Warning", "Camera not initialized or no frame taken.")
        else:
            file_types = [('Zivid Data File', '*.zdf')]
            # Ask the user to select a file name for saving
            file_name = filedialog.asksaveasfilename(defaultextension=".zdf",
                                                     filetypes=file_types,
                                                     initialdir=r'C:\MaH\Data\Zivid_Frames')
            self.zivid_camera.save_current_frame(file_name)
            if self.last_frame_from_sim is not None:
                base, _ = os.path.splitext(file_name)
                new_filename = base + ".pkl"
                with open(new_filename, 'wb') as file:
                    pickle.dump(self.last_frame_from_sim, file)
    def load_zivid_file(self):
        if self.zivid_camera is None:
            self.zivid_camera=ZividCamera(self.zivid_settings)
        file_types = [('Zivid Data File', '*.zdf')]
        zivid_file_path = os.path.join(self.data_directory, 'Data', 'Zivid_Frames')
        # Ask the user to select a file name for saving
        file_name = filedialog.askopenfilename(defaultextension=".zdf",
                                               filetypes=file_types,
                                               initialdir=zivid_file_path)
        self.zivid_camera.load_frame(file_name)
        pkl_file_path = file_name.rsplit('.', 1)[0] + '.pkl'
        # Check if the .pkl file exists
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as file:
                self.last_frame_from_sim=pickle.load(file)
        self.display_image_on_lable(self.zivid_camera.create_image(), self.image_label_perception, self.img_perception_size)

    def show_overlay(self):
        # Extract the point cloud
        xyz = self.zivid_camera.frame.point_cloud().copy_data("xyz")
        xyz = np.nan_to_num(xyz).reshape(-1, 3)
        #rotate 180° around the x-axis (= flipping y and z komponents
        xyz[:,1]*=-1
        xyz[:,2]*=-1
        rgba = self.zivid_camera.frame.point_cloud().copy_data("rgba")

        # Extract color data
        colors = rgba.reshape(-1, 4)[:, :3]  # Drop alpha channel
        colors = colors / 255.0  # Normalize color values to [0, 1]
        # Create a point cloud mesh with colors
        cloud_zivid = pyrender.Mesh.from_points(xyz, colors=colors)


        # Find unique segments and assign a random color to each
        unique_segments = np.unique(self.last_frame_from_sim["index_array"])
        colors = {segment: np.random.rand(3) for segment in unique_segments}

        # Generate the color array for each point based on its segmentation index
        colors_sim = np.array([colors[index] for index in self.last_frame_from_sim["index_array"]])

        # Create a point cloud mesh from simulation
        #colors_sim = np.full((self.last_frame_from_sim["points"].shape[0], 4), [0.4, 0.74, 0.4, 0.1])
        cloud_sim = pyrender.Mesh.from_points(self.last_frame_from_sim["points"], colors=colors_sim)

        # Create a Pyrender scene and add the point cloud
        scene = pyrender.Scene()
        scene.add(cloud_zivid)
        scene.add(cloud_sim)


        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
        # Camera transformation
        camera_pose = np.eye(4)
        #camera_pose[:3, :3] = trimesh.transformations.euler_matrix(-np.pi / 180.0 * 180, -np.pi / 180.0 * 0,
        #                                                           -np.pi / 180.0 * 0)[:3, :3]
        camera_pose[:3, 3] = np.array([0, 0, -175])

        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=camera_pose)
        pyrender.Viewer(scene)

    def show_perception_result(self):
        if self.perception_module.last_step is not None:
            o3d_pointclouds = self.perception_module.result_o3d(self.perception_module.last_step["result"],
                                                                self.perception_module.last_step["segments_sim"],
                                                                self.perception_module.last_step["point_cloud_real"])
            self.display_image_on_lable(self.generate_image_from_pointclouds(o3d_pointclouds, True),
                                        self.image_label_perception, self.img_perception_size)
        else:
            messagebox.showwarning("Warning", "No result to show")

    def perform_allignment(self, part_set = None):
        #legacy function --> use global allignment from perception module
        view_matrix = np.array(self.last_frame_from_sim["view_matrix"]).reshape([4, 4], order="F")
        view_matrix_transform = np.linalg.inv(np.array(self.last_frame_from_sim["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:,3] *= 1000
        view_matrix_transform[3,3] = 1.0

        points_zivid = self.zivid_camera.frame.point_cloud().copy_data("xyz")
        points_zivid_flattened = points_zivid.reshape(-1, 3)
        valid_points = points_zivid_flattened[~np.isnan(points_zivid_flattened).any(axis=1)]
        #rotate 180° around the x-axis (= flipping y and z komponents
        valid_points[:,1]*=-1
        valid_points[:,2]*=-1
        #ones = np.ones((valid_points.shape[0], 1))
        #valid_points = np.hstack((valid_points, ones))
        #valid_points = np.matmul(np.linalg.inv(view_matrix), valid_points.T).T
        point_cloud_zivid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))
        point_cloud_zivid.transform(view_matrix_transform)
        point_cloud_zivid.estimate_normals()
        """ply_filename = "pc_zivid.ply"
        o3d.io.write_point_cloud(ply_filename, point_cloud_zivid)"""


        if part_set is None:
            part_set = {part.name for part in self.assembly.parts}

        resulting_list = [self.last_frame_from_sim["inverted_dict"].get(num, str(num)) for num in self.last_frame_from_sim["index_array"]]
        points_sim = np.array([point for point, name in
                              zip(self.last_frame_from_sim["points"], resulting_list)
                              if name in part_set])
        points_sim *= 1000
        #ones = np.ones((points_sim.shape[0], 1))
        #points_sim = np.hstack((points_sim, ones))
        #points_sim = np.matmul(np.linalg.inv(view_matrix), points_sim.T).T

        point_cloud_sim = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sim))
        point_cloud_sim.transform(view_matrix_transform)
        point_cloud_sim.estimate_normals()
        """ply_filename = "pc_sim.ply"
        o3d.io.write_point_cloud(ply_filename, point_cloud_sim)"""

       #
 #       point_cloud_zivid.points = o3d.utility.Vector3dVector(self.zivid_camera.frame.point_cloud().copy_data("xyz"))
#        point_cloud_sim.points = o3d.utility.Vector3dVector(self.last_frame_from_sim["points"])
        # Visualize aligned point clouds
        print("Displaying aligned point clouds (red: source, blue: target)")
        """o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178/255, 55/255, 44/255]),
                                           point_cloud_zivid.paint_uniform_color([0, 150/255, 130/255])])"""
        self.transformation = calculate_registration(point_cloud_sim, point_cloud_zivid,
                                                     voxelsize_coarse=5.0, voxel_size_fine=0.5, performe_icp=True)
        # Apply the transformation obtained from ICP to the source point cloud
        point_cloud_sim.transform(self.transformation)
        rotation_matrix = self.transformation[:3, :3].copy()
        rotation_euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        print(f"rotation {rotation_euler} translation {self.transformation[:3, 3]}")
        # Visualize aligned point clouds
        """o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178/255, 55/255, 44/255])])
        o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178/255, 55/255, 44/255]),
                                           point_cloud_zivid.paint_uniform_color([0, 150/255, 130/255])])"""

        self.display_image_on_lable(
            self.generate_image_from_pointclouds([point_cloud_sim.paint_uniform_color([178/255, 55/255, 44/255]),
                                                  point_cloud_zivid.paint_uniform_color([0, 150/255, 130/255])], True),
                                    self.image_label_perception, self.img_perception_size)
        # Create the 4x4 transformation matrix for camera position and orientation
        #transformation_matrix = np.eye(4)  # Start with an identity matrix
        #transformation_matrix[:3, :3] = R.from_quat(self.last_frame_from_sim["orn"]).as_matrix()  # Insert the rotation matrix
        #transformation_matrix[:3, 3] = self.last_frame_from_sim["pos"]

        #view_matrix = np.array(self.last_frame_from_sim["view_matrix"]).reshape(4,4)
        #gloabal_transform = np.dot(transformation_matrix, self.transformation)
        #gloabal_transform = transformation_matrix.dot(self.transformation)
        #gloabal_transform = np.dot(np.array(self.last_frame_from_sim["view_matrix"]).reshape(4,4), self.transformation)
        #gloabal_transform = np.matmul(np.linalg.inv(np.array(self.last_frame_from_sim["view_matrix"]).reshape([4,4], order="F")),
        #                              self.transformation.T).T

        # Extract the translation vector
        translation = np.array(self.transformation[:3, 3])

        # Extract the rotation matrix
        rotation_matrix = np.array(self.transformation[:3, :3])

        # Convert the rotation matrix to a quaternion
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns (x, y, z, w)

        data_to_send = {"request": "allign", "pos": translation, "orn": quaternion}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.sim_host, self.sim_port))
                s.sendall(pickle.dumps(data_to_send))
                answer = self.receive_data(s)
                #messagebox.showinfo("Connection sucessfull", f"Received: {answer['state']}")
                self.add_output_line(f"Connection sucessfull, Received: {answer['state']}")
            except:
                messagebox.showwarning("Warning", "Something went wrong connecting to the simulation server.")
            finally:
                s.close()

    def add_output_line(self, line):
        # Enable the text area for inserting text
        self.text_area.config(state=tk.NORMAL)

        # Insert a new line of text
        self.text_area.insert(tk.END, f"{line}\n")

        # Force the text area to update
        self.text_area.update()

        # Scroll to the end of the text area
        self.text_area.yview(tk.END)

        # Disable the text area again to prevent editing
        self.text_area.config(state=tk.DISABLED)


if __name__ == '__main__':
    # Main application
    #app = zivid.Application()
    #PM= PerceptionModule()

    #PM.load_point_clouds_from_folder(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\Data\Zivid_Frames\bsp_1")
    #PM.current_state = Assembly.load_from_xml(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\XML_Winkelschleifer\ass3_gesamt\ass3_gesatm.xml")
    
    root = tk.Tk()
    root.title("WSK disassembly planner")
    root.geometry("1920x1080")  # Set the size of the window to 800x600 pixels
    # Maximize the window
    root.state('zoomed')

    app = Application(master=root)
    app.mainloop()
