import os
import sys
import glob
import trimesh
import itertools
import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import fnmatch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodule_crc_diss_assembly_dataclasses\src')))
print(f"sys.path : {sys.path} ")

from submodule_crc_diss_assembly_dataclasses.src.diss_assembly_dataclasses.part import Part
#import open3d as o3d
#from sklearn.decomposition import PCA
#import heapq

from submodule_crc_diss_assembly_dataclasses.src.diss_assembly_dataclasses.connection import Connection, Geometric_Connection, Screw_Connection, Safety_Ring_Connection
from submodule_crc_diss_assembly_dataclasses.src.diss_assembly_dataclasses.assembly import Assembly
from planning import HierarchicalDisassemblySequencePlanner

# to install pygraphviz install graphviz and then run IN THE VIRTUAL ENVIRONMENT!!!  C:\MaH\contact_analysis\venv>pip install --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                              "PyGraphviz or PyDotPlus")
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

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

def main(motor_name, parts_to_load="", file_endings=('.stl', '.obj'), naming_rules="",
         motor_xml="", graph_pkl="", disassembly_targets=[]):

    if parts_to_load:
        # load parts
        parts = load_parts_from_folder(parts_to_load, 1, file_endings)

        # Combine all the meshes into a single scene
        scene = trimesh.Scene([part.mesh for part in parts])
        # Show the combined scene in a window
        scene.show()

        # create assembly
        motor = Assembly(motor_name, parts)
        motor.find_contacts()
        motor.save_as_xml(motor.name + ".xml")
        if naming_rules:
            motor.rename_parts(naming_rules)
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_01", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_02", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_03", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_04", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_05", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Short_06", part_name2="Lid_01", screw_head="T6"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Long_01", part_name2="Lid_01", screw_head="T12"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Long_02", part_name2="Lid_01", screw_head="T12"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Long_03", part_name2="Lid_01", screw_head="T12"))
            motor.add_connection(Screw_Connection(part_name1="Screw_Long_04", part_name2="Lid_01", screw_head="T12"))
            motor.add_connection(Safety_Ring_Connection(part_name1="Lock_Ring_01", part_name2="Rotor"))
            motor.add_connection(Safety_Ring_Connection(part_name1="Lock_Ring_02", part_name2="Lid_02"))
            motor.remove_part(motor.get_part_by_name("Oil_Cover_01"))
            motor.remove_part(motor.get_part_by_name("Oil_Cover_02"))
            motor.save_as_xml(motor.name + "_renamed.xml")

    elif motor_xml:
        motor = Assembly.load_from_xml(motor_xml)
    else:
        return
    if len(disassembly_targets) == 0:
        disassembly_targets = motor.get_part_list()
    motor.simplify_with_subassemblies(disassembly_targets)

    G, labels = motor.generate_graph()
    fig = plt.figure(figsize=(9, 9))
    for i in range(5):
        print(i)
        pos = nx.spring_layout(G, scale=2, iterations=150, seed=i)
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="white", edgecolors='black', node_shape="s")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=8)
        plt.savefig(f'figure_{i}.png')
        plt.clf()
        # plt.show()

if __name__ == "__main__":
    # load parts #########################################################
    """parts = load_parts_from_folder("DM_EMO", 1, ('.stl', '.obj'))
    # Combine all the meshes into a single scene
    parts[0].mesh.visual.vertex_colors = [84, 110, 122, 255]
    parts[1].mesh.visual.vertex_colors = [84, 110, 122, 255]
    parts[2].mesh.visual.vertex_colors = [0, 131, 143, 255]
    parts[3].mesh.visual.vertex_colors = [20, 68, 102, 255]
    parts[4].mesh.visual.vertex_colors = [20, 68, 102, 255]
    parts[5].mesh.visual.vertex_colors = [20, 68, 102, 255]
    parts[6].mesh.visual.vertex_colors = [20, 68, 102, 255]
    parts[7].mesh.visual.vertex_colors = [0, 150, 130, 255]

    scene = trimesh.Scene([part.mesh for part in parts])
    # Show the combined scene in a window
    scene.show()
    # create assembly
    motor = Assembly("LR_asm", parts)
    motor.find_contacts()
    "" "motor.add_connection(Screw_Connection(part_name1="S1", part_name2="L1", screw_head="H5"))
    motor.add_connection(Screw_Connection(part_name1="S2", part_name2="L1", screw_head="H5"))
    motor.add_connection(Screw_Connection(part_name1="S3", part_name2="L1", screw_head="H5"))
    motor.add_connection(Screw_Connection(part_name1="S4", part_name2="L1", screw_head="H5"))"" "
    motor.save_as_xml(motor.name + ".xml")"""
    # load assembly ######################################################
    test_asm = Assembly.load_from_xml("DM_EMO.xml")
    #test_asm.generate_surface_plot()
    """figures = test_asm.generate_figures(2,2)
    i=0
    for fig in figures:
        fig.savefig(f'dof{i}.png')
        i+=1"""
    #test_asm.generate_urdf_files("urdf/as01", "urdf/mesh")
    planner = HierarchicalDisassemblySequencePlanner(test_asm, cost_milling=-1, cost_screwing=5.0, cost_gripping=1.0, mill_diameter=6)

    planner.generate_full_graph()
    planner.save_graph_as_pkl(r"C:\MaH\Python\sim2real\data\DM_EMO_graph_numbered.pkl")
#    fig = planner.plot_graph()
#    fig.savefig("planning_results/full_graph.png")

   # result = planner.plan_path_to_target([test_asm.get_part_by_name("Ro")], 300, result_path="planning_results")
   # path=planner.graph.nodes[result]['cheapest_sequence_to_node']
   # path.append(result)
   # planner.visualize_sequence(path, distance=500, angle=45, file_path="planning_results")
  #  planner.export_sequence(path, "planning_results")

  #  for node in path:
  #      fig = planner.assembly_states[node].generate_surface_plot()
  #      fig.savefig(f'planning_results/sp_state{node}.png')
    #for op in planner.calculated_milling_operations:
    #    op.scene().show()
    """main("AG2V6", parts_to_load="D:/3D_motor_data/Results/l-0e007-0032-01_asm_export/L-0E007-0032-01_ASM",
         file_endings=('_vis.stl'), naming_rules="naming_rules.txt", motor_xml="", graph_pkl="")"""
    """main("AG2_Test", parts_to_load="D:/3D_motor_data/AG2_Test",
         file_endings=('_vis.stl'), naming_rules="", motor_xml="", graph_pkl="")"""
    """motor=Assembly.load_from_xml("AG2V6_renamed.xml")
    figures=motor.generate_figures()
    i=0
    for fig in figures:
        fig.savefig(f'dof{i}.png')
        i+=1"""

    """main("DM2_filtered", parts_to_load="D:\Code\disassembly_sequence_generator\Objects\DM2Mesh",
         file_endings=('.obj'), naming_rules="", motor_xml="", graph_pkl="")
    """
    """main("DM2",
         parts_to_load="",
         file_endings=('_vis.stl'),
         naming_rules="",
         motor_xml="DM2_short.xml",
         graph_pkl="")"""
    #aktuelste Version
    """main("AG2V6",
         parts_to_load="",
         file_endings=('_vis.stl'),
         naming_rules="",
         motor_xml="AG2V6_renamed.xml",
         graph_pkl="",
         disassembly_targets=[])"""

    """main("AG2V4",
         parts_to_load="",
         file_endings=('_vis.stl'),
         naming_rules="",
         motor_xml="AG2V4_renamed.xml",
         graph_pkl="",
         disassembly_targets=["Rotor"])"""

    """motor=Assembly.load_from_xml("DM2_short.xml")
    figures=motor.generate_figures()
    i=0
    for fig in figures:
        fig.savefig(f'dof{i}.png')
        i+=1
    print("?")"""

#    time.sleep(3600)
#    main("DM2", parts_to_load="", file_endings=('.stl', '.obj'), naming_rules="",
#         motor_xml="DM2.xml", graph_pkl="DM2_graph.pkl")
    """motor = Assembly.load_from_xml("DM2.xml")
    for part in motor.parts:
        test = part.mesh.as_open3d
        o3d.visualization.draw_geometries([test])
    print(motor)"""
#    motor.generate_urdf_files("D:/3D_motor_data/urdf/AG2_Demomotor")

    """motor=Assembly.load_from_xml("DM2_short.xml")

    # Combine all the meshes into a single scene
    scene = trimesh.Scene([part.mesh for part in motor.parts])
    # Show the combined scene in a window
    scene.show()"""

    #aktuelste Version
    """main("AG2V6",
         parts_to_load="",
         file_endings=('_vis.stl'),
         naming_rules="",
         motor_xml="AG2V6_renamed.xml",
         graph_pkl="",
         disassembly_targets=[])"""
    #DM_EMO
    """main("DM_EMO",
         parts_to_load="DM_EMO/",
         file_endings=('.stl'),
         naming_rules="",
         motor_xml="DM_EMO.xml",
         graph_pkl="",
         disassembly_targets=[])"""
    """main("DM_EMO",
         parts_to_load="",
         file_endings=('.stl'),
         naming_rules="",
         motor_xml="DM_EMO.xml",
         graph_pkl="",
         disassembly_targets=[])"""
    #test_parts
    """main("gm",
         parts_to_load="test_parts/geometric_holding/",
         file_endings=('.stl'),
         naming_rules="",
         motor_xml="gm.xml",
         graph_pkl="",
         disassembly_targets=[])"""
 #   mesh = trimesh.load_mesh("DM_EMO\L1.stl", force="mesh")
 #   meshes = mesh.split(only_watertight = False)
 #   mesh_union = trimesh.boolean.union(meshes)
 #   meshes = mesh_union.split(only_watertight = False)

    """test_asm = Assembly.load_from_xml("DM_EMO.xml")

    figures=test_asm.generate_figures()
    i=0
    for fig in figures:
        fig.savefig(f'dof{i}.png')
        i+=1

    for i in range(len(test_asm.connections)):
        if (np.linalg.norm(np.mean(test_asm.connections[i].DOF, axis=0))>0.5):
            planning.plan_geometric_milling2(test_asm.connections[i], test_asm, 3)"""



    """for node in planner.graph.nodes:
        labels[node]=node
        planner.assembly_states[node].show()"""

    print("done")
    """result = planner.plan_path_to_target([test_asm.get_part_by_name("gh_inner_part_01")], 100)
    for n in planner.assembly_states:
        print(f"remaining parts: {planner.assembly_states[n].generate_name_string()}")
        planner.assembly_states[n].show()
    path=planner.graph.nodes[result]['cheapest_sequence_to_node']
    path.append(result)
    for n in path:
        planner.assembly_states[n].show()
    print(f"final node: {result} path: {planner.graph.nodes[result]['cheapest_sequence_to_node']}")"""