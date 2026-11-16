import open3d as o3d    # Open3D
import numpy as np     # Numpy
import pickle    # Pickle
import zivid
import pyrender
import time
import copy
from Registration import calculate_registration
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part, load_parts_from_folder
from diss_assembly_dataclasses.connection import Connection
import tkinter as tk
from colors import get_random_color, wbk_color_dict

import os
from tkinter import simpledialog


vis_params =  {"width": 1500,
                "height": 1500,
                "left": 50,
                "top": 50,
                "point_show_normal": False,
                "mesh_show_wireframe": True,
                "mesh_show_back_face": True,
                "lookat":np.array([-28.790215550951203, 2.929020361280652, 458.33726268305105 ]), #np.array([876, -70, 0]),  # Camera position
                "up": np.array([0.14448541292315401, 0.49951378116111778, 0.85417208329618732]),  # Up vector
                "front": np.array([0.32753343105569549, -0.79042545997505287, 0.51763833297400808]),  # Front direction
                "zoom": 0.67999999999999994 }


class StringSelectionDialog(simpledialog.Dialog):
    def __init__(self, parent, title, string_list):
        self.string_list = string_list
        self.selected_string = None
        super().__init__(parent, title)

    def body(self, master):
        tk.Label(master, text="Select a string:").grid(row=0)
        self.string_var = tk.StringVar(master)
        self.string_var.set(self.string_list[0])
        self.dropdown = tk.OptionMenu(master, self.string_var, *self.string_list)
        self.dropdown.grid(row=1, column=0)
        return self.dropdown

    def apply(self):
        self.selected_string = self.string_var.get()


def select_string_from_list(string_list):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    dialog = StringSelectionDialog(root, "Select component", string_list)
    root.destroy()  # Destroy the main window after the dialog is closed
    return dialog.selected_string


#Function to compare tow point clouds
def compare_point_clouds(point_cloud_1, point_cloud_2):
    # Compute the distance between the two point clouds
    distance = np.linalg.norm(point_cloud_1 - point_cloud_2, axis=1)

    # Compute the mean distance
    mean_distance = np.mean(distance)

    return mean_distance

def extract_segments(points, part_set, mask):
    point_clouds_by_name = {name: [] for name in part_set}
    for point, point_name in zip(points, mask):
        if point_name in part_set:
            point_clouds_by_name[point_name].append(point)
    """for name in part_set:
        # Filter points for the current name
        points_for_name = [point for point, point_name in zip(points, mask) if point_name == name]

        # Convert the list of points into a NumPy array
        #points_array = np.array(points_for_name)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_for_name))

        # Store the array in the dictionary
        point_clouds_by_name[name] = point_cloud"""
    point_clouds_by_name_o3d = {name: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for name, points in point_clouds_by_name.items()}

    return point_clouds_by_name_o3d

def concatenate_point_clouds(point_clouds):
    # Concatenate all point clouds into a single point cloud
    concatenated_point_cloud = o3d.geometry.PointCloud()
    # Iterate over all point clouds
    for point_cloud in point_clouds.values():
        if len(point_cloud.points)>0:
            concatenated_point_cloud += point_cloud

    return concatenated_point_cloud

#Fuction to generate bounding boxes for a dictionary of point clouds
def generate_bounding_boxes_dict(point_clouds, scale_factor = 1.1, dofs=None, dof_factor=30.0):
    bounding_boxes_dict = {}
    for key, point_cloud in point_clouds.items():
        if len(point_cloud.points)>0:
            if dofs is not None and len(dofs[key])>0:
                cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=5.0)
                new_cloud = copy.deepcopy(cloud_downsampled)
                for dof in dofs[key]:
                    cloud_downsampled.translate(dof*dof_factor)
                    new_cloud += cloud_downsampled
                    cloud_downsampled.translate(-dof*dof_factor)
                bounding_box = new_cloud.get_oriented_bounding_box()
            else:
                bounding_box = point_cloud.get_oriented_bounding_box()
            #scale boundingbox
            bounding_box.scale(scale_factor, center=bounding_box.get_center())
            bounding_boxes_dict[key] = bounding_box

    return bounding_boxes_dict
#Funciton to extract pointcloud segments from a given pointcloud given a dictionary of bounding boxes
def extract_segments_from_bounding_boxes(point_cloud, bounding_boxes_dict):
    segments = {}
    for key, bounding_box in bounding_boxes_dict.items():
        segment = point_cloud.crop(bounding_box)
        segments[key] = segment

    return segments

#Function to show the overlay of the Zivid and the simulation
def show_overlay(zivid_frame, sim_frame):
    # Extract the point cloud
    xyz = zivid_frame.point_cloud().copy_data("xyz")
    xyz = np.nan_to_num(xyz).reshape(-1, 3)
    #rotate 180° around the x-axis (= flipping y and z komponents
    xyz[:,1]*=-1
    xyz[:,2]*=-1
    rgba = zivid_frame.point_cloud().copy_data("rgba")

    # Extract color data
    colors = rgba.reshape(-1, 4)[:, :3]  # Drop alpha channel
    colors = colors / 255.0  # Normalize color values to [0, 1]
    # Create a point cloud mesh with colors
    cloud_zivid = pyrender.Mesh.from_points(xyz, colors=colors)


    # Find unique segments and assign a random color to each
    unique_segments = np.unique(sim_frame["index_array"])
    colors = {segment: np.random.rand(3) for segment in unique_segments}

    # Generate the color array for each point based on its segmentation index
    colors_sim = np.array([colors[index] for index in sim_frame["index_array"]])

    # Create a point cloud mesh from simulation
    #colors_sim = np.full((sim_frame["points"].shape[0], 4), [0.4, 0.74, 0.4, 0.1])
    cloud_sim = pyrender.Mesh.from_points(sim_frame["points"], colors=colors_sim)

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

def load_views(base_path):
    # generate a list of all files in the directory
    file_list_sim = os.listdir(os.path.join(base_path, "wd_sim"))
    file_list_sim = [os.path.join(base_path, "wd_sim", file) for file in file_list_sim]
    clouds_sim = []
    file_list_real = os.listdir(os.path.join(base_path, "wd"))
    file_list_real = [os.path.join(base_path, "wd", file) for file in file_list_real]
    clouds_real = []
    for file_to_load_sim, file_to_load_real in zip(file_list_sim, file_list_real):
        print(f"Loading {file_to_load_sim} and {file_to_load_real}")
        with open(file_to_load_sim, 'rb') as file:
            # Load the contents from the file and return it
            data = pickle.load(file)
        data["points"] *= 1000
        view_matrix_transform = np.linalg.inv(np.array(data["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:, 3] *= 1000
        view_matrix_transform[3, 3] = 1.0
        # Extract all unique values
        unique_values = set(data["inverted_dict"].values())
        part_set = list(unique_values)

        resulting_list = [data["inverted_dict"].get(num, str(num)) for num in
                          data["index_array"]]

        segments_sim = extract_segments(data["points"], part_set, resulting_list)
        for key in segments_sim:
            if len(segments_sim[key].points)>0:
                segments_sim[key].transform(view_matrix_transform)
                segments_sim[key].estimate_normals()
        clouds_sim.append(segments_sim)

        frame = zivid.Frame(file_to_load_real)
        points_zivid = frame.point_cloud().copy_data("xyz")
        points_zivid_flattened = points_zivid.reshape(-1, 3)
        valid_points = points_zivid_flattened[~np.isnan(points_zivid_flattened).any(axis=1)]
        # rotate 180° around the x-axis (= flipping y and z komponents
        valid_points[:, 1] *= -1
        valid_points[:, 2] *= -1
        point_cloud_zivid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))
        point_cloud_zivid.transform(view_matrix_transform)
        point_cloud_zivid.estimate_normals()
        clouds_real.append(point_cloud_zivid)
        # point_cloud_sim = concatenate_point_clouds(segments_sim)
        # o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color(wbk_color_dict["yellow"]/255),
        #                                    point_cloud_zivid.paint_uniform_color(wbk_color_dict["blue"]/255)])
    return clouds_sim, clouds_real

def load_sim(path):
    # generate a list of all files in the directory
    file_list_sim = os.listdir(path)
    file_list_sim = [os.path.join(path, file) for file in file_list_sim]
    clouds_sim = []
    for file_to_load_sim in file_list_sim:
        print(f"Loading {file_to_load_sim}")
        with open(file_to_load_sim, 'rb') as file:
            # Load the contents from the file and return it
            data = pickle.load(file)
        data["points"] *= 1000
        view_matrix_transform = np.linalg.inv(np.array(data["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:, 3] *= 1000
        view_matrix_transform[3, 3] = 1.0
        # Extract all unique values
        unique_values = set(data["inverted_dict"].values())
        part_set = list(unique_values)

        resulting_list = [data["inverted_dict"].get(num, str(num)) for num in
                          data["index_array"]]

        segments_sim = extract_segments(data["points"], part_set, resulting_list)
        for key in segments_sim:
            if len(segments_sim[key].points)>0:
                segments_sim[key].transform(view_matrix_transform)
                segments_sim[key].estimate_normals()
        clouds_sim.append(segments_sim)
    return clouds_sim
def concatenate_multiple_views_sim(list_of_segmented_point_clouds):
    concatenated_segments = list_of_segmented_point_clouds[0]
    used_colors = ["blue"]
    for entry in list_of_segmented_point_clouds[1:]:
        for key, segment in entry.items():
            if len(segment.points)>0 and key in concatenated_segments:
                concatenated_segments[key] += segment
            else:
                concatenated_segments[key] = segment
    for key in concatenated_segments:
        concatenated_segments[key].voxel_down_sample(voxel_size=0.25)
        name, color = get_random_color(used_colors)
        used_colors.append(name)
        print(name)
        concatenated_segments[key].paint_uniform_color(color/255)
    return concatenated_segments

def concatenate_multiple_views_real(list_of_point_clouds, roi):
    global_transform = np.eye(4)
    threshold_icp = 2.5
    voxel_size = 1.0
    base_cloud = list_of_point_clouds[0].crop(roi)
    #paint the base cloud in a random color
    base_cloud.paint_uniform_color(np.random.rand(3))
    concatenated_point_clouds = base_cloud
    #concatenated_point_clouds.estimate_normals()
    for entry in list_of_point_clouds[1:]:
        crop=entry.crop(roi)
        print("calculating registration...")
        #crop.estimate_normals()

        icp_result = o3d.pipelines.registration.registration_icp(
            crop.voxel_down_sample(voxel_size=voxel_size),
            concatenated_point_clouds.voxel_down_sample(voxel_size=voxel_size),
            threshold_icp, global_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-7,
                                                                       relative_rmse=1e-7))
        print(icp_result)
        crop.transform(icp_result.transformation)
        # transform = calculate_registration( crop, concatenated_point_clouds,
        #                                     voxelsize_coarse=2.5, voxel_size_fine=0.5, performe_icp=True)
        # crop.transform(transform)
        crop.paint_uniform_color(np.random.rand(3))
        concatenated_point_clouds += crop
        # o3d.visualization.draw_geometries([concatenated_point_clouds])
    """for i in range(1, len(list_of_point_clouds)):
        crop=list_of_point_clouds[i].crop(roi)
        print("calculating registration...")
        #crop.estimate_normals()
        global_transform = calculate_registration(crop, list_of_point_clouds[i-1].crop(roi),
                                                  voxelsize_coarse=1.0, voxel_size_fine=0.25, performe_icp=False)
        icp_result = o3d.pipelines.registration.registration_icp(
            crop.voxel_down_sample(voxel_size=voxel_size),
            concatenated_point_clouds.voxel_down_sample(voxel_size=voxel_size),
            threshold_icp, global_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-7,
                                                                       relative_rmse=1e-7))
        print(icp_result)
        crop.transform(icp_result.transformation)
        # transform = calculate_registration( crop, concatenated_point_clouds,
        #                                     voxelsize_coarse=2.5, voxel_size_fine=0.5, performe_icp=True)
        # crop.transform(transform)
        crop.paint_uniform_color(np.random.rand(3))
        concatenated_point_clouds += crop
        o3d.visualization.draw_geometries([concatenated_point_clouds])"""
    concatenated_point_clouds.voxel_down_sample(voxel_size=0.25)
    #o3d.visualization.draw_geometries([concatenated_point_clouds])
    return concatenated_point_clouds

def check_icp_match(icp_results):
    print(
        f"Checking ICP Results {icp_result.fitness} {icp_result.inlier_rmse} {icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)}")
    if icp_results.fitness > 0.66 and icp_results.inlier_rmse < 1.0 and icp_results.fitness / icp_results.inlier_rmse > 1.5:
        print("match OK")
        return True
    else:
        print("match not OK")
        return False
# Initialize the Zivid application
app = zivid.Application()
# Define the base path
#base_path = r"C:\MaH_Kretschmer\data\Validation\dataset_EMO_SR_240613\Lid_01_wonky"
#base_path = r"C:\MaH_Kretschmer\data\Validation\dataset_EMO_SR_240613\stator_sucsess"
#base_path = r"C:\MaH_Kretschmer\data\Validation\dataset_EMO_SR_240613\Stator_initial"
#base_path = r"C:\MaH\Data\Validation\Stator_initial"
#base_path = r"C:\MaH\Data\Validation\Lid_01_wonky"
#base_path = r"C:\MaH\Data\Validation\S2_erfolg"
#base_path = r"C:\MaH\Data\Validation\S3_fail"

#use tkinter to select the folder with C:\MaH\Data\Validation as base path
base_path = tk.filedialog.askdirectory(initialdir=r"C:\MaH\Data\Validation\240724")
# base_path = tk.filedialog.askdirectory(initialdir=r"C:\MaH\Data\Zivid_Frames")
log = f"part_name;status;iteration;fitness;inlier_rmse;fitness/inlier_rmse;distance;{str(base_path)};\n"
#AUTO####################################################################################
#use tkinter to select xml file to load
file_types = [('XML files', '*.xml')]
motor_file = tk.filedialog.askopenfilename(filetypes=file_types, initialdir=r"C:\MaH\Data\Validation\240724\Sequence\assembly_states", title="Select the motor file")
#motor_file = r"C:\MaH\Data\Validation\240724\Sequence\assembly_states" + "/state_0.xml"
motor = Assembly.load_from_xml(motor_file)

#AUTO####################################################################################
#parts_to_disassemble = []
parts_to_disassemble = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4", ]
# for _ in range(len(motor.get_part_list())):
#     part = select_string_from_list(motor.get_part_list())
#     if part is None:
#         break
#     elif part not in parts_to_disassemble:
#         parts_to_disassemble.append(part)

#AUTO####################################################################################

#ask for filename to load transformation from
file_types = [('numpy array', '*.npy')]
#transformation_file = r"C:\MaH\Data\Validation\240724\initial_transform_complete.npy"
transformation_file = tk.filedialog.askopenfilename(filetypes=file_types, initialdir=r"C:\MaH\Data\Validation\240724", title="Select the transformation file")

#start stopwhatch
start = time.time()
#motor = Assembly.load_from_xml(r"C:\MaH_Kretschmer\data\Validation\dataset_EMO_SR_240613\Sequence\assembly_states\state_0.xml")
#parts_to_disassemble = "06_screw_3"
for part_to_disassemble in parts_to_disassemble:
    connections = motor.get_connections_by_part_name(part_to_disassemble)
    for con in connections:
        if con.connection_type == "screw":
            motor.remove_connection(con)
dofs = {part.name: motor.get_part_dof(part) for part in motor.parts}

#dof = motor.get_part_dof(motor.get_part_by_name("06_screw_3"))
print("Time to load motor: ", time.time()-start)

clouds_sim, clouds_real = load_views(base_path)
#check if folder base_path\wd_sim_match exists
if os.path.exists(os.path.join(base_path, "wd_sim_match")):
    clouds_sim = load_sim(os.path.join(base_path, "wd_sim_match"))

#time to load
print("Time to load pointclouds: ", time.time()-start)

segments_sim = concatenate_multiple_views_sim(clouds_sim)
print("Time to concatenate segments: ", time.time()-start)
point_cloud_sim = concatenate_point_clouds(segments_sim)
print("Time to concatenate whole cloud: ", time.time()-start)
roi_base = point_cloud_sim.get_axis_aligned_bounding_box()
roi_base.scale(1.5, center=roi_base.get_center())

point_cloud_real = concatenate_multiple_views_real(clouds_real, roi_base)
print("Time to concatenate real cloud: ", time.time()-start)

#check if file was selected
if transformation_file:
    global_transform = np.load(transformation_file)
else:
    global_transform = calculate_registration(point_cloud_sim, point_cloud_real,
                                              voxelsize_coarse=2.5, voxel_size_fine=0.5, performe_icp=True, estimation_methode_icp=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_file = tk.filedialog.asksaveasfilename(filetypes=file_types, initialdir=r"C:\MaH\Data\Validation\240724", title="Select the transformation file")
    np.save(transformation_file, global_transform)

point_cloud_sim.transform(global_transform)
#o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178 / 255, 55 / 255, 44 / 255]), point_cloud_real.paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])
for segment in segments_sim.values():
    segment.transform(global_transform)
# o3d.visualization.draw_geometries(list(segments_sim.values()) + [point_cloud_real.paint_uniform_color(wbk_color_dict["blue"]/255)], **vis_params)

# for pcd_real, pcd_sim in zip(clouds_real, clouds_sim):
#     o3d.visualization.draw_geometries([concatenate_point_clouds(pcd_sim).paint_uniform_color(wbk_color_dict["green"]/255), pcd_real.paint_uniform_color(wbk_color_dict["blue"]/255)])


bounding_boxes = generate_bounding_boxes_dict(segments_sim, dofs=dofs)
print("Time to generate bounding boxes: ", time.time()-start)
segments_real = extract_segments_from_bounding_boxes(point_cloud_real, bounding_boxes)
print("Time to extract segments: ", time.time()-start)

parts_to_show_complete = [point_cloud_real.paint_uniform_color(wbk_color_dict["blue"]/255)]
for key in bounding_boxes:
    print(f"calculating registration for part {key}")
    global_transform = np.eye(4)
    threshold_icp = 2.5
    voxel_size = 1.0
    if len(segments_sim[key].points)>100000:
        voxel_size_coarse = 5.0
        voxel_size_fine = 1.0
        estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        voxel_size_coarse = 0.25
        voxel_size_fine = 0.25
        estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    option = o3d.pipelines.registration.FastGlobalRegistrationOption()
    print(f"number of points: {len(segments_sim[key].points)}")
    found_part = False
    segment_old = copy.deepcopy(segments_sim[key])
    iterations = 2

    for i in range(iterations):
        icp_result = o3d.pipelines.registration.registration_icp(
            segments_sim[key],
            segments_real[key],
            1.0, global_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1, relative_fitness=1e-7,
                                                                       relative_rmse=1e-7))
        if check_icp_match(icp_result):
            found_part = True
            #log += f"Part {key} OK {i+1}/{iterations}; ICP Results {icp_result.fitness} {icp_result.inlier_rmse} {icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)}\n"
            log += f"{key};OK;{icp_result.fitness};{icp_result.inlier_rmse};{icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)};"
            break
        else:
            print("calculating new transformation")
            if len(segments_real[key].points)==0:
                transformation = np.eye(4)
            else:
                transformation = calculate_registration(segments_sim[key], segments_real[key],
                                                        voxelsize_coarse=voxel_size_coarse,
                                                        voxel_size_fine=voxel_size_fine, performe_icp=True,
                                                        estimation_methode_icp=estimation_methode_icp)
            segments_sim[key].transform(transformation)
            # o3d.visualization.draw_geometries([segments_sim[key], segments_real[key]])

            # icp_result = o3d.pipelines.registration.registration_icp(
            #     segments_sim[key].voxel_down_sample(voxel_size=voxel_size_coarse*2),
            #     segments_real[key].voxel_down_sample(voxel_size=voxel_size_coarse*2),
            #     20.0, global_transform,
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-7,
            #                                                                relative_rmse=1e-7))
            # icp_result = o3d.pipelines.registration.registration_icp(
            #     segments_sim[key],
            #     segments_real[key],
            #     2.5, icp_result.transformation,
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-7,
            #                                                                relative_rmse=1e-7))
            # segments_sim[key].transform(icp_result.transformation)
    bounding_boxes[key].color = wbk_color_dict["blue"] / 255
    parts_to_show_complete.append(bounding_boxes[key])
    parts_to_show = [bounding_boxes[key],
                     segments_real[key].paint_uniform_color(wbk_color_dict["blue"]/255)]
    if found_part:
        parts_to_show.append(segments_sim[key].paint_uniform_color(wbk_color_dict["green"]/255))
        parts_to_show_complete.append(segments_sim[key].paint_uniform_color(wbk_color_dict["green"]/255))
    else:
        log += f"{key};NG;{icp_result.fitness};{icp_result.inlier_rmse};{icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)};"
        segments_sim[key] = copy.deepcopy(segment_old)
    if i>0:
        parts_to_show.append(segment_old.paint_uniform_color(wbk_color_dict["red"]/255))
        parts_to_show_complete.append(segment_old.paint_uniform_color(wbk_color_dict["red"]/255))
        log += f"{segments_sim[key].get_center()-segment_old.get_center()};"
        print({f"distance between com: {segments_sim[key].get_center()-segment_old.get_center()}"})
    log += "\n"
    # o3d.visualization.draw_geometries(parts_to_show)
o3d.visualization.draw_geometries(parts_to_show_complete)

#remove log file if exists
if os.path.exists(os.path.join(base_path, "log.txt")):
    os.remove(os.path.join(base_path, "log.txt"))
#save log to file in base path
with open(os.path.join(base_path, "log.txt"), "w") as file:
    file.write(log)

    #print(f"icp_result for {key}: {icp_result}, transformation: {icp_result.transformation}")
    # icp_result = o3d.pipelines.registration.registration_icp(
    #     segments_sim[key].voxel_down_sample(voxel_size=voxel_size),
    #     segments_real[key].voxel_down_sample(voxel_size=voxel_size),
    #     threshold_icp, global_transform,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200, relative_fitness=1e-7,
    #                                                                relative_rmse=1e-7))
    # print(f"icp_result: {icp_result}\n transformation:\n {icp_result.transformation}")
    # segments_sim[key].transform(icp_result.transformation)
    # transformation = calculate_registration(segments_sim[key], segments_real[key],
    #                                              voxelsize_coarse=5.0, voxel_size_fine=0.5, performe_icp=True)
    #segment_old = copy.deepcopy(segments_sim[key])
    # segments_sim[key].transform(transformation)
    # #calculate the distance between the center of mass of the two segments
    # center_of_mass_new = segments_sim[key].get_center()
    # center_of_mass_old = segment_old.get_center()
    # print(f"center of mass new: {center_of_mass_new}, center of mass old: {center_of_mass_old}, delta: {center_of_mass_new-center_of_mass_old}")
    # bounding_boxes[key].color = (238 / 255, 183 / 255, 13 / 255)
    # o3d.visualization.draw_geometries([ bounding_boxes[key],
    #                                     segment_old.paint_uniform_color([238 / 255, 183 / 255, 13 / 255]),
    #                                     segments_sim[key].paint_uniform_color([178 / 255, 55 / 255, 44 / 255]),
    #                                     segments_real[key].paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])

#o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178 / 255, 55 / 255, 44 / 255]), point_cloud_real.paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])
# o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178 / 255, 55 / 255, 44 / 255])])
# o3d.visualization.draw_geometries([point_cloud_real.paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])


"""#generate a list of all files in the directory
file_list_real = os.listdir(os.path.join(base_path, "wd"))
file_list_sim = os.listdir(os.path.join(base_path, "wd_sim"))


# Generate the paths
frame_file_path = os.path.join(base_path, "wd", "026.zdf")
frame_file_path = os.path.join(base_path, "wd", "010.zdf")
frame_file_path = os.path.join(base_path, "wd", "000.zdf")
file_path = os.path.join(base_path, "wd_sim", "010.pkl")
file_path = os.path.join(base_path, "wd_sim", "000.pkl")


with open(file_path, 'rb') as file:
    # Load the contents from the file and return it
    data = pickle.load(file)
data["points"] *= 1000



frame = zivid.Frame(frame_file_path)

show_overlay(frame, data)

view_matrix_transform = np.linalg.inv(np.array(data["view_matrix"]).reshape([4, 4], order="F"))
view_matrix_transform[:, 3] *= 1000
view_matrix_transform[3, 3] = 1.0

# Extract all unique values
unique_values = set(data["inverted_dict"].values())
part_set = list(unique_values)

resulting_list = [data["inverted_dict"].get(num, str(num)) for num in
                  data["index_array"]]
# points_sim = np.array([point for point, name in
#                        zip(data["points"], resulting_list)
#                        if name in part_set])

segments_sim = extract_segments(data["points"], part_set, resulting_list)
for segment in segments_sim.values():
    if len(segment.points)>0:
        segment.transform(view_matrix_transform)
point_cloud_sim = concatenate_point_clouds(segments_sim)

points_zivid = frame.point_cloud().copy_data("xyz")
points_zivid_flattened = points_zivid.reshape(-1, 3)
valid_points = points_zivid_flattened[~np.isnan(points_zivid_flattened).any(axis=1)]
# rotate 180° around the x-axis (= flipping y and z komponents
valid_points[:, 1] *= -1
valid_points[:, 2] *= -1
rgba = frame.point_cloud().copy_data("rgba")
pcd = frame.point_cloud()
# Extract color data
colors_zivid = rgba.reshape(-1, 4)[:, :3]  # Drop alpha channel
colors_zivid = colors_zivid / 255.0  # Normalize color values to [0, 1]
point_cloud_zivid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))
# Assign colors to the point cloud
point_cloud_zivid.colors = o3d.utility.Vector3dVector(colors_zivid)
point_cloud_zivid.transform(view_matrix_transform)


bounding_boxes = generate_bounding_boxes_dict(segments_sim)
segments_real = extract_segments_from_bounding_boxes(point_cloud_zivid, bounding_boxes)


o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178 / 255, 55 / 255, 44 / 255]),
                                   point_cloud_zivid.paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])

for key in bounding_boxes:
    bounding_boxes[key].color = (238 / 255, 183 / 255, 13 / 255)
    o3d.visualization.draw_geometries([ bounding_boxes[key],
                                        segments_sim[key].paint_uniform_color([178 / 255, 55 / 255, 44 / 255]),
                                        segments_real[key].paint_uniform_color([0 / 255, 150 / 255, 130 / 255])])"""

# point_cloud_sim = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sim))
# point_cloud_sim.transform(view_matrix_transform)

# Initialize the Zivid application
"""app = zivid.Application()

frame = zivid.Frame(frame_file_path)

points_zivid = frame.point_cloud().copy_data("xyz")
points_zivid_flattened = points_zivid.reshape(-1, 3)
valid_points = points_zivid_flattened[~np.isnan(points_zivid_flattened).any(axis=1)]
# rotate 180° around the x-axis (= flipping y and z komponents
valid_points[:, 1] *= -1
valid_points[:, 2] *= -1
rgba = frame.point_cloud().copy_data("rgba")
pcd = frame.point_cloud()
# Extract color data
colors_zivid = rgba.reshape(-1, 4)[:, :3]  # Drop alpha channel
colors_zivid = colors_zivid / 255.0  # Normalize color values to [0, 1]
# ones = np.ones((valid_points.shape[0], 1))
# valid_points = np.hstack((valid_points, ones))
# valid_points = np.matmul(np.linalg.inv(view_matrix), valid_points.T).T
point_cloud_zivid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))
# Assign colors to the point cloud
point_cloud_zivid.colors = o3d.utility.Vector3dVector(colors_zivid)

point_cloud_zivid.transform(view_matrix_transform)

o3d.visualization.draw_geometries([point_cloud_sim.paint_uniform_color([178/255, 55/255, 44/255]),
                                   point_cloud_zivid])"""

print("done")

