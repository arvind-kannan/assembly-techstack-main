import open3d as o3d    # Open3D
import numpy as np     # Numpy
import trimesh
import pickle    # Pickle
try:
    import zivid  # only needed to load/record real camera frames
except ImportError:
    zivid = None
import pyrender
import time
import copy
from Registration import calculate_registration
from diss_assembly_dataclasses.assembly import Assembly
from diss_assembly_dataclasses.part import Part
from diss_assembly_dataclasses.connection import Connection
import tkinter as tk
from colors import get_random_color, wbk_color_dict
from utility import save_o3d_render

import os
from tkinter import simpledialog

from scipy.spatial.transform import Rotation as R

"""vis_params =  {"width": 1500,
                "height": 1500,
                "left": 50,
                "top": 50,
                "point_show_normal": False,
                "mesh_show_wireframe": True,
                "mesh_show_back_face": True,
                "lookat":np.array([-28.790215550951203, 2.929020361280652, 458.33726268305105 ]), #np.array([876, -70, 0]),  # Camera position
                "up": np.array([0.14448541292315401, 0.49951378116111778, 0.85417208329618732]),  # Up vector
                "front": np.array([0.32753343105569549, -0.79042545997505287, 0.51763833297400808]),  # Front direction
                "zoom": 0.67999999999999994 }"""
vis_params = {  "width": 900,
                "height": 900,
                "left": 50,
                "top": 50,
                "field_of_view": 60.0,
                "front": [0,0.5,0.5],
                "lookat": [-2.3548726458102465, 21.575264632701874, 525.0],
                "up": [0,0,1],
                "zoom": 0.95999999999999996}

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


def extract_segments(points, part_set, mask):
    point_clouds_by_name = {name: [] for name in part_set}
    for point, point_name in zip(points, mask):
        if point_name in part_set:
            point_clouds_by_name[point_name].append(point)
    point_clouds_by_name_o3d = {name: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for name, points in point_clouds_by_name.items()}

    return point_clouds_by_name_o3d


def concatenate_point_clouds(point_clouds):
    # Concatenate all point clouds into a single point cloud
    concatenated_point_cloud = o3d.geometry.PointCloud()
    # Iterate over all point clouds
    for point_cloud in point_clouds.values():
        if len(point_cloud.points) > 0:
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

def check_icp_match(icp_result):
    print(
        f"Checking ICP Results {icp_result.fitness} {icp_result.inlier_rmse} {icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)}")
    #if icp_result.fitness > 0.66 and icp_result.inlier_rmse < 1.0 and icp_result.fitness / icp_result.inlier_rmse > 1.5:
    if icp_result.fitness > 0.7 and icp_result.inlier_rmse < 1.0 and icp_result.fitness / icp_result.inlier_rmse > 1.5:
        print("match OK")
        return True
    else:
        print("match not OK")
        return False

class PerceptionModule:
    def __init__(self):
        #self.simulated_point_clouds_list = []
        #self.real_point_clouds_list = []

        self.initialize_current_data_objects()
        self.last_step = None
        self.log = ""

        self.global_transformation = None
        self.assembly_base_rotation_matrix = None
    def initialize_current_data_objects(self):
        self.current_state = None
        self.assembly_base_rotation_matrix = None

        self.current_views_sim = []
        self.current_clouds_real = []

        self.current_segments_sim = {}
        self.current_bounding_boxes = {}
        self.current_point_cloud_sim = None
        self.current_point_cloud_real = None
        self.current_segments_real = {}


    def add_view(self, frame_sim, frame_zivid):
        if self.assembly_base_rotation_matrix is None:
            self.assembly_base_rotation_matrix = R.from_quat(frame_sim["assembly_base_orn"]).as_matrix()
        frame_sim["points"] *= 1000
        view_matrix_transform = np.linalg.inv(np.array(frame_sim["view_matrix"]).reshape([4, 4], order="F"))
        view_matrix_transform[:, 3] *= 1000
        view_matrix_transform[3, 3] = 1.0
        # Extract all unique values
        unique_values = set(frame_sim["inverted_dict"].values())
        part_set = list(unique_values)

        resulting_list = [frame_sim["inverted_dict"].get(num, str(num)) for num in
                          frame_sim["index_array"]]

        segments_sim = extract_segments(frame_sim["points"], part_set, resulting_list)
        for key in segments_sim:
            if len(segments_sim[key].points) > 0:
                segments_sim[key].transform(view_matrix_transform)
                segments_sim[key].estimate_normals()
        self.current_views_sim.append(segments_sim)

        parts_to_show = []
        colors = ["l_gray"]
        for segment in segments_sim.values():
            color_name, color = get_random_color(colors)
            parts_to_show.append(segment.paint_uniform_color(color / 255))
            colors.append(color_name)
        #o3d.visualization.draw_geometries(parts_to_show)

        points_zivid = frame_zivid.point_cloud().copy_data("xyz")
        points_zivid_flattened = points_zivid.reshape(-1, 3)
        valid_points = points_zivid_flattened[~np.isnan(points_zivid_flattened).any(axis=1)]
        # rotate 180° around the x-axis (= flipping y and z komponents
        valid_points[:, 1] *= -1
        valid_points[:, 2] *= -1
        point_cloud_zivid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_points))
        point_cloud_zivid.transform(view_matrix_transform)
        point_cloud_zivid.estimate_normals()
        self.current_clouds_real.append(point_cloud_zivid)
        #o3d.visualization.draw_geometries([point_cloud_zivid.paint_uniform_color(wbk_color_dict["blue"] / 255)])

    def load_point_clouds_from_folder(self, base_path):
        # generate a list of all files in the directory
        file_list_sim = os.listdir(os.path.join(base_path, "wd_sim"))
        file_list_sim = [os.path.join(base_path, "wd_sim", file) for file in file_list_sim]
        file_list_real = os.listdir(os.path.join(base_path, "wd"))
        file_list_real = [os.path.join(base_path, "wd", file) for file in file_list_real]
        for file_to_load_sim, file_to_load_real in zip(file_list_sim, file_list_real):
            print(f"Loading {file_to_load_sim} and {file_to_load_real}")
            with open(file_to_load_sim, 'rb') as file:
                # Load the contents from the file and return it
                frame_sim = pickle.load(file)

            frame_real = zivid.Frame(file_to_load_real)
            self.add_view(frame_sim, frame_real)

    def concatenate_current_views_sim(self):
        self.current_segments_sim = self.current_views_sim[0]
        used_colors = ["green", "red", "yellow", "blue"]
        colors = {}
        for key in self.current_segments_sim:
            name, color = get_random_color(used_colors)
            colors[key] = color/255
            used_colors.append(name)
        """view_params = { "lookat": [-2.3548726458102465, 21.575264632701874, 525.0],
                        "up": [0, 0, 1],
                        "front": [0, -1, 1],
                        "zoom": 0.8}
        angle = np.radians(-90.0)  # Define the start angle
        for i in range(30):
            for key in self.current_segments_sim:
                self.current_segments_sim[key].paint_uniform_color(colors[key] * (1 - i / 30 * 2 / 3))
            view_params["front"] = [np.cos(angle), np.sin(angle), 1]  # Interpolate x and y
            save_o3d_render([pcd for pcd in self.current_segments_sim.values()],
                            view_params, f"C:/MaH/Data/Renders/sim/view_00_render_{i:02d}.png")
            angle += np.radians(180/4/30)"""  # Increment the angle by 12 degrees for each iteration


        j = 1
        for entry in self.current_views_sim[1:]:
            """for i in range(30):
                for key, segment in entry.items():
                    segment.paint_uniform_color(colors[key] * (1 - i / 30 * 2 / 3))
                view_params["front"] = [np.cos(angle), np.sin(angle), 1]  # Interpolate x and y
                save_o3d_render([pcd for pcd in self.current_segments_sim.values()] +
                                [segment for segment in entry.values()],
                view_params, f"C:/MaH/Data/Renders/sim/view_{j:02d}_render_{i:02d}.png")
                angle += np.radians(180 / 4 / 30)"""
            for key, segment in entry.items():
                if len(segment.points)>0 and key in self.current_segments_sim:
                    self.current_segments_sim[key] += segment
                elif len(segment.points) > 0:
                    self.current_segments_sim[key] = segment
                    name, color = get_random_color(used_colors)
                    colors[key] = color/255
                    used_colors.append(name)
            for key in self.current_segments_sim:
                self.current_segments_sim[key].paint_uniform_color(colors[key]/3)

            j += 1

        for key in self.current_segments_sim:
            self.current_segments_sim[key].voxel_down_sample(voxel_size=0.25)
            self.current_segments_sim[key].paint_uniform_color(colors[key])

        self.current_point_cloud_sim = concatenate_point_clouds(self.current_segments_sim)

    def concatenate_current_views_real(self):
        global_transform = np.eye(4)
        threshold_icp_coarse = 10.0
        threshold_icp_fine = 2.5
        voxel_size = 1.0
        run_global_registration = True
        roi = self.current_point_cloud_sim.get_axis_aligned_bounding_box()
        roi.scale(1.5, center=roi.get_center())
        base_cloud = self.current_clouds_real[0].crop(roi)
        #paint the base cloud in a random color
        used_colors = ["green", "red", "yellow", "blue"]
        name, color = get_random_color(used_colors)
        used_colors.append(name)
        base_cloud.paint_uniform_color(color / 255)
        self.current_point_cloud_real = base_cloud
        self.current_point_cloud_real.estimate_normals()
        """view_params = { "lookat": [-2.3548726458102465, 21.575264632701874, 525.0],
                        "up": [0, 0, 1],
                        "front": [0, -1, 1],
                        "zoom": 0.95}
        angle = np.radians(-90.0)
        for i in range(30):
            view_params["front"] = [np.cos(angle), np.sin(angle), 1]  # Interpolate x and y
            save_o3d_render([self.current_point_cloud_real],
                            view_params, f"C:/MaH/Data/Renders/real/view_00_render_{i:02d}.png")
            angle += np.radians(180/4/30)"""
        j=1
        for entry in self.current_clouds_real[1:]:
            crop=entry.crop(roi)
            print("calculating registration...")
            crop.estimate_normals()
            print(f"number of points: {len(crop.points)}")
            #o3d.visualization.draw_geometries([crop, self.current_point_cloud_real])
            if (run_global_registration):
                global_transform = calculate_registration(crop, self.current_point_cloud_real,
                                                          voxelsize_coarse=voxel_size*2, performe_icp=False)
            icp_result = o3d.pipelines.registration.registration_icp(
                crop.voxel_down_sample(voxel_size=voxel_size*2),
                self.current_point_cloud_real.voxel_down_sample(voxel_size=voxel_size*2),
                threshold_icp_coarse, global_transform,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, relative_fitness=1e-5,
                                                                           relative_rmse=1e-5))

            print(icp_result)
            print(icp_result.transformation)
            icp_result = o3d.pipelines.registration.registration_icp(
                crop.voxel_down_sample(voxel_size=voxel_size),
                self.current_point_cloud_real.voxel_down_sample(voxel_size=voxel_size),
                threshold_icp_fine, icp_result.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000, relative_fitness=1e-7,
                                                                           relative_rmse=1e-7))
            print(f"number of points: {len(crop.points)}")
            print(icp_result)
            print(icp_result.transformation)
            crop.transform(icp_result.transformation)
            # transform = calculate_registration( crop, concatenated_point_clouds,
            #                                     voxelsize_coarse=2.5, voxel_size_fine=0.5, performe_icp=True)
            # crop.transform(transform)
            name, color = get_random_color(used_colors)
            used_colors.append(name)
            crop.paint_uniform_color(color / 255)
            self.current_point_cloud_real += crop
            """for i in range(30):
                view_params["front"] = [np.cos(angle), np.sin(angle), 1]  # Interpolate x and y
                save_o3d_render([self.current_point_cloud_real],
                                view_params, f"C:/MaH/Data/Renders/real/view_{j:02d}_render_{i:02d}.png")
                angle += np.radians(180 / 4 / 30)
            j += 1"""


            #o3d.visualization.draw_geometries([self.current_point_cloud_real])

        self.current_point_cloud_real.voxel_down_sample(voxel_size=0.25)
        #o3d.visualization.draw_geometries([self.current_point_cloud_real])

    def calculate_global_transform(self):
        if self.current_point_cloud_sim is None:
            self.concatenate_current_views_sim()
        if self.current_point_cloud_real is None:
            self.concatenate_current_views_real()
        self.global_transformation = calculate_registration(self.current_point_cloud_sim, self.current_point_cloud_real,
                                                            voxelsize_coarse=2.5, voxel_size_fine=0.5,
                                                            performe_icp=True,
                                                            estimation_methode_icp=o3d.pipelines.registration.TransformationEstimationPointToPlane())

    def process_current_state(self):
        if self.current_point_cloud_sim is None:
            self.concatenate_current_views_sim()
        if self.current_point_cloud_real is None:
            self.concatenate_current_views_real()
        # check if file was selected
        if self.global_transformation is None:
            self.calculate_global_transform()

        #temp = copy.deepcopy(self.current_point_cloud_sim)
        #create a 4x4 transformation matrix to rotate 10° around the z-axis
        """temp_transformation_rot = np.eye(4)
        temp_transformation_rot[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.pi/18))

        temp.transform(temp_transformation_rot)"""
        self.current_point_cloud_sim.transform(self.global_transformation)
        """o3d.visualization.draw_geometries([temp.paint_uniform_color(wbk_color_dict["yellow"]/255),
                                          self.current_point_cloud_real.paint_uniform_color(wbk_color_dict["blue"]/255),
                                          self.current_point_cloud_sim.paint_uniform_color(wbk_color_dict["green"]/255)])"""
        for segment in self.current_segments_sim.values():
            segment.transform(self.global_transformation)
        #self.current_bounding_boxes = self.calculate_bboxes_for_state(self.current_segments_sim, self.current_state)
        result_compare_current, self.log = self.compare_segments(self.current_segments_sim,
                                                                 self.current_point_cloud_real,
                                                                 self.current_state,
                                                                 self.global_transformation,
                                                                 self.assembly_base_rotation_matrix,
                                                                 self.log)

        with open("log.txt", "w") as file:
            file.write(self.log)


        if self.last_step:#TODO check if global transformation und assembly base orientation should be saved in last step data --> probably sicne needed for roi calculation
            segments_sim_adjusted = self.apply_result_to_segments(self.last_step["result"], self.last_step["segments_sim"])
            result_compare_last, self.log = self.compare_segments(segments_sim_adjusted,
                                                                  self.current_point_cloud_real,
                                                                  self.last_step["state"],
                                                                  self.last_step["global_transformation"],
                                                                  self.last_step["assembly_base_rotation_matrix"],
                                                                  self.log)
            with open("log.txt", "w") as file:
                file.write(self.log)
            """self.show_result(result_compare_last,
                             self.last_step["segments_sim"],
                             self.current_point_cloud_real,
                             self.calculate_bboxes_for_state(self.last_step["segments_sim"], self.last_step["state"]))"""

        self.last_step = {"segments_sim": copy.deepcopy(self.current_segments_sim),
                          "point_cloud_real": copy.deepcopy(self.current_point_cloud_real),
                          "state": self.current_state,
                          "result": result_compare_current,
                          "global_transformation":self.global_transformation,
                          "assembly_base_rotation_matrix":self.assembly_base_rotation_matrix}

        self.initialize_current_data_objects()

    def show_current_segments(self, individual_view = False):
        parts_to_show = []
        for key in self.current_bounding_boxes:
            self.current_bounding_boxes[key].color = wbk_color_dict["blue"] / 255
            parts_to_show.append(self.current_bounding_boxes[key])
        parts_to_show.append(self.current_point_cloud_real.paint_uniform_color(wbk_color_dict["blue"] / 255))
        parts_to_show.append(self.current_point_cloud_sim.paint_uniform_color(wbk_color_dict["yellow"] / 255))
        o3d.visualization.draw_geometries(parts_to_show)
        if individual_view:
            for key in self.current_bounding_boxes:
                self.current_bounding_boxes[key].color = wbk_color_dict["blue"] / 255
                parts_to_show = [self.current_bounding_boxes[key],
                                 self.current_segments_real[key].paint_uniform_color(wbk_color_dict["blue"]/255)]
                if key in self.current_segments_sim:
                    parts_to_show.append(self.current_segments_sim[key].paint_uniform_color(wbk_color_dict["green"]/255))
                o3d.visualization.draw_geometries(parts_to_show)
    def show_result(self, result, segments_sim, point_cloud_real, bounding_boxes, individual_view = False):
        parts_to_show_complete = [point_cloud_real.paint_uniform_color(wbk_color_dict["blue"]/255)]
        if individual_view:
            segments_real = extract_segments_from_bounding_boxes(point_cloud_real, bounding_boxes)
        for key in result:
            individual_parts_to_show = []
            if result[key]["found_part"]:
                if result[key]["part_moved"]:
                    segments_sim[key].paint_uniform_color(wbk_color_dict["yellow"]/255)
                    moved_part = copy.deepcopy(segments_sim[key])
                    moved_part.transform(result[key]["transformation"])
                    parts_to_show_complete.append(moved_part.paint_uniform_color(wbk_color_dict["green"]/255))
                    individual_parts_to_show.append(moved_part)
                else:
                    segments_sim[key].paint_uniform_color(wbk_color_dict["green"]/255)
            else:
                segments_sim[key].paint_uniform_color(wbk_color_dict["red"]/255)
            parts_to_show_complete.append(segments_sim[key])
            individual_parts_to_show.append(segments_sim[key])
            bounding_boxes[key].color = wbk_color_dict["blue"]/255
            parts_to_show_complete.append(bounding_boxes[key])
            individual_parts_to_show.append(bounding_boxes[key])
            if individual_view:
                individual_parts_to_show.append(segments_real[key].paint_uniform_color(wbk_color_dict["blue"]/255))
                o3d.visualization.draw_geometries(individual_parts_to_show)
        """view_params = { "lookat": [-2.3548726458102465, 21.575264632701874, 525.0],
                        "up": [0, 0, 1],
                        "front": [0, -1, 1],
                        "zoom": 0.95}
        angle = np.radians(-90.0)
        for i in range(30*8):
            view_params["front"] = [np.cos(angle), np.sin(angle), 1]  # Interpolate x and y
            save_o3d_render(parts_to_show_complete,
                            view_params, f"C:/MaH/Data/Renders/result/render_{i:03d}.png")
            angle += np.radians(180/4/30)"""
        o3d.visualization.draw_geometries(parts_to_show_complete)

    def result_o3d(self, result, segments_sim, point_cloud_real):
        parts_to_show_complete = [point_cloud_real.paint_uniform_color(wbk_color_dict["blue"]/255)]
        for key in result:
            if result[key]["found_part"]:
                if result[key]["part_moved"]:
                    segments_sim[key].paint_uniform_color(wbk_color_dict["yellow"]/255)
                    moved_part = copy.deepcopy(segments_sim[key])
                    moved_part.transform(result[key]["transformation"])
                    parts_to_show_complete.append(moved_part.paint_uniform_color(wbk_color_dict["green"]/255))
                else:
                    segments_sim[key].paint_uniform_color(wbk_color_dict["green"]/255)
            else:
                segments_sim[key].paint_uniform_color(wbk_color_dict["red"]/255)
            parts_to_show_complete.append(segments_sim[key])
        return parts_to_show_complete

    @staticmethod
    def calculate_bboxes_for_state(segments_sim, state, global_transformation=np.eye(4), base_orientation=np.eye(3)): #TODO check wether this should be static or not
        dofs = {part.name: state.get_part_dof(part) for part in state.parts}
        for part_name in dofs.keys():
            if len(dofs[part_name]) > 0:
                print(f"part {part_name} has dofs {dofs[part_name]}, transforming with {base_orientation} and {global_transformation[:3,:3]}")
                dofs[part_name] = np.dot(global_transformation[:3,:3], np.dot(base_orientation, np.array(dofs[part_name]).T)).T
                print(f"part {part_name} has now dofs {dofs[part_name]}")
        return generate_bounding_boxes_dict(segments_sim, dofs=dofs)

    @staticmethod
    def apply_result_to_segments(result, segments_sim):
        for key in result:
            if result[key]["found_part"] and result[key]["part_moved"]:
                segments_sim[key].transform(result[key]["transformation"])
        return segments_sim


    @staticmethod
    def compare_segments(segments_sim, point_cloud_real, state, global_transformation=np.eye(4), base_orientation=np.eye(3), log = ""): #TODO check wether this should be static or not, YES to differentiate for last step
        result = {}
        rois = PerceptionModule.calculate_bboxes_for_state(segments_sim, state, global_transformation, base_orientation)
        segments_real = extract_segments_from_bounding_boxes(point_cloud_real, rois)
        for key in rois.keys():
            result[key] = {"found_part": False, "part_moved": False, "transformation": np.eye(4)}
            print(f"calculating registration for part {key}")
            global_transform = np.eye(4)
            if len(segments_sim[key].points) > 100000:
                voxel_size_coarse = 5.0
                voxel_size_fine = 1.0
                threshold_icp = None
                estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            elif len(segments_sim[key].points) < 20000:
                voxel_size_coarse = 1.0
                voxel_size_fine = 0.1
                threshold_icp = 3.0
                estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            else:
                voxel_size_coarse = 1.0
                voxel_size_fine = 0.25
                threshold_icp = 3.0
                estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            print(f"number of points: {len(segments_sim[key].points)}")
            result[key]["found_part"] = False
            segment_old = copy.deepcopy(segments_sim[key])
            iterations = 2

            for i in range(iterations):
                print("calculating new transformation")
                if len(segments_real[key].points) > 0:
                    result[key]["transformation"] = calculate_registration(segments_sim[key], segments_real[key],
                                                                           voxelsize_coarse=voxel_size_coarse,
                                                                           voxel_size_fine=voxel_size_fine,
                                                                           performe_icp=True,
                                                                           estimation_methode_icp=estimation_methode_icp,
                                                                           threshold_icp=threshold_icp,)
                    rotation_matrix = copy.deepcopy(result[key]["transformation"][:3, :3])
                    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)  # Convert to Euler angles (x, y, z) in degrees
                    print(f"Rotations for part {key}: {euler_angles}")
                    result[key]["part_moved"] = True  # TODO not ideal. Need to check what this is otherwise needed for
                    segments_sim[key].transform(result[key]["transformation"])
                    obb_pcd = segments_sim[key].get_oriented_bounding_box()
                    obb_mesh = state.get_part_by_name(key).mesh.bounding_box_oriented
                    voulume_ratio = obb_pcd.volume() / obb_mesh.volume
                    print(f"obb volume ratio for {key}: {voulume_ratio}")
                    if voulume_ratio < 0.25: #Supresses exessive rotation for where not much is visible
                        mean_old = np.mean(np.asarray(segment_old.points), axis=0)
                        mean_new = np.mean(np.asarray(segments_sim[key].points), axis=0)
                        print("small part detected, adjusting transformation to translation only. "
                              f"Old mean: {mean_old}, New mean: {mean_new} ")
                        result[key]["transformation"] = np.eye(4)
                        result[key]["transformation"][:3, 3] = mean_new - mean_old
                        segments_sim[key] = copy.deepcopy(segment_old)
                        segments_sim[key].transform(result[key]["transformation"])

                icp_result = o3d.pipelines.registration.registration_icp(
                    segments_sim[key],
                    segments_real[key],
                    1.0, global_transform,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1, relative_fitness=1e-7,
                                                                               relative_rmse=1e-7))
                if check_icp_match(icp_result):
                    result[key]["found_part"] = True
                    # log += f"Part {key} OK {i+1}/{iterations}; ICP Results {icp_result.fitness} {icp_result.inlier_rmse} {icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)}\n"
                    log += f"{key};OK;{icp_result.fitness};{icp_result.inlier_rmse};{icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)};"
                    break

            segments_sim[key] = copy.deepcopy(segment_old)
            if not result[key]["found_part"]:
                log += f"{key};NG;{icp_result.fitness};{icp_result.inlier_rmse};{icp_result.fitness / max(icp_result.inlier_rmse, 0.00001)};"
            """if i > 0:
                result[key]["part_moved"] = True
                log += f"{segments_sim[key].get_center() - segment_old.get_center()};"
                print({f"distance between com: {segments_sim[key].get_center() - segment_old.get_center()}"})"""
            log += "\n"
        return result, log

    def save_real_point_cloud(self, path):
        if self.current_point_cloud_sim is None:
            self.concatenate_current_views_sim()
        if self.current_point_cloud_real is None:
            self.concatenate_current_views_real()
        o3d.io.write_point_cloud(path, self.current_point_cloud_real)

    def generate_mesh(self):
        if self.current_point_cloud_sim is None:
            self.concatenate_current_views_sim()
        if self.current_point_cloud_real is None:
            self.concatenate_current_views_real()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=self.current_point_cloud_real, depth=8)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()
        return mesh


if __name__ == '__main__':
    # Initialize the Zivid application
    app = zivid.Application()
    PM = PerceptionModule()

    #PM.load_point_clouds_from_folder(r"C:\MaH\Data\Zivid_Frames\SR_Test")
    #PM.current_state = Assembly.load_from_xml(r"C:\MaH\Data\Assemblies\SRG_Test.xml")

    PM.load_point_clouds_from_folder(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\Data\Zivid_Frames\bsp_1")
    PM.current_state = Assembly.load_from_xml(r"C:\Users\david\OneDrive\Uni\00_Masterarbeit\20_Code\probabilistic_dissasembly_planner\submodule_crc_disassembly_sequence_generator\XML_Winkelschleifer\ass3_gesamt\ass3_gesatm.xml")

    connections_to_remove = []
    for con in PM.current_state.connections:
        if con.connection_type == "screw":
            connections_to_remove.append(con)
    for con in connections_to_remove:
        PM.current_state.remove_connection(con)


    PM.process_current_state()
    PM.show_result( PM.last_step["result"],
                    PM.last_step["segments_sim"],
                    PM.last_step["point_cloud_real"],
                    PM.calculate_bboxes_for_state(
                        PM.last_step["segments_sim"],
                        PM.last_step["state"],
                        PM.last_step["global_transformation"],
                        PM.last_step["assembly_base_rotation_matrix"]),
                    individual_view=False)




    PM.load_point_clouds_from_folder(r"C:\MaH\Data\Validation\240724\i_i\0_initial")
    PM.current_state = Assembly.load_from_xml(r"C:\MaH\Data\Validation\240724\Sequence\assembly_states" + "/state_0.xml")



    parts_to_disassemble = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]
    for part_to_disassemble in parts_to_disassemble:
        connections = PM.current_state.get_connections_by_part_name(part_to_disassemble)
        for con in connections:
            if con.connection_type == "screw":
                PM.current_state.remove_connection(con)

    PM.process_current_state()

    PM.generate_trimesh_scene_for_result(PM.last_step["result"], PM.last_step["segments_sim"], PM.last_step["point_cloud_real"]).show()

    PM.load_point_clouds_from_folder(r"C:\MaH\Data\Validation\240724\moved_parts\screws_5mm_01_no_screw_1")
    PM.current_state = Assembly.load_from_xml(r"C:\MaH\Data\Validation\240724\Sequence\assembly_states" + "/state_1.xml")
    parts_to_disassemble = ["04_screw_1", "05_screw_2", "06_screw_3", "07_screw_4"]
    for part_to_disassemble in parts_to_disassemble:
        connections = PM.current_state.get_connections_by_part_name(part_to_disassemble)
        for con in connections:
            if con.connection_type == "screw":
                PM.current_state.remove_connection(con)
    PM.process_current_state()

    for i in range(2):
        base_path = tk.filedialog.askdirectory(initialdir=r"C:\MaH\Data\Validation\240724")
        PM.load_point_clouds_from_folder(base_path)
        #use tkinter to select xml file to load
        file_types = [('XML files', '*.xml')]
        motor_file = tk.filedialog.askopenfilename(filetypes=file_types, initialdir=r"C:\MaH\Data\Validation\240724\Sequence\assembly_states", title="Select the motor file")
        #motor_file = r"C:\MaH\Data\Validation\240724\Sequence\assembly_states" + "/state_0.xml"
        PM.current_state = Assembly.load_from_xml(motor_file)
        #ask for filename to load transformation from
        file_types = [('numpy array', '*.npy')]
        #transformation_file = r"C:\MaH\Data\Validation\240724\initial_transform_complete.npy"
        transformation_file = tk.filedialog.askopenfilename(filetypes=file_types, initialdir=r"C:\MaH\Data\Validation\240724", title="Select the transformation file")
        if transformation_file:
            PM.global_transformation = np.load(transformation_file)
        PM.process_current_state()
        #PM.show_current_segments(individual_view=True)

    print("done")
    