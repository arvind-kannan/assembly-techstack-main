from typing import Dict

import numpy as np
import pybullet as p

from pybullet_industrial.endeffector_tool import EndeffectorTool
from pybullet_industrial.robot_base import RobotBase
from pybullet_industrial.sensors import Camera

class RGBDCamera(Camera):
    """Extension of the Camera Endeffecto Tool that incorporates depth data.

    Args:
        urdf_model (str): A valid path to a urdf file describint the tool geometry
        start_position (np.array): the position at which the tool should be spawned
        start_orientation (np.array): the orientation at which the tool should be spawned
        camera_parameters (Dict[str,float]): A dictionary containing the camera parameters.
                                             Default Values are:
                                             'width': 480,
                                             'height': 240,
                                             'fov': 60,
                                             'aspect ratio': 1,
                                             'near plane distance': 0.01,
                                             'far plane distance': 100
        coupled_robot (RobotBase, optional): A pybullet_industrial.RobotBase object if
                                             the robot is coupled from the start.
                                             Defaults to None.
        camera_frame (str, optional): The name of the urdf_link
                                      at which the camera is located.
                                      Defaults to None in which case the last link is used.
        connector_frame (str, optional): The name of the urdf_link
                                         at which a robot connects.
                                         Defaults to None in which case the base link is used.
    """

    def __init__(self, urdf_model: str, start_position: np.array, start_orientation: np.array,
                 camera_parameters: Dict, coupled_robot: RobotBase = None,
                 camera_frame: str = None, connector_frame: str = None):

        super().__init__(urdf_model, start_position, start_orientation, camera_parameters,
                         coupled_robot, camera_frame, connector_frame)
        self.last_frame = {}
    def get_view_matrix(self):
        """link_state = p.getLinkState(
            self.urdf, self._tcp_id, computeForwardKinematics=True)"""
        #com_p = np.array(link_state[0])
        #com_o = np.array(link_state[1])
        com_p, com_o = self.get_tool_pose()
        rot_matrix = p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(
            com_p, com_p + 0.1 * camera_vector, up_vector)
        return view_matrix

    def get_image(self,cutoff_depth = 5.0):
        """Captures a camera image and depth data of the current simulation

        Returns:
            np.array: A array of pixel colors in rgba format in 255 color format
            np.array: A array of depth values as float
            np.array: mask of the image (object id per pixel)
        """
        width = self.camera_parameters['width']
        height = self.camera_parameters['height']
        images = p.getCameraImage(
            width, height, self.get_view_matrix(), self.projection_matrix)
        img = np.reshape(images[2], (height, width, 4))
        depth_img = np.reshape(images[3], (height, width))
        mask = np.reshape(images[4], (height, width))

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.get_view_matrix()).reshape([4, 4], order="F")
        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_img.reshape(-1)
        h = np.ones_like(z)
        index = mask.reshape(-1)
        index_array = np.stack([z, index], axis=1)
        index_array = index_array[z < cutoff_depth][:, -1]


        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < cutoff_depth]
        pixels[:, 2] = 2 * pixels[:, 2] - 1
        #tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
        tran_pix_world = np.linalg.inv(proj_matrix)
        # turn pixels to world coordinates
        points = np.matmul(np.linalg.inv(proj_matrix), pixels.T).T
        #points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        #points = np.matmul(np.linalg.inv(view_matrix), points.T).T
        points = points[:, :3]

        pos, orn = self.get_tool_pose()
        self.last_frame = {"img": img, "depth_img": depth_img, "mask": mask,
                           "points": points, "index_array": index_array,
                           "view_matrix": self.get_view_matrix(),
                           "pos": pos, "orn": orn}
        return self.last_frame
