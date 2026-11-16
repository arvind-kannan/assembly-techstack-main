import copy
import math
import time

import pybullet as p
import pybullet_data
import pybullet_industrial as pi
import os
import numpy as np

if __name__ == "__main__":
    # Setup the simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    working_dir = os.path.dirname(__file__)

    # Initilizing urdf files
    urdf_comau = os.path.join(working_dir,
                              'robot_descriptions', 'gantry.urdf')

    # Load the robot
    robot = pi.RobotBase(urdf_comau, start_position=[0, 0, 0], start_orientation=[0, 0, 0, 1], default_endeffector="wagon_c")

    robot.set_joint_position({"x_axis": 0.5, "y_axis": 0.5, "z_axis": 0.5, "a_axis": 0., "b_axis": 0.5})
    for i in range(100):
        time.sleep(0.05)
        p.stepSimulation()
    pos, orn = robot.get_endeffector_pose()
    print(f"pos: {pos}, orn: {p.getEulerFromQuaternion(orn)}")
    robot.set_joint_position({"x_axis": 0.5, "y_axis": 0.5, "z_axis": 0.5, "a_axis": 0., "b_axis": -0.5, "c_axis": -0.5})
    for i in range(100):
        time.sleep(0.05)
        p.stepSimulation()
    pos, orn = robot.get_endeffector_pose()
    print(f"pos: {pos}, orn: {p.getEulerFromQuaternion(orn)}")
    robot.set_joint_position({"x_axis": 0.5, "y_axis": 0.5, "z_axis": 0.5, "a_axis": 0.5, "b_axis": 0.5, "c_axis": 0.5})
    for i in range(100):
        time.sleep(0.05)
        p.stepSimulation()
    pos, orn = robot.get_endeffector_pose()
    print(f"pos: {pos}, orn: {p.getEulerFromQuaternion(orn)}")
    robot.set_joint_position({"x_axis": 0.5, "y_axis": 0.5, "z_axis": 0.5, "a_axis": 0.5, "b_axis": 0.5, "c_axis": -0.5})
    for i in range(100):
        time.sleep(0.05)
        p.stepSimulation()
    pos, orn = robot.get_endeffector_pose()
    print(f"pos: {pos}, orn: {p.getEulerFromQuaternion(orn)}")
    while True:
        robot.set_endeffector_pose([1.0, 1.0, 1.0], p.getQuaternionFromEuler([0, 0,  0]))
        for i in range(100):
            time.sleep(0.05)
            p.stepSimulation()
        print("point 1.1")
        robot.set_endeffector_pose([1.0, 1.0, 1.0], p.getQuaternionFromEuler([-math.pi/4,math.pi/4,0]))
        for i in range(100):
            time.sleep(0.05)
            p.stepSimulation()
        print("point 1.2")
        robot.set_endeffector_pose([1.0, 1.0, 1.0], p.getQuaternionFromEuler([math.pi/4,-math.pi/4,0]))
        for i in range(100):
            time.sleep(0.05)
            p.stepSimulation()
        print("point 1.3")

