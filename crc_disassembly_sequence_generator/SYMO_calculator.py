"""import tkinter as tk
from tkinter import messagebox
import math

def solve():
    try:
        x = -float(entry_x.get())
        y = -float(entry_y.get())
        L1 = -x/math.sqrt(2) + y/math.sqrt(2)
        L3 = -x/math.sqrt(2) - y/math.sqrt(2)
        result_label.config(text=f"L1 = {L1:.3f}\nL3 = {L3:.3f}")
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numbers for x and y.")

# GUI setup
root = tk.Tk()
root.title("Solve for L1 and L3")

tk.Label(root, text="Enter x:").grid(row=0, column=0, padx=10, pady=5)
entry_x = tk.Entry(root)
entry_x.grid(row=0, column=1)

tk.Label(root, text="Enter y:").grid(row=1, column=0, padx=10, pady=5)
entry_y = tk.Entry(root)
entry_y.grid(row=1, column=1)

solve_button = tk.Button(root, text="Solve", command=solve)
solve_button.grid(row=2, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="")
result_label.grid(row=3, column=0, columnspan=2)

root.mainloop()"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial.transform import Rotation as R

# Core logic to compute transformation
def pose_to_matrix(position, euler_angles_deg):
    rotation = R.from_euler('xyz', euler_angles_deg, degrees=True)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix

def matrix_to_pose(matrix):
    position = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3])
    euler_angles_deg = rotation.as_euler('xyz', degrees=True)
    return position, euler_angles_deg

def compute_relative_transform_post_rotation(posA, rotA, posB, rotB):
    # Convert to rotation matrices
    RA = R.from_euler('XYZ', rotA, degrees=True).as_matrix()
    RB = R.from_euler('XYZ', rotB, degrees=True).as_matrix()

    # Compute relative rotation and translation
    R_rel = RB.T@RA

    # World-frame translation
    t_world = np.array(posB) - np.array(posA)

    # Express translation in R_rel's frame
    t_local = t_world @ RA @ R_rel.T

    # Convert relative rotation matrix to Euler angles
    rel_euler = R.from_matrix(R_rel).as_euler('XYZ', degrees=True)

    return t_local, rel_euler

# GUI logic
def compute():
    try:
        posA = [float(entries_poseA[i].get()) for i in range(3)]
        rotA = [float(entries_poseA[i+3].get()) for i in range(3)]
        posB = [float(entries_poseB[i].get()) for i in range(3)]
        rotB = [float(entries_poseB[i+3].get()) for i in range(3)]

        rel_pos, rel_rot = compute_relative_transform_post_rotation(posA, rotA, posB, rotB)
        result_label.config(
            text=f"Relative Translation:\n"
                 f"L1 = {rel_pos[2]:.3f}, L2 = {rel_pos[1]:.3f}, L3 = {rel_pos[0]:.3f}\n"
                 f"Relative Rotation (Euler XYZ):\n"
                 f"R1 = {rel_rot[2]:.3f}°, R2 = {rel_rot[1]:.3f}°, R3 = {rel_rot[0]:.3f}°"
        )
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numeric values.")

# Build GUI
root = tk.Tk()
root.title("Relative Pose Calculator")

# Labels
tk.Label(root, text="Pose A (target) – Position [x y z] and Rotation [rx ry rz] (°)").grid(row=0, column=0, columnspan=6)
entries_poseA = []
for i in range(6):
    e = tk.Entry(root, width=8)
    e.insert(0, "0")  # Set default value to 0
    e.grid(row=1, column=i)
    entries_poseA.append(e)

tk.Label(root, text="Pose B (current) – Position [x y z] and Rotation [rx ry rz] (°)").grid(row=2, column=0, columnspan=6)
entries_poseB = []
for i in range(6):
    e = tk.Entry(root, width=8)
    e.insert(0, "0")  # Set default value to 0
    e.grid(row=3, column=i)
    entries_poseB.append(e)

tk.Button(root, text="Compute Relative Transformation", command=compute).grid(row=4, column=0, columnspan=6, pady=10)

result_label = tk.Label(root, text="", justify=tk.LEFT, font=("Arial", 10))
result_label.grid(row=5, column=0, columnspan=6, pady=10)

root.mainloop()