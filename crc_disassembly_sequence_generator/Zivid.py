import numpy as np
import pyrender
import zivid
import tkinter as tk
from PIL import Image, ImageTk
import sys
import trimesh
from tkinter import filedialog
import os
from pathlib import Path

# Adjust the path as necessary for your setup
sys.path.insert(0, r'C:\Program Files\Zivid\bin')

base_path = Path(__file__).parent.parent.resolve()
print(f"base path: {base_path}")

def resolve_path(relative_path):
    """Resolve a relative path to an absolute path."""
    return (base_path / relative_path).resolve()

class ZividCamera:
    def __init__(self, settings_path = "Zivid2_Settings_Zivid_Two_M70_ManufacturingSpecular.yml"):
        # Initialize the Zivid application
        self.app = zivid.Application()

        # Connect to the first available camera
        self.camera = self.app.connect_camera()

        # Initialize settings for the camera
        complete_path = resolve_path(os.path.join("submodule_crc_disassembly_sequence_generator", "Data", "ZIVID_Settings", "manu_spec.yml"))
        print(f"settings_path: {complete_path}")
        self.settings = zivid.Settings.load(complete_path)

        # Placeholder for the last captured frame
        self.frame = None

    def capture(self):
        # Capture a frame using the current settings and save it to the class variable
        self.frame = self.camera.capture(self.settings)
    def save_current_frame(self, file_name):
        self.frame.save(file_name)

    def load_frame(self, file_name):
        self.frame = zivid.Frame(file_name)

    def create_image(self):
        # Ensure a frame has been captured
        if self.frame is None:
            raise ValueError("No frame captured. Call capture() before create_image().")

        # Extract the point cloud
        xyz = self.frame.point_cloud().copy_data("xyz")
        xyz = np.nan_to_num(xyz).reshape(-1, 3)
        #rotate 180° around the x-axis (= flipping y and z komponents
        xyz[:,1]*=-1
        xyz[:,2]*=-1
        rgba = self.frame.point_cloud().copy_data("rgba")

        # Extract color data
        colors = rgba.reshape(-1, 4)[:, :3]  # Drop alpha channel
        colors = colors / 255.0  # Normalize color values to [0, 1]

        # Create a point cloud mesh with colors
        cloud = pyrender.Mesh.from_points(xyz, colors=colors)

        # Create a Pyrender scene and add the point cloud
        scene = pyrender.Scene()
        scene.add(cloud)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
        # Camera transformation
        camera_pose = np.eye(4)
        #camera_pose[:3, :3] = trimesh.transformations.euler_matrix(-np.pi / 180.0 * 180, -np.pi / 180.0 * 0,
        #                                                           -np.pi / 180.0 * 0)[:3, :3]
        camera_pose[:3, 3] = np.array([0, 0, -175.0])

        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        # Create an offscreen renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=camera_pose)

        # Render the scene
        color, _ = renderer.render(scene)

        # Convert the rendered image to a PIL image and return it
        return Image.fromarray(color, 'RGB')


# Function to capture and display the image using ZividCamera
def capture_and_display_image():
    global label  # Reference the label object from the global scope

    # Use the ZividCamera object to capture and create an image
    zivid_camera.capture()
    image = zivid_camera.create_image()

    # Convert the PIL image to a format Tkinter can use and display it
    tk_image = ImageTk.PhotoImage(image)
    label.config(image=tk_image)
    label.image = tk_image  # Keep a reference, so it's not garbage collected

if __name__ == '__main__':
    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Zivid Camera Image Capture")

    # Create a ZividCamera object
    zivid_camera = ZividCamera()

    # Create a button to take the image
    take_image_button = tk.Button(root, text="Take Image", command=capture_and_display_image)
    take_image_button.pack()

    # Create an empty label to display the image
    label = tk.Label(root)
    label.pack()

    # Start the Tkinter event loop
    root.mainloop()