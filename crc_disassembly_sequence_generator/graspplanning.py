import trimesh
import random
import itertools
import numpy as np
from crc_disassembly_sequence_generator.utility import fit_minimum_volume_cylinder_along_axis
import networkx as nx

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

"""def find_orthogonal_faces(mesh, vector, tolerance=1e-6):
    normals = mesh.face_normals
    dot_products = np.dot(normals, vector)
    orthogonal_indices = np.where(np.abs(dot_products) < tolerance)[0]
    return orthogonal_indices, normals[orthogonal_indices]

def cluster_normals(normals, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(normals)
    return labels

def find_opposing_clusters(normals, labels):
    cluster_normals = [normals[labels == i].mean(axis=0) for i in np.unique(labels)]
    opposing_pairs = []
    for i, normal_i in enumerate(cluster_normals):
        for j, normal_j in enumerate(cluster_normals):
            if i < j:
                if np.dot(normal_i, normal_j) < -0.9:  # Adjust the threshold as needed
                    opposing_pairs.append((i, j))
    return opposing_pairs


def generate_distinct_colors(n):
    # Use itertools to generate combinations of colors with a large step to ensure distinction
    step = 256 // int(n ** (1 / 3))
    colors = list(itertools.product(range(0, 256, step), repeat=3))

    # Shuffle the colors and pick the first n
    random.shuffle(colors)
    selected_colors = colors[:n]

    # Add the alpha channel (255) to each color
    rgba_colors = [(r, g, b, 255) for r, g, b in selected_colors]

    return rgba_colors

# Load the mesh
mesh = trimesh.load('C:\\MaH\\Data\\Parts\\AD2V5_Test_Sensor_cut\\V5_Sensor_cut.stl')

# Define the given vector
given_vector = np.array([0, 0, 1])  # Example vector

# Find orthogonal faces
orthogonal_faces_indices, orthogonal_normals = find_orthogonal_faces(mesh, given_vector)

# Assuming we want to cluster into 3 clusters as an example
n_clusters = 8
labels = cluster_normals(orthogonal_normals, n_clusters)

# Find opposing clusters
opposing_pairs = find_opposing_clusters(orthogonal_normals, labels)

# Output the results
print("Orthogonal Face Indices:", orthogonal_faces_indices)
print("Opposing Clusters:", opposing_pairs)

# Define colors for different clusters
colors = generate_distinct_colors(n_clusters)
# Create a new mesh with colored vertices
orthogonal_faces = mesh.faces[orthogonal_faces_indices]
orthogonal_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=orthogonal_faces)
vertex_colors = np.full((orthogonal_mesh.vertices.shape[0],4),  [128, 128, 128, 255]) # RGBA
for pair_index, (cluster_a, cluster_b) in enumerate(opposing_pairs):
    color=colors[pair_index % len(colors)]
    indices = [index for index, element in enumerate(labels) if element == cluster_a or element == cluster_b]
    vertex_colors[orthogonal_mesh.faces[indices]] = color

orthogonal_mesh.visual.vertex_colors = vertex_colors

# Visualize the orthogonal mesh with colored vertices
orthogonal_mesh.show()
"""

# Load the mesh using trimesh
"""mesh = trimesh.load('C:\\MaH\\Data\\Parts\\AD2V5_Test_Sensor_cut\\V5_Sensor_cut.stl')

# Define the voxel size (pitch)
pitch = 5.0  # Adjust the pitch size as needed

# Voxelize the mesh
voxelized_mesh = mesh.voxelized(pitch)

# Convert the voxelized mesh to a Trimesh object for visualization
voxel_mesh = voxelized_mesh.as_boxes()

# Create a scene and add the voxelized mesh
scene = trimesh.Scene([voxel_mesh])

# Show the scene
scene.show()
def visualize_mesh_with_plane(mesh, plane_origin, plane_normal):
    # Create a plane for visualization
    plane_size = mesh.scale * 2  # Adjust the plane size based on the mesh scale
    plane_transform = trimesh.transformations.rotation_matrix(
        np.pi / 2, plane_normal)
    plane = trimesh.creation.box(
        extents=(plane_size, plane_size, 0.001),
        transform=trimesh.transformations.compose_matrix(translate=plane_origin) @ plane_transform)

    # Create a scene and add the mesh and plane
    scene = trimesh.Scene([mesh, plane])
    scene.show()

# Load a mesh using trimesh
mesh = trimesh.load('C:\MaH\Data\Parts\AD2V5_Test_Sensor_cut\V5_Sensor_cut.stl')

# Calculate the center of gravity (centroid) of the mesh
center_of_gravity = np.mean(mesh.vertices, axis=0)


# Define the slicing plane's normal vector
plane_normal = np.array([0, 0, 1])  # For a horizontal slice, change as needed

# Slice the mesh with the plane through the center of gravity
slice_result = mesh.section(plane_origin=center_of_gravity, plane_normal=plane_normal)

# Check if the slice result is valid
if slice_result is not None:
    # Create a scene
    scene = trimesh.Scene()

    # Define a list of colors
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0],  # Blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [1.0, 0.0, 1.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0, 1.0],  # Cyan
        [0.5, 0.5, 0.5, 1.0],  # Gray
        [1.0, 0.5, 0.0, 1.0],  # Orange
        [0.5, 0.0, 0.5, 1.0],  # Purple
        [0.5, 0.5, 0.0, 1.0]  # Olive
    ]

    # Ensure there are enough colors for each path
    color_index = 0
    # Add the path to the scene
    scene.add_geometry(slice_result)

    # Show the scene
    scene.show()
    # Iterate through the entities of the Path3D object
    for entity in slice_result.entities:
        # Set the color for this closed loop
        entity.color = colors[color_index % len(colors)]
        color_index += 1

    # Show the scene
    scene.show()
else:
    print("Slicing failed. Visualizing the mesh and the slicing plane.")"""
def find_pertruding_grip_points(mesh, normal, max_grip_width=50.0,slicing_thickness = 1.0):
    normal = normal / np.linalg.norm(normal)  # Normalize the vector

    # Calculate the bounds of the mesh along the direction of the slicing plane normal
    cylinder, cylinder_data, cylinder_top, cylinder_bottom = fit_minimum_volume_cylinder_along_axis(mesh.vertices, normal, 32)

    # Create a scene to visualize the slices
    #scene = trimesh.Scene()
    grip_point = None
    grip_depth = None
    # Iterate through the mesh along the axis defined by the plane normal at 5mm intervals
    for i in range(0, round(cylinder_data["height"]/slicing_thickness)+1):
        # Define the plane origin for the current slice
        plane_origin = cylinder_top - i*normal*slicing_thickness

        # Slice the mesh at the current position
        slice_result = mesh.section(plane_origin=plane_origin, plane_normal=normal)

        if slice_result is not None:
            # Compute the oriented bounds
            transform, ext = trimesh.bounds.oriented_bounds(slice_result.vertices)
            if np.min(ext[ext>5.0])>max_grip_width:
                break
            grip_point = slice_result.centroid
            grip_depth = np.linalg.norm(i*normal)
            """if np.min(ext[ext>5.0])>max_grip_width:
                color = [178, 55, 44, 255]
            else:
                color = [0, 150, 130, 255]
            # Create a box from the oriented bounds
            #box = trimesh.primitives.Box(extents=ext, transform=np.linalg.inv(transform))
            #scene.add_geometry(box)
                slice_result.colors = [color for _ in range(len(slice_result.entities))]
            # Add the slice to the scene
            scene.add_geometry(slice_result)"""

    #scene.add_geometry(mesh)
    # Show the scene
    #scene.show()
    return grip_point, grip_depth

def generate_grip_lines(mesh, normal, slicing_thickness):
    normal = normal / np.linalg.norm(normal)  # Normalize the vector

    # Calculate the bounds of the mesh along the direction of the slicing plane normal
    cylinder, cylinder_data, cylinder_top, cylinder_bottom = fit_minimum_volume_cylinder_along_axis(mesh.vertices, normal, 32)

    # Create a scene to visualize the slices
    scene = trimesh.Scene()
    # Iterate through the mesh along the axis defined by the plane normal at 5mm intervals
    slice_result = []
    for i in range(0, round(cylinder_data["height"]/slicing_thickness)+1):
        # Define the plane origin for the current slice
        plane_origin = cylinder_top - i*normal*slicing_thickness
        if i == 0:
            plane_origin -= 0.0001*normal
        print(plane_origin)

        # Slice the mesh at the current position
        sr = mesh.section(plane_origin=plane_origin, plane_normal=normal)
        if sr is not None:
            slice_result.append(sr)
        """color = [int(i*255/(round(cylinder_data["height"]/slicing_thickness))), 150, 130, 255]
        slice_result[-1].colors = [color for _ in range(len(slice_result[-1].entities))]
        scene.add_geometry(slice_result)"""


    """trees = []
    for i in range(0, len(slice_result)):
        trees.append(cKDTree(slice_result[i].vertices))
    closest_points = np.full((len(slice_result[0].vertices),3), [0.0,0.0,0.0])
    for i in range(len(slice_result[0].vertices)):
        closest_points[i]=[i, i, 99999999.0]


    for i in range(1, len(trees)):
        for j in range(len(closest_points)):
            distance, index = trees[i].query(slice_result[i-1].vertices[int(closest_points[j][1])])
            closest_points[j][1] = index
            closest_points[j][2] = distance
        "" "for j in range(len(closest_points)):
            distance, index = trees[i-1].query(slice_result[i].vertices[int(closest_points[j][1])])
            if index != closest_points[j][0] and distance > closest_points[j][2]:
                closest_points[j][0] = index
                closest_points[j][2] = distance"" "

            
            
    for i, pair in enumerate(closest_points):#len(slice_result[0].vertex_nodes)):
        # Create a Line object with the defined start and end points
        line_entity = trimesh.path.entities.Line(
            points=[0, 1],
            color=[255, 0, 0, 255]  # RGBA color for the line (red in this case)
        )

        # Create a Path3D object and add the line entity
        path = trimesh.path.Path3D(entities=[line_entity], vertices=[slice_result[0].vertices[int(pair[0])],
                                                                     slice_result[-1].vertices[int(pair[1])]])
        scene.add_geometry([path])

    scene.show()"""


    # Build a KDTree for the vertices of path2
    """tree = cKDTree(slice_result[-1].vertices)

    # Find the closest point in path2 for each point in path1
    closest_points = []
    for point in slice_result[0].vertices:
        distance, index = tree.query(point)
        closest_points.append([point, slice_result[-1].vertices[index]])

    for pair in closest_points:#len(slice_result[0].vertex_nodes)):
        # Create a Line object with the defined start and end points
        line_entity = trimesh.path.entities.Line(
            points=[0, 1],
            color=[255, 0, 0, 255]  # RGBA color for the line (red in this case)
        )

        # Create a Path3D object and add the line entity
        path = trimesh.path.Path3D(entities=[line_entity], vertices=[pair[0],
                                                                     pair[1]])
        scene.add_geometry([path])

    scene.show()"""
    initial_path = []
    for node in slice_result[0].vertex_nodes:
        if len(initial_path)==0:
            initial_path.append(slice_result[0].vertices[node[0]])
        else:
            if np.linalg.norm(initial_path[-1]-slice_result[0].vertices[node[0]])>0.01:
                initial_path.append(slice_result[0].vertices[node[0]])
    tree = cKDTree(slice_result[-1].vertices)
    distance, index = tree.query(initial_path[0])
    closest_points = [[initial_path[0], slice_result[-1].vertices[index]]]
    for point in initial_path:
        distance, index = tree.query(point+closest_points[-1][1]-closest_points[-1][0])
        closest_points.append([point, slice_result[-1].vertices[index]])
        """line_entity = trimesh.path.entities.Line(
            points=[0, 1],
            color=[0, 255, 0, 255]  # RGBA color for the line (red in this case)
        )

        # Create a Path3D object and add the line entity
        path = trimesh.path.Path3D(entities=[line_entity], vertices=[point,
                                                                     point+closest_points[-1][1]-closest_points[-1][0]])
        scene.add_geometry([path])"""
    distance, index = tree.query(initial_path[0]+closest_points[-1][1]-closest_points[-1][0])
    closest_points = [[initial_path[0], slice_result[-1].vertices[index]]]
    for point in initial_path:
        distance, index = tree.query(point+closest_points[-1][1]-closest_points[-1][0])
        closest_points.append([point, slice_result[-1].vertices[index]])

    num = int(len(closest_points)/2)
    minD = 0
    for i, pair in enumerate(closest_points[:num]):  # len(slice_result[0].vertex_nodes)):
        # Create a Line object with the defined start and end points
        line_entity = trimesh.path.entities.Line(
            points=[0, 1],
            color=[178, 55, 44,minD+i*(255-minD)/num]  # RGBA color for the line (red in this case)
        )

        # Create a Path3D object and add the line entity
        path = trimesh.path.Path3D(entities=[line_entity], vertices=[pair[0]-(pair[1]-pair[0]),
                                                                     pair[1]+(pair[1]-pair[0])])
        scene.add_geometry([path])
    scene.add_geometry(mesh)
    scene.show()
    print("done")

def find_opposing_lines(mesh, normal, slicing_thickness):
    normal = normal / np.linalg.norm(normal)  # Normalize the vector

    # Calculate the bounds of the mesh along the direction of the slicing plane normal
    cylinder, cylinder_data, cylinder_top, cylinder_bottom = fit_minimum_volume_cylinder_along_axis(mesh.vertices, normal, 32)

    # Create a scene to visualize the slices
    scene = trimesh.Scene()
    # Iterate through the mesh along the axis defined by the plane normal at 5mm intervals
    slice_result = []
    for i in range(0, round(cylinder_data["height"]/slicing_thickness)+1):
        # Define the plane origin for the current slice
        plane_origin = cylinder_top - i*normal*slicing_thickness
        if i == 0:
            plane_origin -= 0.0001*normal
        print(plane_origin)

        # Slice the mesh at the current position
        sr = mesh.section(plane_origin=plane_origin, plane_normal=normal)
        if sr is not None:
            slice_result.append(sr)
            color = [int(i*255/(round(cylinder_data["height"]/slicing_thickness))), 150, 130, 255]
            slice_result[-1].colors = [color for _ in range(len(slice_result[-1].entities))]
            scene.add_geometry(slice_result)


            slice_2D, to_3D = sr.to_planar()
            #slice_2D.show()
            #slice_2D.medial_axis().show()
            test2 = slice_2D.enclosure_directed.degree()
            colors = [[0, 0, 255, 255] for _ in range(len(sr.entities))]
            print(len(sr.entities), len(test2))
            print(test2)
            print("temp")
            for i in range(len(sr.entities)):
                if i in test2._nodes and test2[i] == 0:
                    colors[i] = [0, 255, 0, 255]
                elif i in test2._nodes and test2[i] == 1:
                    colors[i] = [255, 255, 0, 255]
                elif i in test2._nodes:
                    colors[i] = [255, 0, 0, 255]

            sr.colors = colors
            scene.add_geometry(sr)

            print("temp")

    scene.show()



if __name__ == '__main__':
    mesh = trimesh.load(r"C:\MaH\Data\Parts\EMO_SR\01_upper_lid.stl")
    #mesh = trimesh.load(r"C:\MaH\Data\Parts\AD2V5\V5_Sensor.stl")
    voxels = mesh.voxelized(pitch=10.0)
    voxels.show()

    find_opposing_lines(mesh, np.array([0.0, 0.0, 1.0]), 2.5)

    #mesh = trimesh.load(r"C:\Users\MaH_Admin\Downloads\Spur_M5_Pa20_Z20.stl")
    mesh = trimesh.load(r"C:\Users\MaH_Admin\Downloads\Helical_M5_Pa20_Z20_Ha10.stl")
    plane_normal = np.array([0.0, 0.0, 1.0])
    generate_grip_lines(mesh, plane_normal, 15.0)
    mesh = trimesh.load('C:\MaH\Data\Parts\AD2V5_Test_Sensor_cut\V5_Sensor_cut.stl')
    #plane_normal = np.array([-0.38268, 0.0, 0.92388])  # Adjust this vector for different directions
    plane_normal = np.array([0.0, 0.0, 1.0])  # Adjust this vector for different directions
    gp, depth = find_pertruding_grip_points(mesh, plane_normal)
    sphere = trimesh.creation.icosphere(radius=5, subdivisions=0)
    sphere.vertices += gp
    sphere.visual.vertex_colors = [255, 0, 0, 255]
    mesh.visual.vertex_colors = [128, 128, 128, 128]
    scene = trimesh.Scene([sphere, mesh])
    scene.show()