import trimesh
import math
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize



#TODO: check proximity.max_tangent_sphere as alternative to minimum cylinder approximation
import trimesh
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_screw_dummy(screw_diameter=6.0, screw_length=20.0, head_position=[0, 0, 0], shaft_direction=[0, 0, -1]):
    # Normalize the shaft direction
    shaft_direction = np.array(shaft_direction, dtype=np.float64)
    shaft_direction /= np.linalg.norm(shaft_direction)

    # Default axis of screw is +Z
    default_axis = np.array([0, 0, 1])

    # Create the shaft (along +Z by default)
    shaft_height = screw_length
    shaft_radius = screw_diameter / 2
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_height)

    # Create the head (centered at origin, aligned with +Z)
    head_height = screw_diameter
    head_radius = 1.5 * shaft_radius
    head = trimesh.creation.cylinder(radius=head_radius, height=head_height)

    # Move the head so its top is at z=0
    head.apply_translation([0, 0, -head_height / 2])

    # Move the shaft so its top is at z = -head_height
    shaft.apply_translation([0, 0, -(head_height + shaft_height / 2)])

    # Combine into one screw mesh
    screw = trimesh.util.concatenate([shaft, head])

    # Compute rotation from Z-axis to desired shaft direction
    rotation_vector = np.cross(default_axis, shaft_direction)
    if np.allclose(rotation_vector, 0):
        # Vectors are parallel or anti-parallel
        if np.dot(default_axis, shaft_direction) > 0:
            rotation = R.identity()
        else:
            rotation = R.from_rotvec(np.pi * np.array([1, 0, 0]))  # 180° rotation
    else:
        angle = np.arccos(np.clip(np.dot(default_axis, shaft_direction), -1.0, 1.0))
        rotation = R.from_rotvec(rotation_vector / np.linalg.norm(rotation_vector) * angle)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.as_matrix()
    # Apply rotation
    screw.apply_transform(transform_matrix)

    # Translate so the top of the head sits at the given position
    screw.apply_translation(head_position)

    return screw

def generate_orthogonal_vectors(base_vector):
    base_vector = np.array(base_vector, dtype=np.float64)
    base_vector /= np.linalg.norm(base_vector)  # normalize

    # Step 1: Find a vector orthogonal to base_vector
    if np.allclose(base_vector, [0, 0, 1]) or np.allclose(base_vector, [0, 0, -1]):
        v1 = np.array([1, 0, 0])
    else:
        # Cross with Z to get something perpendicular
        v1 = np.cross(base_vector, [0, 0, 1])
        v1 /= np.linalg.norm(v1)

    # Step 2: Rotate v1 around base_vector in 90-degree steps
    angles = [0, 90, 180, 270]  # degrees
    orthogonal_vectors = []

    for angle in angles:
        rot = R.from_rotvec(np.radians(angle) * base_vector)
        rotated_vec = rot.apply(v1)
        orthogonal_vectors.append(rotated_vec)

    return np.array(orthogonal_vectors)


def distance_point_to_line(point, line_direction, line_point):
    # Convert to NumPy arrays for vectorized calculations
    point = np.array(point)
    line_direction = np.array(line_direction)
    line_point = np.array(line_point)

    # Calculate the vector from the line_point to the point
    vector_to_point = point - line_point

    # Calculate the distance using the perpendicular distance formula
    distance = np.linalg.norm(np.cross(vector_to_point, line_direction)) / np.linalg.norm(line_direction)

    return distance

def fit_cylinder(points, par_sections=256):
    cylinder_data = trimesh.bounds.minimum_cylinder(points)
    # Generate the cylinder
    cylinder = trimesh.creation.cylinder(radius=cylinder_data["radius"], height=cylinder_data["height"],
                                         transform=cylinder_data["transform"], sections=par_sections)

    cylinder_top = np.dot(cylinder_data["transform"], [0, 0, cylinder_data["height"] / 2, 1])[:3]
    cylinder_bottom = np.dot(cylinder_data["transform"], [0, 0, -cylinder_data["height"] / 2, 1])[:3]
    #cylinder.apply_transform(cylinder_data["transform"])
    return cylinder, cylinder_data, cylinder_top, cylinder_bottom

def fit_min_max_cylinder(points, par_sections=256):
    cylinder_max, cylinder_max_data, cylinder_top, cylinder_bottom = fit_cylinder(points, par_sections)
    min_dist = 9999999999999.9
    line_direction = cylinder_top - cylinder_bottom
    for pnt in points:
        calc_dist = distance_point_to_line(pnt, line_direction, cylinder_bottom)
        if calc_dist<min_dist:
            min_dist = calc_dist
    cylinder_min = trimesh.creation.cylinder(radius=min_dist, height=cylinder_max_data["height"],
                                             transform=cylinder_max_data["transform"],sections=par_sections)
   # cylinder_min.apply_transform(cylinder_max_data["transform"])

    return cylinder_max, cylinder_max_data, cylinder_min, min_dist, cylinder_top, cylinder_bottom

def fit_min_max_cylinder_along_axis(points, direction, par_sections=256):
    cylinder_max, cylinder_max_data, cylinder_top, cylinder_bottom = fit_minimum_volume_cylinder_along_axis(points, direction, par_sections)
    min_dist = 9999999999999.9
    line_direction = cylinder_top - cylinder_bottom
    for pnt in points:
        calc_dist = distance_point_to_line(pnt, line_direction, cylinder_bottom)
        if calc_dist < min_dist:
            min_dist = calc_dist
    #print(f"creating cylinder {min_dist} {cylinder_max_data['height']} {cylinder_bottom} {cylinder_top}")
    cylinder_min = trimesh.creation.cylinder(radius=min_dist, height=cylinder_max_data["height"],
                                             transform=cylinder_max_data["transform"], sections=par_sections)
    # cylinder_min.apply_transform(cylinder_max_data["transform"])

    return cylinder_max, cylinder_max_data, cylinder_min, min_dist, cylinder_top, cylinder_bottom


def cylinder_volume(radius, height):
    return np.pi * radius ** 2 * height


def cylinder_objective(center, points, axis_vector):
    axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize axis vector
    vectors_to_points = points - center
    distances = np.array([np.linalg.norm(vtp*axis_vector) for vtp in vectors_to_points])
    height = np.max(distances)

    radius = 0.0
    for pnt in points:
        calc_dist = distance_point_to_line(pnt, axis_vector, center - axis_vector*height)
        if calc_dist>radius:
            radius = calc_dist

    # Compute the volume of the cylinder and use it as penalty
    volume = cylinder_volume(radius, height)

    return volume


def distance_to_points(point, points):
    # Calculate distances from the point to all other points
    distances = np.linalg.norm(points - point, axis=1)
    # Return the maximum distance
    return np.max(distances)


def objective_function(point, *args):
    # Unpack additional arguments
    points = args[0]
    # Calculate negative maximum distance
    return distance_to_points(point, points)


def minimize_maximum_distance(points, initial_guess=None):
    # Define initial guess if not provided
    if initial_guess is None:
        initial_guess = np.mean(points, axis=0)

    # Minimize the negative of the objective function to find the maximum distance
    result = minimize(objective_function, initial_guess, args=(points,), method='Nelder-Mead')

    # Return the optimized point
    return result.x


def fit_minimum_volume_cylinder_along_axis(point_cloud, axis_vector, par_sections = 256):
    axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize axis vector
    #print(f"axis vector: {axis_vector}")
    #distances = np.array([np.linalg.norm(pnt * axis_vector) for pnt in point_cloud])
    points_on_line = np.dot(point_cloud, axis_vector)
    points_on_line = points_on_line[:, np.newaxis] * axis_vector
    signed_distances = np.linalg.norm(points_on_line, axis=1)*np.sign(np.dot(points_on_line, axis_vector)) #TODO: check if this is correct!!! used do be points_on_line*axis_vector
    #min_dist = np.min(distances)
    #max_dist = np.max(distances)
    min_point = axis_vector*np.min(signed_distances)
    max_point = axis_vector*np.max(signed_distances)
    height = np.linalg.norm(max_point-min_point)
    projected_points = np.array([pnt-np.dot(pnt, axis_vector) * axis_vector for pnt in point_cloud])
    #center_pc = np.mean(np.array(point_cloud), axis=0)
    #center_p = np.mean(projected_points, axis=0)
    #center = np.mean(projected_points, axis=0) + axis_vector * (min_dist + height/2)
    #test= np.dot(np.mean(np.array(point_cloud), axis=0), axis_vector)
    center = minimize_maximum_distance(projected_points) + min_point + (max_point-min_point)/2
    radius = 0.0
    for pnt in point_cloud:
        calc_dist = distance_point_to_line(pnt, axis_vector, center - axis_vector * height)
        if calc_dist > radius:
            radius = calc_dist

    # Construct transformation matrix
    if np.array_equal(axis_vector, [0, 0, 1]):
        rotation_matrix = np.eye(3)
    elif np.array_equal(axis_vector, [0, 0, -1]):
        # Directly setting the rotation matrix for 180 degrees around the x-axis
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])
    else:
        # Calculate rotation axis using cross product
        rotation_axis = np.cross(np.array([0,0,1]), axis_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)
        # Calculate rotation angle using dot product
        rotation_angle = np.arccos(np.dot(np.array([0,0,1]), axis_vector))
        # Create rotation object
        rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
        # Get transformation matrix
        rotation_matrix = rotation.as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = center
    cylinder_data = {"radius": radius, "height": height, "transform": transformation_matrix}
    cylinder = trimesh.creation.cylinder(radius=cylinder_data["radius"], height=cylinder_data["height"],
                                         transform=cylinder_data["transform"], sections=par_sections)

    cylinder_top = np.dot(cylinder_data["transform"], [0, 0, cylinder_data["height"] / 2, 1])[:3]
    cylinder_bottom = np.dot(cylinder_data["transform"], [0, 0, -cylinder_data["height"] / 2, 1])[:3]
    return cylinder, cylinder_data, cylinder_top, cylinder_bottom
"""def fit_min_max_cylinder_directed(points, axis, par_sections=256):
    center = np.mean(points)
    #cylinder_max, cylinder_max_data, cylinder_top, cylinder_bottom = fit_cylinder(points, par_sections)
    max_dist = 0.0
    for pnt in points:
        calc_dist = distance_point_to_line(pnt, axis, center)
        if calc_dist>max_dist:
            max_dist = calc_dist
    min_dist = 9999999999999.9
    for pnt in points:
        calc_dist = distance_point_to_line(pnt, axis, center)
        if calc_dist < min_dist:
            min_dist = calc_dist
    cylinder_max = trimesh.creation.cylinder(radius=min_dist, height=cylinder_max_data["height"],
                                             transform=cylinder_max_data["transform"],sections=par_sections)

    cylinder_min = trimesh.creation.cylinder(radius=min_dist, height=cylinder_max_data["height"],
                                             transform=cylinder_max_data["transform"],sections=par_sections)
   # cylinder_min.apply_transform(cylinder_max_data["transform"])

    return cylinder_max, cylinder_max_data, cylinder_min, min_dist, cylinder_top, cylinder_bottom"""
def offset_mesh(mesh, offset_distance):
    # Calculate the surface normals of the original mesh
    normals = mesh.vertex_normals

    # Move each vertex along its normal by the offset distance
    offset_vertices = mesh.vertices + offset_distance * normals

    # Create a new mesh with the offset vertices
    offset_mesh = trimesh.Trimesh(vertices=offset_vertices, faces=mesh.faces)

    return offset_mesh

def scale_mesh_along_vector(mesh, scale_factor, scaling_vector):
    # scales mesh in given direction.
    s_mesh = mesh.copy()
    # Calculate the scaling matrix
    scaling_matrix = np.eye(3) * scale_factor

    # Get the direction vector of the scaling
    scaling_direction = scaling_vector / np.linalg.norm(scaling_vector)

    # Apply the scaling transformation along the direction vector
    scaling_transform = np.eye(4)
    scaling_transform[:3, :3] = np.eye(3) + (scaling_matrix - np.eye(3)) * np.outer(scaling_direction,
                                                                                    scaling_direction)
    s_mesh.apply_transform(scaling_transform)

    return s_mesh

def point_to_mesh_distance(mesh, point):
    # Find the closest point on the mesh to the given point
    closest_point, _, _ = trimesh.proximity.closest_point(mesh, [point])

    # Calculate the distance between the two points
    distance = np.linalg.norm(point - closest_point)

    return distance

def distance_along_vector(point_P, point_Q, vector_V):
    # Calculate the vector between the two points: (Q - P)
    point_difference = point_Q - point_P

    # Calculate the dot product between the vector and the given vector: (Q - P) · V
    dot_product = np.dot(point_difference, vector_V)

    # Calculate the magnitude of the dot product to get the distance
    distance = dot_product

    return distance
def generate_helix_path(distance_between_points, radius, pitch, height):

    circumference = 2 * np.pi * radius
    num_points = max(int(circumference * height / pitch / distance_between_points ),1)
    t = np.linspace(0, height, num_points)
    x = radius * np.cos(2 * np.pi * t / pitch)
    y = radius * np.sin(-2 * np.pi * t / pitch)
    z = -t + height/2

    # Generate circle points for the end
    circle_t = np.linspace(0, 2 * np.pi, int(circumference/distance_between_points))
    circle_x = radius * np.cos(circle_t + 2 * np.pi * t[-1] / pitch)
    circle_y = radius * np.sin(-circle_t - 2 * np.pi * t[-1] / pitch)
    circle_z = z[-1] * np.ones_like(circle_t)

    # Append circle points to the helix path
    x = np.concatenate((x, circle_x))
    y = np.concatenate((y, circle_y))
    z = np.concatenate((z, circle_z))

    return x, y, z


def generate_spiral_above_origin(distance_between_points, initial_radius, pitch, height, z_offset=0.0):
    import numpy as np

    # Calculate the total length of the spiral path
    circumference = 2 * np.pi * initial_radius
    num_points = max(int(circumference * height / pitch / distance_between_points), 1)

    # Linearly decrease radius from initial_radius to 0
    t = np.linspace(0, height, num_points)
    radius = initial_radius * (1 - t / height)  # radius decreases as t increases
    radius[radius < 0] = 0  # Ensure radius doesn't go negative

    # Calculate x, y, z coordinates
    x = radius * np.cos(2 * np.pi * t / pitch)
    y = radius * np.sin(-2 * np.pi * t / pitch)
    z = np.linspace(height, 0, num_points) + z_offset  # Ensure z is always above z_offset

    return x, y, z

def generate_double_spiral_above_origin(distance_between_points, initial_radius, pitch, height, z_offset=0.0):
    import numpy as np

    # Estimate number of points needed
    circumference = 2 * np.pi * initial_radius
    num_points = max(int(circumference * height / pitch / distance_between_points), 2)

    # Create linear parameter along spiral height
    t = np.linspace(0, height, num_points)

    # Compute radius: increase to initial_radius at halfway, then decrease
    radius = np.zeros_like(t)
    midpoint = height / 2
    radius = np.where(
        t <= midpoint,
        initial_radius * (t / midpoint),                          # ramp up
        initial_radius * (1 - (t - midpoint) / (height - midpoint))  # ramp down
    )

    # Compute x, y, z coordinates of the double spiral
    x = radius * np.cos(2 * np.pi * t / pitch)
    y = radius * np.sin(-2 * np.pi * t / pitch)
    z = np.linspace(height, 0, num_points) + z_offset  # descending in z

    return x, y, z

def _to_tmesh(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.TriangleMesh:
    # Convert legacy TriangleMesh -> tensor TriangleMesh
    v = np.asarray(mesh.vertices, dtype=np.float32)
    t = np.asarray(mesh.triangles, dtype=np.int32)
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(v, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(t, dtype=o3d.core.Dtype.Int32),
    )
    return tmesh

def _sample_points_on_mesh(mesh: o3d.geometry.TriangleMesh, n: int) -> np.ndarray:
    # Faster + more uniform than iterating all vertices
    # Uses triangle area sampling internally
    pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    return np.asarray(pcd.points, dtype=np.float32)

def check_reachability_o3d_raycast(
    mesh_to_check: o3d.geometry.TriangleMesh,
    other_mesh: o3d.geometry.TriangleMesh,
    direction: np.ndarray,
    *,
    n_samples: int = 500,
    ignore_near: float = 0.15,
    max_distance: float = 1e6,
    visualize: bool = False,
) -> bool:
    """
    Returns True if rays from sampled points on mesh_to_check along direction
    do NOT hit other_mesh (beyond ignore_near). Otherwise returns False.

    This matches the intent of the old trimesh-based implementation.
    """
    d = np.asarray(direction, dtype=np.float32)
    dn = np.linalg.norm(d)
    if dn == 0:
        raise ValueError("direction must be non-zero")
    d = d / dn

    # Build raycasting scene with other_mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(_to_tmesh(other_mesh))

    # Sample query points on mesh_to_check (avoid per-vertex loop)
    pts = _sample_points_on_mesh(mesh_to_check, n_samples)  # (N,3)

    # Build rays: [ox, oy, oz, dx, dy, dz]
    rays = np.hstack([pts, np.repeat(d[None, :], len(pts), axis=0)]).astype(np.float32)
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_t)
    t_hit = ans["t_hit"].numpy()  # (N,)

    # Open3D returns +inf if no hit
    hit_mask = np.isfinite(t_hit) & (t_hit <= float(max_distance))

    # Ignore very near hits (self-intersection / numerical issues)
    blocking = np.any(hit_mask & (t_hit > float(ignore_near)))

    reachable = not blocking

    if visualize:
        # Minimal visualization: show the two meshes and a few rays
        geoms = []
        m1 = mesh_to_check.paint_uniform_color([0.6, 0.6, 0.6])
        m2 = other_mesh.paint_uniform_color([0.8, 0.8, 0.2])
        geoms += [m1, m2]

        # Show up to 50 rays
        k = min(50, len(pts))
        idx = np.linspace(0, len(pts) - 1, k).astype(int)
        ray_lines = []
        for i in idx:
            p = pts[i]
            # draw a finite segment; if it hits, draw until hit, else draw some length
            L = float(t_hit[i]) if np.isfinite(t_hit[i]) else 50.0
            L = max(0.0, min(L, 50.0))
            ray_lines.append([p, p + d * L])

        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack(ray_lines)),
            lines=o3d.utility.Vector2iVector([[2*i, 2*i+1] for i in range(len(ray_lines))]),
        )
        geoms.append(ls)

        o3d.visualization.draw_geometries(geoms)

    return reachable


#def check_reachability(mesh_to_check, other_mesh, direction, visualize = False):
#    reachable = True
#    #########################
#    if visualize:
#        scene = trimesh.Scene([mesh_to_check,other_mesh])
#    #########################
#    for point in mesh_to_check.vertices:
#        # Perform ray-triangle intersection with the merged meshes
#        try:
#            other_mesh.visual.face_colors = [128,128,0,255]
#            intersects = list(other_mesh.ray.intersects_location(ray_origins=[point], ray_directions=[direction])[0])
#            for i in intersects:
#                if np.linalg.norm(point - i) < 0.15:
#                    intersects.remove(i)
#        except:
#            print("something went wrong")
#            #scene = trimesh.Scene([mesh_to_check, other_mesh])
#            #scene.show()
#            reachable = False #TODO: This is only a quickfix something is not working correctly
#            break
#
#        #####################
#        if visualize:
#            p = trimesh.load_path(np.array([[point, point+direction*100]]))
#            scene.add_geometry(p)
#        #####################
#
#        ####################################################################
#        if visualize:
#            for intersect in intersects:
#                sphere = trimesh.creation.icosphere(radius=0.5, subdivisions=1)
#                sphere.vertices += intersect
#                sphere.visual.face_colors = [255, 0, 0, 255]  # Set the color to red (RGBA format)
#                scene.add_geometry(sphere)
#        ####################################################################
#        if len(intersects) != 0:
#            reachable = False  # Target mesh is visible in this direction
#            break
#    #if reachable:
#    #    print("Target mesh is visible from direction:", direction)
#    ############
#    if visualize:
#        #if not reachable:
#        scene.show()
#    #VISUALIZATION###########
#    return reachable


def unit_vector_to_spherical_coordinate_angles(vec: list):
    # Berechnung der Winkel in Radiant
    theta = math.atan2(vec[1], vec[0])
    phi = math.acos(vec[2])
    return theta, phi

def calculate_new_chord_length(R_start, c_start, R_target):
    """
    Calculate a new chord length to reach a target radius, given that the LONGER circel segment stays constant.

    Parameters:
        R_start (float): Initial radius of the circular segment.
        c_start (float): Initial chord length.
        R_target (float): Target radius.

    Returns:
        float: new chord length required to reach the target radius.
    """
    return 2 * R_target * np.sin(R_start/R_target*(np.pi - np.arcsin(c_start/(2*R_start))))


def compute_center_shift(R_start, c_start, R_new, c_new):
    """
    Compute the vertical shift of the circular segment center given
    initial and new radii and chord widths.

    Parameters:
        R_start (float): Initial radius of the circle segment.
        c_start (float): Initial chord width.
        R_new (float): New radius of the circle segment.
        c_new (float): New chord width.

    Returns:
        float: Vertical shift of the circle center (Δd).
    """
    # Compute the initial and new center heights
    d_start = R_start - np.sqrt(R_start**2 - (c_start / 2)**2)
    d_new = R_new - np.sqrt(R_new**2 - (c_new / 2)**2)

    # Compute shift in center position
    delta_d = d_new - d_start

    return delta_d

def rotation_matrix_from_zx(z_vector, x_direction_hint):
    # Normalize z-axis vector
    z_axis = z_vector / np.linalg.norm(z_vector)

    # Create orthogonal x-axis vector by projecting the hint vector onto the plane perpendicular to z
    x_axis = x_direction_hint - np.dot(x_direction_hint, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    # y-axis from cross product
    y_axis = np.cross(z_axis, x_axis)

    # Re-orthogonalize x-axis (to handle minor numerical issues)
    x_axis = np.cross(y_axis, z_axis)

    # Create rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix


def rotation_matrix_from_zy(z_vector, y_direction_hint):
    """
    Constructs a rotation matrix that aligns:
    - The given `z_vector` with the new z-axis.
    - The given `y_direction_hint` as close as possible to the new y-axis.

    Parameters:
        z_vector (numpy array): The vector that will become the z-axis.
        y_direction_hint (numpy array): The vector guiding the y-axis direction.

    Returns:
        numpy array: A 3x3 rotation matrix.
    """
    # Normalize the z-axis vector
    z_axis = z_vector / np.linalg.norm(z_vector)

    # Create y-axis vector by projecting the hint vector onto the plane perpendicular to z
    y_axis = y_direction_hint - np.dot(y_direction_hint, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)

    # Compute x-axis as the cross product of y and z
    x_axis = np.cross(y_axis, z_axis)

    # Ensure orthogonality by re-adjusting y-axis (for numerical stability)
    y_axis = np.cross(z_axis, x_axis)

    # Construct the rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    return rotation_matrix


def rotation_matrix_to_align_z(v, current_orn_matrix):
    # Convert current orientation quaternion to rotation matrix
    current_rot = R.from_matrix(current_orn_matrix)

    # Transform the global z-axis using the current orientation
    z = current_rot.apply([0, 0, 1])

    # Normalize the target vector
    v = v / np.linalg.norm(v)

    # Compute axis and angle between current z and desired v
    axis = np.cross(z, v)
    dot = np.clip(np.dot(z, v), -1.0, 1.0)
    angle = np.arccos(dot)

    print(f"z: {z}, v: {v}, angle: {np.degrees(angle)}")

    if np.linalg.norm(axis) <= 1e-8:
        # v is parallel or anti-parallel to z
        if np.allclose(v, z):
            return np.eye(3)  # No rotation needed
        else:
            # 180-degree rotation around any axis orthogonal to z
            # Pick x-axis for simplicity
            return R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)

    # Create the rotation matrix
    rotvec = axis * angle
    R_align = R.from_rotvec(rotvec).as_matrix()
    return R_align


def save_o3d_render(pcds, view_params, output_path, resolution=(1920, 1080)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=resolution[0], height=resolution[1])
    for obj in pcds:
        vis.add_geometry(obj)

    ctr = vis.get_view_control()
    ctr.set_lookat(view_params["lookat"])
    ctr.set_up(view_params["up"])
    ctr.set_front(view_params["front"])
    ctr.set_zoom(view_params["zoom"])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(output_path)
    vis.destroy_window()
class UniqueIDgenerator:
    def __init__(self):
        self.counter = -1

    def get_id(self):
        self.counter += 1
        return self.counter

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def generate_surface_plot(angle_resolution = np.pi/8):
        num_parts = 3
        num_long = int(np.round(2 * np.pi / angle_resolution))  # Number of longitudinal lines
        num_lat = int(np.round(np.pi / angle_resolution)) + 1  # Number of latitudinal lines
        data = np.random.randint(0, 2, (num_long, num_lat))

        # Define the colors: 1 is red, 0 is green
        colors = np.array([[0, 150, 130], [178, 55, 44]])

        # Define the row and column labels
        row_labels = ['Row1', 'Row2', 'Row3']
        col_labels = ['Col1', 'Col2', 'Col3']

        # Create a figure with 3x3 subplots
        fig, axs = plt.subplots(num_parts, num_parts, figsize=(6, 6))

        # Plot each matrix
        for i in range(num_parts):
            for j in range(num_parts):
                # Get the current matrix
                matrix = data
                # Convert the matrix to RGB colors
                image = colors[matrix]
                # Plot the matrix
                axs[i, j].imshow(image, aspect='auto')
                axs[i, j].axis('off')

        padding_l = 0.125
        padding_r = 0.05
        # Add row labels
        for i, label in enumerate(reversed(row_labels), start=0):
            fig.text(0.1, padding_l + (1-padding_l-padding_r)/num_parts*i, label, ha='center', va='bottom', rotation='vertical')

        # Add column labels
        for i, label in enumerate(col_labels):
            fig.text(padding_l + (1-padding_l-padding_r)/num_parts*i, 0.9, label, ha='left', va='center')

        plt.show()
    generate_surface_plot()