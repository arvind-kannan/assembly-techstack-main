import pybullet as p
import numpy as np
import copy
from pybullet_industrial.toolpath import ToolPath

def apply_tool_path_rotation(tool_path: ToolPath, rotation):
    path_positions = np.transpose(tool_path.positions)
    path_orientations = np.transpose(tool_path.orientations)
    
    rot_matrix = p.getMatrixFromQuaternion(rotation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    for i in range(len(tool_path)):
        path_positions[i] = rot_matrix @ path_positions[i]
        _, path_orientations[i] = p.multiplyTransforms([0, 0, 0],
                                                       rotation,
                                                       [0, 0, 0],
                                                       path_orientations[i])
    
    tool_path.positions = np.transpose(path_positions)
    tool_path.orientations = np.transpose(path_orientations)

def invert_toolpath(toolpath: ToolPath) -> ToolPath:
    """Creates and returns a new ToolPath with reversed poses and tool activations.

    Args:
        toolpath (ToolPath): The toolpath to invert.

    Returns:
        ToolPath: A new toolpath with all elements reversed.
    """
    # Reverse position, orientation, and tool activation arrays
    reversed_positions = np.flip(copy.deepcopy(toolpath.positions), axis=1)
    reversed_orientations = np.flip(copy.deepcopy(toolpath.orientations), axis=1)
    #reversed_tool_activations = np.flip(toolpath.tool_activations)

    return ToolPath(
        positions=reversed_positions,
        orientations=reversed_orientations
    )

def get_all_active_constraint_ids():
    num_constraints = p.getNumConstraints()
    return [p.getConstraintUniqueId(i) for i in range(num_constraints)]