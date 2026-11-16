import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 2
    #radius_normal = voxel_size * 4

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    """mutual_filter = False
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter, distance_threshold)"""

    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000*2, 0.999999))
    return result

def calculate_registration(source_pcd, target_pcd, voxelsize_coarse = 1.0, voxel_size_fine = 0.25, performe_icp = True,
                           estimation_methode_icp = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                           threshold_icp = None):

    source_pcd_down = source_pcd.voxel_down_sample(voxel_size=voxelsize_coarse)
    target_pcd_down = target_pcd.voxel_down_sample(voxel_size=voxelsize_coarse)

    source_pcd_down_fphf = preprocess_point_cloud(source_pcd_down, voxelsize_coarse)
    target_pcd_down_fphf = preprocess_point_cloud(target_pcd_down, voxelsize_coarse)
    global_transform = np.eye(4)


    result_ransac = execute_global_registration(source_pcd_down, target_pcd_down,
                                                source_pcd_down_fphf, target_pcd_down_fphf, voxelsize_coarse)
    global_transform = np.dot(global_transform, result_ransac.transformation)
    source_pcd_down.transform(global_transform)
    """print("Displaying aligned point clouds (red: source, blue: target)")
    o3d.visualization.draw_geometries([source_pcd_down.paint_uniform_color([1, 0, 0]),
                                       target_pcd_down.paint_uniform_color([0, 0, 1])])"""
    if not performe_icp:
        return global_transform
    else:
        if threshold_icp is None:
            threshold_icp = voxel_size_fine * 3.0
        # downsample pointclouds
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size=voxel_size_fine)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size=voxel_size_fine)


        # fine  alignment using icp
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd_down, target_pcd_down, threshold_icp, global_transform,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
            estimation_method=estimation_methode_icp)

        print(icp_result)
        print("Transformation is:")
        print(icp_result.transformation)

        return icp_result.transformation


