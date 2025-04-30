import glob
import os
import open3d as o3d
import numpy as np
import pathlib
import time
from time import strftime, gmtime
from patchcore.utils import fix_seeds



def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # estimate_normals计算每个点的法线。 该函数查找相邻点，并使用协方差分析计算相邻点的主轴。
    # 该函数将KDTreeSearchParamHybrid类的实例作为参数, 其两个关键参数radius = 0.1和max_nn = 30指定搜索半径和最大最近邻居。
    # 它的搜索半径为10厘米，最多可考虑30个邻居，以节省计算时间。
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source_data, target_data):
    # 构建3D点云对象
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_data)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # 基于ransac的点云配准
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # 使用 ICP 进行精细配准
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )

    return result


def get_registration_np(source_data, target_data):
    voxel_size = 0.25  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source_data, target_data)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    source.transform(result_ransac.transformation)
    return np.asarray(source.points)



if __name__ == "__main__":
    fix_seeds(0)
    root_dir = 'Real3D-AD'
    real_3d_classes = ['airplane', 'car', 'candybar', 'chicken', 'diamond', 'duck',
                       'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees']
    voxel_size = 0.25

    for _, dataset_name in enumerate(real_3d_classes):
        start_time = time.time()
        if (not os.path.exists(root_dir + dataset_name + '/reg_pc')):
            os.makedirs(root_dir + dataset_name + '/reg_pc')
            os.makedirs(root_dir + dataset_name + '/reg_fpfh')

        temp_list = glob.glob(str(os.path.join(root_dir, dataset_name, 'train')) + '/*template*.pcd')[:1]
        temp_pc = o3d.io.read_point_cloud(temp_list[0])
        temp_pc = np.array(temp_pc.points)
        center = np.average(temp_pc, axis=0)
        temp_pc = temp_pc - np.expand_dims(center, axis=0)
        np.save(root_dir + dataset_name + '/temp_pc.npy', temp_pc.astype(np.float32))
        #************************************************************************************
        o3d_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(temp_pc))
        radius_normal = voxel_size * 2
        o3d_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        temp_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_temp,
                                                                  o3d.geometry.KDTreeSearchParamHybrid(
                                                                      radius=radius_feature, max_nn=100))
        temp_fpfh = temp_fpfh.data.T
        np.save(root_dir + dataset_name + '/temp_fpfh.npy', temp_fpfh.astype(np.float32))
        # ************************************************************************************
        sample_list = glob.glob(str(os.path.join(root_dir, dataset_name, 'test')) + '/*.pcd')
        gt_path = str(os.path.join(root_dir, dataset_name, 'gt'))
        # 对每一个类
        for idx in range(len(sample_list)):
            if 'good' in sample_list[idx]:
                pcd = o3d.io.read_point_cloud(sample_list[idx])
                pointcloud = np.array(pcd.points)
            else:
                filename = pathlib.Path(sample_list[idx]).stem
                txt_path = os.path.join(gt_path, filename + '.txt')
                pcd = np.genfromtxt(txt_path, delimiter=" ")
                pointcloud = pcd[:, :3]
            center = np.average(pointcloud, axis=0)
            pc = pointcloud - np.expand_dims(center, axis=0)
            # ************************************************************************************
            # reg_pc
            save_name = pathlib.Path(sample_list[idx]).stem
            pc = get_registration_np(pc, temp_pc)
            save_dir = root_dir + dataset_name + '/reg_pc/' + save_name + ".npy"
            np.save(save_dir, pc.astype(np.float32))
            # ************************************************************************************
            # FPFH
            o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
            radius_normal = voxel_size * 2
            o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

            radius_feature = voxel_size * 5
            pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc,
                                                                        o3d.geometry.KDTreeSearchParamHybrid(
                                                                            radius=radius_feature, max_nn=100))
            pc_fpfh = pc_fpfh.data.T
            save_dir = root_dir + dataset_name + '/reg_fpfh/' + save_name + ".npy"
            np.save(save_dir, pc_fpfh.astype(np.float32))


        end_time = time.time()
        print(f'Finished on {dataset_name}: ', strftime('%H:%M:%S', gmtime(end_time - start_time)))




