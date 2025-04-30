import numpy as np
import torch
import patchcore.patchcore
from patchcore.utils import fix_seeds
import patchcore.metrics
import dataset_pc
from dataset_pc import Dataset3dad_test
from torch.utils.data import DataLoader
import time
from time import strftime, gmtime
import open3d as o3d

fix_seeds(42)

def run():
    device = torch.device("cuda:0")
    device_context = torch.cuda.device("cuda:0")
    root_dir = 'Real3D-AD'
    print('Task start: Real3D-AD')

    real_3d_classes = dataset_pc.real3d_classes()

    image_pmae = 0
    pixel_pmae = 0
    image_fpfh = 0
    pixel_fpfh = 0
    image_mul = 0
    pixel_mul = 0

    # 对于每个类
    for dataset_count, dataset_name in enumerate(real_3d_classes):
        start_time = time.time()
        test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, True), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=False)

        temp_pc = np.load(root_dir + dataset_name + '/temp_pc.npy')
        temp_fpfh = np.load(root_dir + dataset_name + '/temp_fpfh.npy')
        temp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(temp_pc))
        temp_pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
        covs = np.asarray(temp_pcd.covariances)
        vals, vecs = np.linalg.eig(covs)
        temp_cur = np.min(vals, axis=1) / np.sum(vals, axis=1)
        
        with device_context:
            PatchCore = patchcore.patchcore.PatchCore(
                device=device,
                template=temp_pc,
                temp_fpfh=temp_fpfh,
                temp_cur=temp_cur
            )

            PatchCore.predict(test_loader)

            image_rocauc_pmae, pixel_rocauc_pmae, image_rocauc_fpfh, pixel_rocauc_fpfh, image_rocauc_mul, pixel_rocauc_mul = PatchCore.eval(dataset_name)

            image_pmae = image_pmae + image_rocauc_pmae
            pixel_pmae = pixel_pmae + pixel_rocauc_pmae
            image_fpfh = image_fpfh + image_rocauc_fpfh
            pixel_fpfh = pixel_fpfh + pixel_rocauc_fpfh
            image_mul = image_mul + image_rocauc_mul
            pixel_mul = pixel_mul + pixel_rocauc_mul

        end_time = time.time()
        print('running time: ', strftime('%H:%M:%S', gmtime(end_time - start_time)))

    image_pmae = round(image_pmae / len(real_3d_classes), 4)
    pixel_pmae = round(pixel_pmae / len(real_3d_classes), 4)
    image_fpfh = round(image_fpfh / len(real_3d_classes), 4)
    pixel_fpfh = round(pixel_fpfh / len(real_3d_classes), 4)
    image_mul = round(image_mul / len(real_3d_classes), 4)
    pixel_mul = round(pixel_mul / len(real_3d_classes), 4)
    print('Mean', '\timage_pmae:', image_pmae, '\tpixel_pmae:', pixel_pmae)
    print('Mean', '\timage_fpfh:', image_fpfh, '\tpixel_fpfh:', pixel_fpfh)
    print('Mean', '\timage_mul:', image_mul, '\tpixel_mul:', pixel_mul)


if __name__ == "__main__":
    run()
