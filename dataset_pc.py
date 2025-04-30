import pathlib
from torch.utils.data import Dataset
import glob
import os
import open3d as o3d
import numpy as np

def real3d_classes():
    return ['airplane',
            'car',
            'candybar',
            'chicken',
            'diamond',
            'duck',
            'fish',
            'gemstone',
            'seahorse',
            'shell',
            'starfish',
            'toffees'
            ]

class Dataset3dad_train(Dataset):
    def __init__(self, dataset_dir, cls_name, if_norm=True):
        self.dataset_dir = dataset_dir
        self.train_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'train')) + '/*template*.pcd')
        self.if_norm = if_norm

    def norm_pcd(self, point_cloud):
        center = np.average(point_cloud,axis=0)
        new_points = point_cloud-np.expand_dims(center, axis=0)
        return new_points

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.train_sample_list[idx])
        pointcloud = np.array(pcd.points)
        if(self.if_norm):
            pointcloud = self.norm_pcd(pointcloud)

        mask = np.zeros((pointcloud.shape[0]))
        label = 0
        return pointcloud, mask, label, self.train_sample_list[idx]

    def __len__(self):
       return len(self.train_sample_list)


class Dataset3dad_test(Dataset):
    def __init__(self, dataset_dir, cls_name, if_norm=True):
        self.dataset_dir = dataset_dir
        self.if_norm = if_norm
        test_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'test')) + '/*.pcd')
        test_sample_list = [s for s in test_sample_list if 'temp' not in s]

        self.test_sample_list = test_sample_list
        self.gt_path = str(os.path.join(dataset_dir, cls_name, 'gt'))
        self.data_name = cls_name

    def __getitem__(self, idx):
        sample_path = self.test_sample_list[idx]
        if 'good' in sample_path:
            filename = pathlib.Path(sample_path).stem
            pcd = np.load(self.dataset_dir + self.data_name + '/reg_pc/' + filename + ".npy")
            fpfh = np.load(self.dataset_dir + self.data_name + '/reg_fpfh/' + filename + '.npy')
            pointcloud = pcd.astype(np.float32)
            fpfh = fpfh.astype(np.float32)
            mask = np.zeros((pointcloud.shape[0]))
            label = 0
        else:
            filename = pathlib.Path(sample_path).stem
            pcd = np.load(self.dataset_dir + self.data_name + '/reg_pc/' + filename + ".npy")
            fpfh = np.load(self.dataset_dir + self.data_name + '/reg_fpfh/' + filename + '.npy')
            pointcloud = pcd.astype(np.float32)
            fpfh = fpfh.astype(np.float32)
            txt_path = os.path.join(self.gt_path, filename + '.txt')
            mask = np.genfromtxt(txt_path, delimiter=" ")
            mask = mask[:, 3]
            label = 1

        return pointcloud, fpfh, mask, label

    def __len__(self):
        return len(self.test_sample_list)