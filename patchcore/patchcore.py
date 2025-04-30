"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
import numpy as np
import torch
import tqdm
import open3d as o3d
from patchcore.utils import fix_seeds
from M3DM.models import Model
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(
            self,
            device,
            template=None,
            temp_fpfh=None,
            temp_cur=None,
            **kwargs,):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        fix_seeds(42)
        self.device = device
        # numpy [点数, 3]
        self.template = template
        # numpy [点数, 33]
        self.temp_fpfh = temp_fpfh
        # numpy [点数, 1]
        self.temp_cur = temp_cur

        self.deep_feature_extractor = Model(device='cuda',
                                            group_size=128,
                                            num_group=4096)
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()
        self.image_preds_pmae = list()
        self.image_preds_fpfh = list()
        self.image_labels = list()
        self.pixel_preds_pmae = list()
        self.pixel_preds_fpfh = list()
        self.pixel_labels = list()

    def _embed_pointmae(self, pc, pc_cur):
        temp_cur = torch.from_numpy(self.temp_cur).cuda().float()
        pc_cur = pc_cur.cuda().float()
        pc = pc.cuda().float()
        temp_pc = torch.from_numpy(self.template)
        temp_pc = temp_pc.unsqueeze(0).cuda().float()
        pc_features, temp_features, pc_center, pc_center_idx = self.deep_feature_extractor(pc, temp_pc, pc_cur, temp_cur)

        # pc_features: [1, 1152, 中心点数量]
        # pc_center(1, 中心点数量, 3): 中心点坐标
        # pc_center_idx(1, 中心点数量): 中心点索引
        return pc_features.cpu(), temp_features.cpu(), pc_center.cpu(), pc_center_idx.cpu()

    def _embed_fpfh(self, pc_fpfh, pc_center, pc_center_idx):
        # pc_fpfh
        pc_fpfh = pc_fpfh.squeeze(0).numpy().astype(np.float32)
        pc_fpfh = pc_fpfh[pc_center_idx.squeeze(0).long(), :]
        # temp_fpfh
        temp_pc = torch.from_numpy(self.template).float()
        dist = torch.cdist(pc_center.squeeze(0), temp_pc, p=2)
        temp_center_idx = torch.argmin(dist, dim=1)
        temp_center = temp_pc[temp_center_idx.long(), :]
        temp_fpfh = self.temp_fpfh[temp_center_idx.squeeze(0).long(), :]

        pc_fpfh = torch.from_numpy(pc_fpfh)
        temp_fpfh = torch.from_numpy(temp_fpfh)
        temp_center = temp_center.unsqueeze(0)
        return pc_fpfh, temp_fpfh, temp_center

    def pame_score_map(self, pc_features, temp_features, pc, pc_center, temp_center, pc_center_idx):
        anomaly_map = torch.zeros(pc_features.shape[0])
        center_dist = torch.cdist(pc_center.squeeze(0), temp_center.squeeze(0), p=2)
        pc_temp_dist = torch.cdist(pc_features, temp_features, p=2)
        for i in range(pc_features.shape[0]):
            anomaly_map[i] = pc_temp_dist[i, i] * center_dist[i, i]

        anomaly_score = torch.max(anomaly_map)
        # 配准前的中心点坐标(1, 中心点数量, 3)
        sampling_center = pc[:, pc_center_idx.squeeze(0).long(), :]
        # anomaly_map插值到全部点
        # 创建最近邻居模型
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(sampling_center[0])
        # 找到每个点的最近邻居
        distances, indices = nn.kneighbors(pc[0])
        anomaly_map = torch.mean(anomaly_map[indices], axis=1)

        return anomaly_score, anomaly_map

    def fpfh_score_map(self, pc_fpfh, temp_fpfh, pc, pc_center_idx):
        anomaly_map = torch.zeros(pc_fpfh.shape[0])
        for i in range(pc_fpfh.shape[0]):
            anomaly_map[i] = torch.max(torch.abs(pc_fpfh[i] - temp_fpfh[i]))

        anomaly_score = torch.max(anomaly_map)
        # anomaly_map插值到全部点
        sampling_center = pc[:, pc_center_idx.squeeze(0).long(), :]
        # 创建最近邻居模型
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(sampling_center[0])
        # 找到每个点的最近邻居
        distances, indices = nn.kneighbors(pc[0])
        anomaly_map = torch.mean(anomaly_map[indices], axis=1)

        return anomaly_score, anomaly_map

    def predict(self, dataloader):
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            # pc: Tensor[1, 点的数量, 3]
            # fpfh: Tensor[1, 点的数量, 33]
            # mask: Tensor[1, 点的数量]
            # label: Tensor[1]
            for input_pc, fpfh, mask, label in data_iterator:
                with torch.no_grad():
                    # 曲率
                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_pc.squeeze(0).numpy()))
                    pcd.estimate_covariances(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
                    covs = np.asarray(pcd.covariances)
                    vals, vecs = np.linalg.eig(covs)
                    cur_pc = np.min(vals, axis=1) / np.sum(vals, axis=1)
                    cur_pc = torch.from_numpy(cur_pc)

                    # 特征提取
                    # reg_pc: numpy [点数, 3]
                    # pc_features: [1, 1152, 中心点数量]
                    # pc_center(1, 中心点数量, 3): 中心点坐标
                    # pc_center_idx(1, 中心点数量): 中心点索引
                    pc_features, temp_features, pc_center, pc_center_idx = self._embed_pointmae(input_pc, cur_pc)
                    # FPFH特征 [点数， 33]
                    pc_fpfh, temp_fpfh, temp_center = self._embed_fpfh(fpfh, pc_center, pc_center_idx)

                    # 计算pmae特征的score和map
                    # [1, 1152, 中心点数量]->[中心点数量, 1152]
                    pc_features = pc_features.squeeze(0).permute(1, 0)
                    temp_features = temp_features.squeeze(0).permute(1, 0)
                    pmae_score, pmae_map = self.pame_score_map(pc_features, temp_features, input_pc, pc_center, temp_center, pc_center_idx)
                    self.image_preds_pmae.append(pmae_score.numpy())
                    self.image_labels.append(label.numpy())
                    self.pixel_preds_pmae.extend(pmae_map.flatten().numpy())
                    self.pixel_labels.extend(mask.flatten().numpy())
                    # 计算FPFH特征的score和map
                    fpfh_score, fpfh_map = self.fpfh_score_map(pc_fpfh, temp_fpfh, input_pc, pc_center_idx)
                    self.image_preds_fpfh.append(fpfh_score.numpy())
                    self.pixel_preds_fpfh.extend(fpfh_map.flatten().numpy())

    
    def eval(self, dataset_name):
        self.image_preds_pmae = np.array(self.image_preds_pmae)
        self.pixel_preds_pmae = np.array(self.pixel_preds_pmae)
        self.image_preds_fpfh = np.array(self.image_preds_fpfh)
        self.pixel_preds_fpfh = np.array(self.pixel_preds_fpfh)

        image_pmae_max = np.max(self.image_preds_pmae)
        image_pmae_min = np.min(self.image_preds_pmae)
        self.image_preds_pmae = (self.image_preds_pmae - image_pmae_min) / (image_pmae_max - image_pmae_min)
        pixel_pmae_max = np.max(self.pixel_preds_pmae)
        pixel_pmae_min = np.min(self.pixel_preds_pmae)
        self.pixel_preds_pmae = (self.pixel_preds_pmae - pixel_pmae_min) / (pixel_pmae_max - pixel_pmae_min)

        image_fpfh_max = np.max(self.image_preds_fpfh)
        image_fpfh_min = np.min(self.image_preds_fpfh)
        self.image_preds_fpfh = (self.image_preds_fpfh - image_fpfh_min) / (image_fpfh_max - image_fpfh_min)
        pixel_fpfh_max = np.max(self.pixel_preds_fpfh)
        pixel_fpfh_min = np.min(self.pixel_preds_fpfh)
        self.pixel_preds_fpfh = (self.pixel_preds_fpfh - pixel_fpfh_min) / (pixel_fpfh_max - pixel_fpfh_min)

        # 指标
        image_pmae = round(roc_auc_score(self.image_labels, self.image_preds_pmae), 4)
        pixel_pmae = round(roc_auc_score(self.pixel_labels, self.pixel_preds_pmae), 4)

        image_fpfh = round(roc_auc_score(self.image_labels, self.image_preds_fpfh), 4)
        pixel_fpfh = round(roc_auc_score(self.pixel_labels, self.pixel_preds_fpfh), 4)

        image_mul = self.image_preds_pmae * self.image_preds_fpfh
        pixel_mul = self.pixel_preds_pmae * self.pixel_preds_fpfh
        image_mul = round(roc_auc_score(self.image_labels, image_mul), 4)
        pixel_mul = round(roc_auc_score(self.pixel_labels, pixel_mul), 4)

        print(dataset_name, '\timage_pmae:', image_pmae, '\tpixel_pmae:', pixel_pmae)
        print(dataset_name, '\timage_fpfh:', image_fpfh, '\tpixel_fpfh:', pixel_fpfh)
        print(dataset_name, '\timage_mul:', image_mul, '\tpixel_mul:', pixel_mul)

        return image_pmae, pixel_pmae, image_fpfh, pixel_fpfh, image_mul, pixel_mul



