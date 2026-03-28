import glob
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class LagmemoDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "local_pos.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_dir = "rgb"
        if not os.path.isdir(os.path.join(self.input_folder, color_dir)):
            color_dir = "images"
        color_paths = natsorted(
            glob.glob(f"{self.input_folder}/{color_dir}/img*.png")
        )
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/img*.npy"))
        embedding_paths = None
        return color_paths, depth_paths, embedding_paths
    
    # def load_poses(self):
    #     """
    #     从 local_pos.txt 读取相机位姿（四元数 + 平移），
    #     直接构造 camera-to-world 矩阵，不做 RUB→RDF 坐标系转换
    #     """
    #     poses = []
    #     with open(self.pose_path, "r") as f:
    #         lines = f.readlines()
        
    #     for i in range(self.num_imgs):
    #         line = lines[i].split()
    #         index = line[0]
    #         q = np.array([float(x) for x in line[4:]])  # 四元数
    #         position = np.array([float(x) for x in line[1:4]])  # 平移
    
    #         # 四元数 → 旋转矩阵
    #         R_c2w = self.quaternion_to_rotation_matrix(q)
    
    #         # 构造齐次变换矩阵
    #         c2w = np.eye(4, dtype=np.float32)
    #         c2w[:3, :3] = R_c2w       # 直接用 c2w
    #         c2w[:3, 3] = position     # 直接用原始平移
    
    #         # 转为 torch tensor
    #         c2w = torch.tensor(c2w, dtype=torch.float32)
    #         poses.append(c2w)
    
    #     return poses

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        print(f"[LagmemoDataset] num_imgs={self.num_imgs}, pose_lines={len(lines)}")
        n = min(self.num_imgs, len(lines))
        if n == 0:
            raise RuntimeError(
                f"No poses loaded. num_imgs={self.num_imgs}, pose_lines={len(lines)}"
            )
        for i in range(self.num_imgs):
            line = lines[i]
            line = line.split()
            index = line[0]
            q = line[1:5]
            position = line[5:]
            q = np.array([float(x) for x in q])

            R_c2w = self.quaternion_to_rotation_matrix(q)
            R_w2c = np.linalg.inv(R_c2w)

            R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            R = R_RUB_to_RDF @ R_w2c

            position = np.array([float(x) for x in position])
            t = R_RUB_to_RDF @ position

            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = t

            c2w = torch.tensor(c2w, dtype=torch.float32)
            poses.append(c2w)

            """
            R_c2w = self.quaternion_to_rotation_matrix(q)

            position = np.array([float(x) for x in position])

            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = position

            # transform = np.array(
            #    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            # )
            # c2w = c2w @ transform

            c2w = torch.tensor(c2w, dtype=torch.float32)
            poses.append(c2w)
            """

        return poses

    def quaternion_to_rotation_matrix(self, q):
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        R = np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )
        return R
