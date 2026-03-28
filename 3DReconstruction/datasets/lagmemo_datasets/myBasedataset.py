"""
PyTorch dataset classes for GradSLAM v1.0.

The base dataset class now loads one sequence at a time
(opposed to v0.1.0 which loads multiple sequences).

A few parts of this code are adapted from NICE-SLAM
https://github.com/cvg/nice-slam/blob/645b53af3dc95b4b348de70e759943f7228a61ca/src/utils/datasets.py
"""

import abc
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import yaml
from natsort import natsorted

from .geometryutils import relative_transformation
from . import datautils


def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the input to a scalar
    """
    if isinstance(inp, float):
        return inp

    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()

    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(K):
    """
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


class BaseLagmemoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        # self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose

        self.first_frame_pose = None
        # shape: (4, 4)
        
        self.poses = [] # 这里记的直接就是相对位姿了
        # shape: (N, 4, 4)
        
        self.rgb_paths = []
        self.depth_paths = []
        
        self.pose_path = None
        self.input_folder = None

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = (
            config_dict["camera_params"]["crop_size"]
            if "crop_size" in config_dict["camera_params"]
            else None
        )

        self.crop_edge = None
        if "crop_edge" in config_dict["camera_params"].keys():
            self.crop_edge = config_dict["camera_params"]["crop_edge"]
            
    def _to_pose_matrix(self, rotation, translation):
        """
            在我们的数据集中, rotation: (4, ), translation: (3, )
            返回tensor c2w, shape: (4, 4)
        """
        q = np.array([float(x) for x in rotation])
        t = np.array([float(x) for x in translation])
        R_c2w = self.quaternion_to_rotation_matrix(q)
        R_w2c = np.linalg.inv(R_c2w)
        R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = R_RUB_to_RDF @ R_w2c
        t = R_RUB_to_RDF @ t
        
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = t
        c2w = torch.tensor(c2w, dtype=torch.float32)
        
        return c2w

    def add_frame(self, rgb_path, depth_path, pose, is_first_frame=False):
        """
            pose: 4x4的矩阵, 所以还要加转换函数, 从旋转和平移的形式转换成4x4的矩阵
            rgb_path: str
            depth_path: str
        """
        
        if is_first_frame:
            self.first_frame_pose = pose
        self.poses.append(self._preprocess_poses(pose)) # 好像还要注意维度
        self.rgb_paths.append(rgb_path)
        self.depth_paths.append(depth_path)   
        
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/img*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/img*.npy"))
        embedding_paths = None
        return color_paths, depth_paths, embedding_paths
    
    def work_on_full_sequence(self):
        """
            对于目前直接给所有帧的情况, 进行处理
        """
        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(len(self.color_paths)):
            line = lines[i]
            line = line.split()
            index = line[0]
            q = line[1:5]
            position = line[5:]
            pose = self._to_pose_matrix(q, position)
            self.add_frame(self.color_paths[i], self.depth_paths[i], pose, is_first_frame=(i==0))
        
        
    def __len__(self):
        return len(self.rgb_paths)
    

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        color = color[:, :, :3]
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        # depth = cv2.resize(
        #    depth.astype(float),
        #    (self.desired_width, self.desired_height),
        #    interpolation=cv2.INTER_NEAREST,
        # )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        # return depth / self.png_depth_scale
        return depth

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        
        #return relative_transformation(
        #    poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
        #    poses,
        #    orthogonal_rotations=False,
        #)
        
        # in this case, poses is a 4x4 matrix
        output = relative_transformation(
            self.first_frame_pose.unsqueeze(0), # 1x4x4
            poses.unsqueeze(0), # 1x4x4
            orthogonal_rotations=False,
        )
        return output.squeeze(0) # 4x4

    def get_cam_K(self):
        """
        Return camera intrinsics matrix K

        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    def read_embedding_from_file(self, embedding_path: str):
        """
        Read embedding from file and process it. To be implemented in subclass for each dataset separately.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        # if ".png" in depth_path:
        # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        #    depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
        # elif ".exr" in depth_path:
        #    depth = readEXR_onlydepth(depth_path)

        depth = np.load(depth_path)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        color = torch.from_numpy(color)
        K = torch.from_numpy(K)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(
            K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),  # Allow embedding to be another dtype
                # self.retained_inds[index].item(),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )
        
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
