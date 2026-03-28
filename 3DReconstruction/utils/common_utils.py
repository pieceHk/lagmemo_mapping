import os

import numpy as np
import random
import torch
from torch import nn


def seed_everything(seed=42):
    """
    Set the `seed` value for torch and numpy seeds. Also turns on
    deterministic execution for cudnn.

    Parameters:
    - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params_pth(output_params, variables, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "chkpnt30000.pth")
    # add the variables to the output_params
    output_params.update(variables)
    # torch.save((output_params, 30000), save_path)

    num = output_params["logit_opacities"].shape[0]
    sh_degree = 0
    xyz = nn.Parameter(output_params["means3D"])
    features_dc = nn.Parameter(output_params["rgb_colors"])  # (num, 3)
    # (num, 3) -> (num, 1, 3)
    features_dc = features_dc.unsqueeze(1)
    featrues_rest = None
    scaling = nn.Parameter(output_params["log_scales"])  # (num, 1)
    # (num, 1) -> (num, 3)
    scaling = torch.cat([scaling, scaling, scaling], dim=1)
    standard_rotation = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
    # rotation : num * 4, each row is a standard rotation
    rotation = nn.Parameter(standard_rotation.repeat(num, 1))
    # move to gpu
    rotation = rotation.cuda()

    opacity = nn.Parameter(output_params["logit_opacities"])
    max_radii2D = variables["max_2D_radius"]
    xyz_gradient_accum = variables["means2D_gradient_accum"]  # (num)
    # (num) -> (num, 1)
    xyz_gradient_accum = xyz_gradient_accum.unsqueeze(1)
    denom = variables["denom"]
    denom = denom.unsqueeze(1)
    state_dict = {}
    spatial_lr_scale = None

    tuple = (
        sh_degree,
        xyz,
        features_dc,
        featrues_rest,
        scaling,
        rotation,
        opacity,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        state_dict,
        spatial_lr_scale,
    )
    torch.save((tuple, 30000), save_path)

    """return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )"""


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)
    
def save_w2c(output_params, output_dir):
    to_save = params2cpu(output_params)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving w2c to: {output_dir}")
    save_path = os.path.join(output_dir, "w2c.npz")
    new_data = {"gt_w2c_all_frames": to_save["gt_w2c_all_frames"]}
    np.savez(save_path, **new_data)


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params" + str(time_idx) + ".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir, time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params" + str(time_idx) + ".npz")
    np.savez(save_path, **params_to_save)
