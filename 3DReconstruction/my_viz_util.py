from utils.slam_helpers import *
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation
import cv2
import os
import json
from importlib.machinery import SourceFileLoader
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from datasets.gradslam_datasets import LagmemoDataset, load_dataset_config
    

def get_dataset(config_dict, basedir, sequence, **kwargs):
    return LagmemoDataset(config_dict, basedir, sequence, **kwargs)
    
def render_with_pose(config: dict, params, w2c, index):
    dataset_config = config["data"]
    device = torch.device(config["primary_device"])
    gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=False,
        use_train_split=False,
    )
    col, dep, intr, pos = dataset[0]
    col = col.permute(2, 0, 1) / 255
    first_frame_w2c = torch.linalg.inv(pos)
    cam = setup_camera(col.shape[2], col.shape[1], intr.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # 要调用transform_to_frame，里面是...
    # 1044 - 1050行 

    #rel_w2c = curr_gt_w2c[-1]
    #rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
    #rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
    #rel_w2c_tran = rel_w2c[:3, 3].detach()
    # Update the camera parameters
    #params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
    #params["cam_trans"][..., time_idx] = rel_w2c_tran
    
    # 然后
    #color, depth, _, gt_pose = dataset[time_idx]
    #gt_w2c = torch.linalg.inv(gt_pose)
    # Process RGB-D Data
    #color = color.permute(2, 0, 1) / 255
    #depth = depth.permute(2, 0, 1)
    #gt_w2c_all_frames.append(gt_w2c)
    #curr_gt_w2c = gt_w2c_all_frames
    
    # 先拿一帧看看
    #color, depth, intrinsics, pose = dataset[0]
    #w2c = torch.linalg.inv(pose)
    #rot = w2c[:3, :3]
    #rot_quat = matrix_to_quaternion(rot)
    #tran = w2c[:3, 3]
    #print("rot", rot)
    #print("rot_quat", rot_quat)
    #print("tran", tran)
    
    #my_w2c = torch.eye(4).cuda().float()
    #my_quat = torch.tensor([[1,0,0,0]], dtype=torch.float32, device=device)
    # my_quat = F.normalize(my_quat)
    #print("my_quat", my_quat)
    #my_w2c[:3, :3] = build_rotation(my_quat)
    #my_w2c[:3, 3] = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    
    transformed_gaussians = {}
    # Transform Centers of Gaussians to Camera Frame
    pts = params['means3D']
    unnorm_rots = params['unnorm_rotations']
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (w2c @ pts4.T).T[:, :3]
    transformed_gaussians['means3D'] = transformed_pts
    transformed_gaussians['unnorm_rotations'] = unnorm_rots
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    im, radius, _, = Renderer(raster_settings=cam)(**rendervar)
    viz_render_im = torch.clamp(im, 0, 1)
    viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
        
    folder = os.path.join(os.path.dirname(__file__), "viz_output")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"test{index}.png")
    cv2.imwrite(path, cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))

        
if __name__ == "__main__":
    experiment = SourceFileLoader(
        "splatam640480.py","/home/sfs/SplaTAM/configs/lagmemo/splatam640480.py"
    ).load_module()
    params = np.load("/home/sfs/SplaTAM/data_for_gsmodel/room1_wxl/60/0/params.npz")
    
    params = dict(params)
    
    for k, v in params.items():
        params[k] = torch.from_numpy(v).cuda()
    
    camera_pos = torch.tensor([0.0, 0.0, 0.0], device='cuda')  # x, y, z
    camera_yaw = 0.0  # degrees
    step_size = 0.3  # movement step size
    turn_angle = 15  # rotation step in degrees
    
    index = 0
    
    while True:
    
        command = input("Enter command (w/a/s/d/q/e/exit): ").strip().lower()
        
        # Process command
        if command == 's':
            dx = torch.sin(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            dz = -torch.cos(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            camera_pos[0] += dx
            camera_pos[2] += dz
        elif command == 'w':
            dx = -torch.sin(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            dz = torch.cos(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            camera_pos[0] += dx
            camera_pos[2] += dz
        elif command == 'd':
            dx = torch.cos(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            dz = torch.sin(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            camera_pos[0] += dx
            camera_pos[2] += dz
        elif command == 'a':
            dx = -torch.cos(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            dz = -torch.sin(torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))) * step_size
            camera_pos[0] += dx
            camera_pos[2] += dz
        elif command == 'q':
            camera_yaw += turn_angle
        elif command == 'e':
            camera_yaw -= turn_angle
        elif command == 'exit':
            exit()
        else:
            print("Unknown command")
            continue
        
        
        camera_yaw = camera_yaw % 360
        
        # Build rotation matrix from yaw
        theta = torch.deg2rad(torch.tensor(camera_yaw, device='cuda'))
        qw = torch.cos(theta / 2)
        qy = torch.sin(theta / 2)
        my_quat = torch.tensor([[qw, 0.0, qy, 0.0]], dtype=torch.float32, device='cuda')
        R = build_rotation(my_quat).squeeze(0)
        
        # Build world-to-camera matrix
        t_tensor = camera_pos.unsqueeze(1)
        print("R", R)
        print("t_tensor", t_tensor)
        rotated_t = torch.mm(R, t_tensor).squeeze()
        w2c = torch.eye(4, device='cuda')
        w2c[:3, :3] = R
        w2c[:3, 3] = -rotated_t
        
        render_with_pose(experiment.config, params, w2c, index)
        index += 1
     