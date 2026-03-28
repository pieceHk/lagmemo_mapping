import argparse
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
import cv2
from tqdm import tqdm


def read_trajectory(traj_path):
    """读取SplaTAM格式的相机轨迹文件"""
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 8:
                poses.append(vals)
    return np.array(poses)


def interp_poses(pose1, pose2, num_steps):
    """在两个相机位姿之间进行平滑插值"""
    t0, tx0, ty0, tz0, qx0, qy0, qz0, qw0 = pose1
    t1, tx1, ty1, tz1, qx1, qy1, qz1, qw1 = pose2

    # 平移插值
    trans_interp = np.linspace([tx0, ty0, tz0], [tx1, ty1, tz1], num_steps)

    # 四元数插值 (Slerp)
    key_times = [0, 1]
    key_rots = R.from_quat([[qx0, qy0, qz0, qw0],
                             [qx1, qy1, qz1, qw1]])
    slerp = Slerp(key_times, key_rots)
    times = np.linspace(0, 1, num_steps)
    rots_interp = slerp(times).as_quat()

    poses_interp = []
    for i in range(num_steps):
        poses_interp.append([
            np.interp(i, [0, num_steps - 1], [t0, t1]),
            *trans_interp[i],
            *rots_interp[i]
        ])
    return poses_interp


def render_frame(pcd, extrinsic, intrinsic, width, height, renderer):
    """渲染单帧"""
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("pcd", pcd, o3d.visualization.rendering.MaterialRecord())

    renderer.setup_camera(intrinsic, extrinsic)
    img = renderer.render_to_image()
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser(description="Render 3DGS or point cloud along trajectory (offscreen)")
    parser.add_argument('--ply', type=str, required=True, help='Path to .ply file')
    parser.add_argument('--traj', type=str, required=True, help='Path to trajectory txt file')
    parser.add_argument('--out_dir', type=str, default='frames_out', help='Output frame folder')
    parser.add_argument('--video', type=str, default='traj.mp4', help='Output video filename')
    parser.add_argument('--width', type=int, default=848)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fx', type=float, required=True)
    parser.add_argument('--fy', type=float, required=True)
    parser.add_argument('--cx', type=float, required=True)
    parser.add_argument('--cy', type=float, required=True)
    parser.add_argument('--speed', type=int, default=10, help='Number of interpolation steps per pose')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading PLY...")
    pcd = o3d.io.read_point_cloud(args.ply)

    print("[INFO] Loading trajectory...")
    raw_poses = read_trajectory(args.traj)

    print("[INFO] Building renderer...")
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    renderer.scene.set_background([1, 1, 1, 1])  # white background

    # 相机内参矩阵
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(args.width, args.height, args.fx, args.fy, args.cx, args.cy)

    frame_idx = 0
    all_frames = []

    print("[INFO] Rendering trajectory...")
    for i in tqdm(range(len(raw_poses) - 1)):
        seg_poses = interp_poses(raw_poses[i], raw_poses[i + 1], args.speed)
        for pose in seg_poses:
            _, tx, ty, tz, qx, qy, qz, qw = pose
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            trans = np.array([tx, ty, tz])
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot
            extrinsic[:3, 3] = trans

            img = render_frame(pcd, extrinsic, intrinsic, args.width, args.height, renderer)
            frame_path = os.path.join(args.out_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            all_frames.append(frame_path)
            frame_idx += 1

    print(f"[INFO] Writing video to {args.video} ...")
    frame = cv2.imread(all_frames[0])
    h, w, _ = frame.shape
    out = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for f in tqdm(all_frames):
        out.write(cv2.imread(f))
    out.release()

    print("[DONE] Trajectory video rendered successfully.")


if __name__ == '__main__':
    main()
