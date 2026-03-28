import numpy as np
from scipy.spatial.transform import Rotation as R

# ===== 修改路径 =====
npz_path = "/home/sfs/SplaTAM/HaiNing[450-465)/0/w2c.npz"
output_txt = "/home/sfs/SplaTAM/HaiNing[450-465)/0/camera_poses.txt"

# 读取 npz
data = np.load(npz_path)
w2c_all = data["gt_w2c_all_frames"]   # shape: (N, 4, 4)
N = w2c_all.shape[0]

poses = []

for i in range(N):
    w2c = w2c_all[i]
    
    # === 反转为 c2w ===
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    # === 转四元数 ===
    quat = R.from_matrix(R_c2w).as_quat()  # [x, y, z, w]

    # === 拼接 [timestamp, x, y, z, qx, qy, qz, qw] ===
    timestamp = float(i)
    pose = [timestamp, *t_c2w.tolist(), *quat.tolist()]
    poses.append(pose)

poses = np.array(poses)

# 保存
np.savetxt(output_txt, poses, fmt="%.6f", delimiter=" ")
print(f"✅ 已生成相机位姿文件: {output_txt}")
print(f"共导出 {N} 帧，每行 8 个数值。")
