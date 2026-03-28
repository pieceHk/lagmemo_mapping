import numpy as np

# ===== 修改此处为你的 npz 文件路径 =====
npz_path = "/home/sfs/SplaTAM/HaiNing[450-465)/0/params.npz"
output_txt = "/home/sfs/SplaTAM/HaiNing[450-465)/0/camera_poses.txt"

# 读取 npz 文件
data = np.load(npz_path)

# 提取相机平移与旋转
cam_trans = data["cam_trans"]        # shape: (1, 3, N)
cam_rots = data["cam_unnorm_rots"]   # shape: (1, 4, N)
N = cam_trans.shape[2]

# 尝试获取时间戳
if "timestep" in data and len(data["timestep"]) >= N:
    timestamps = data["timestep"][:N]
else:
    # 若无有效时间戳，则按帧索引代替
    timestamps = np.arange(N, dtype=np.float32)

# 重排形状方便拼接
cam_trans = cam_trans[0].T  # (N, 3)
cam_rots = cam_rots[0].T    # (N, 4)

# 合并为 [timestamp, x, y, z, qx, qy, qz, qw]
poses = np.concatenate([timestamps[:, None], cam_trans, cam_rots], axis=1)

# 保存为 txt 文件
np.savetxt(output_txt, poses, fmt="%.6f", delimiter=" ")

print(f"✅ 已生成相机位姿文件: {output_txt}")
print(f"共导出 {poses.shape[0]} 帧，每行8个数值")
