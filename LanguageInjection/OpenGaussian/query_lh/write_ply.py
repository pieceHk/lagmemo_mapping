import os
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def write_colored_ply(xyz, region_ids, output_path):
    num_points = xyz.shape[0]
    region_ids = region_ids.astype(np.uint8)
    np.random.seed(42)
    
    # 为每个区域生成颜色（最多256个）
    unique_regions = np.unique(region_ids)
    color_map = {}
    for i, rid in enumerate(unique_regions):
        if rid == 0:
            color_map[rid] = (100, 100, 100)  # 无效区域灰色
        elif i < 5:
            color_map[rid] = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255)
            ][i]
        else:
            color_map[rid] = tuple(np.random.randint(0, 255, 3))

    # tqdm 包装点遍历
    vertices = []
    for i in tqdm(range(num_points), desc="Writing colored PLY"):
        x, y, z = xyz[i]
        rid = region_ids[i]
        r, g, b = color_map.get(rid, (200, 200, 200))
        vertices.append((x, y, z, r, g, b))

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    ply_data = PlyElement.describe(np.array(vertices, dtype=dtype), 'vertex')
    PlyData([ply_data], text=True).write(output_path)
    print(f"[INFO] Colored point cloud saved to {output_path}")

if __name__ == "__main__":
    # 路径配置（你可以按需改）
    model_path = "/home/sfs/OpenGaussian/output/HaiNing/sam_level_3/HaiNing[600-615)"
    ply_path = os.path.join(model_path, "point_cloud/iteration_90000/point_cloud.ply")
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    output_ply_path = os.path.join(model_path, "colored_point_cloud.ply")

    # 读取聚类信息和点云
    saved_data = np.load(mapping_file)
    leaf_ind = saved_data["leaf_ind.npy"]  # 每个点的聚类索引
    occu_count = saved_data["occu_count.npy"]
    valid_mask = occu_count[leaf_ind] >= 2  # 过滤有效叶子
    leaf_ind_valid = leaf_ind.copy()
    leaf_ind_valid[~valid_mask] = 0  # 将无效点标记为0区域

    ply_data = PlyData.read(ply_path)
    xyz = np.stack([ply_data['vertex'][axis] for axis in ['x', 'y', 'z']], axis=-1)
    opacity = np.array(ply_data['vertex']['opacity'])
    keep_mask = sigmoid(opacity) > 0.1

    # 过滤点
    xyz = xyz[keep_mask]
    region_ids = leaf_ind_valid[keep_mask]

    # 写入彩色ply
    write_colored_ply(xyz, region_ids, output_ply_path)
