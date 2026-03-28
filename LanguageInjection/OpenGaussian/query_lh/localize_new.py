import os
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import numpy as np
import torch
import json
from openclip import OpenCLIPNetwork
import torchvision
import matplotlib.pyplot as plt

from position_trans import get_first_frame, transform_position
import colorsys
import sys

# 保存print输出到文件和终端
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'a')
        self.stdout = sys.stdout
        sys.stdout = self  # 全局重定向

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):  # 确保实时刷新
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):  # 支持上下文管理器（with语法）
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # 退出时恢复stdout
        sys.stdout = self.stdout
        self.file.close()


def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  

def write_ply(vertex_data, output_path, region_id, colors):
    vertices = []
    color_map = {i+1: color for i, color in enumerate(colors)}  # map id to color
    
    for i, vertex in enumerate(vertex_data):
        rid = region_id[i]
        if rid > 0:
            # region id to color
            r, g, b = color_map[rid]
            new_vertex = (vertex['x'], vertex['y'], vertex['z'], r, g, b)
            vertices.append(new_vertex)
        else:
            if i % 5 == 0:
                new_vertex = (vertex['x'], vertex['y'], vertex['z'], 128, 128, 128)
                vertices.append(new_vertex)
    
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_array = np.array(vertices, dtype=vertex_dtype)
    
    PlyData([PlyElement.describe(vertex_array, 'vertex')], text=True).write(output_path)


def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Extract the coordinates and labels of the points. The labels are from 1 to 40 for the NYU40 dataset, with 0 being invalid.
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_ply", action="store_true")
    args = parser.parse_args()
    # gt_pose_path = "/home/sfs/4_new_lagmemo_scenes/record_data_for4rooms_v2/BAb/0/local_pos.txt"
    gt_pose_path = "/home/sfs/lagmemo_scenes/tee/0/local_pos.txt"
    # gt_pose_path = "/home/sfs/OpenGaussian/5cd/local_pos.txt"
    # gt_pose_path = "/home/sfs/SplaTAM/gs_data/0/local_pos.txt"
    model_name = "tee_32_5_30_60"
    # 选择top k
    k = 5
    output_path = f"/home/sfs/OpenGaussian/query_lh/experiment_new3/output/sam_level_3/tabs_and_figs_{model_name.split('/')[-1]}/episode3"
    os.makedirs(output_path, exist_ok=True)
    output_txt = os.path.join(output_path, "output.txt")
    model_path = f"/home/sfs/OpenGaussian/output/4scenes/sam_level_3/{model_name}"
    # "/home/zht/github_play/OpenGaussian/sfs_output/0504data" # "/home/zht/github_play/OpenGaussian/output/0504data_32_3_20_40_seem_delete500"
    goal_dir="/home/sfs/OpenGaussian/query_lh/3_episode_data/TEEsavR23oF/2"
    # goal_dir ="/home/sfs/OpenGaussian/query_lh/groundtruth_data_new3/new_data3/TEEsavR23oF"
    # goal_dir = "/home/sfs/OpenGaussian/query_lh/groundtruth_data_new2/groundtruth_data_tee"
    # goal_dir = "/home/sfs/OpenGaussian/query/groundtruth_data"
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)
    leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=k1*k2, 512] 
    leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=k1*k2] 
    leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=k1*k2] 
    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] 
    leaf_lang_feat[leaf_occu_count < 2] *= 0.0
    # 待确认
    leaf_ind = leaf_ind.clamp(max=319)  # 64*5=320
    # codebook feature normalize
    leaf_lang_feat = F.normalize(leaf_lang_feat, dim=-1)
    
    ply_path = os.path.join(model_path, "point_cloud/iteration_90000/point_cloud.ply")
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    
    opacity = sigmoid(vertex_data["opacity"])
    opacity_mask = opacity >= 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    open_clip_network = OpenCLIPNetwork(device)

    _, t_0, R_0_inv = get_first_frame(gt_pose_path)
    
    task_dirs = os.listdir(goal_dir)
    # 按照字典序排序
    task_dirs.sort()

    # 统计top5最短距离和对应的排名、相似度、属于codebook的索引、模态
    all_distances = []
    top_ranks = []
    all_similarities = []
    all_indexs = []
    all_modes = []
    all_numbers = []  # 用于存储任务编号（如00, 01, ...）
    all_categories = []
    # 上下文管理器控制print输出到文件和终端
    with Tee(output_txt) as t:
        for task_dir in task_dirs: # task_dir "00display cabinet"
            all_numbers.append(task_dir[:2])  # 提取任务编号（如00, 01, ...）
            all_categories.append(task_dir[2:])  # 提取类别名
            # 如果文件路径中存在language.txt，输出此为language goal任务
            if "language.txt" in os.listdir(os.path.join(goal_dir, task_dir)):
                language_txt = os.path.join(goal_dir, task_dir, "language.txt")
                language = open(language_txt, "r").read().strip()
                word_count = len(language.split())
                if word_count <= 3:
                    flag = "object"
                    print(f"{task_dir} is an object goal task")
                else:
                    flag = "text"
                    print(f"{task_dir} is a language goal task")
                print("the content is: ", language)
                feature = open_clip_network.encode_text([language]).float()
                feature = F.normalize(feature, dim=-1)
                similarity = torch.matmul(leaf_lang_feat, feature.T).squeeze(1)
            else:
                flag = "image"
                print(f"{task_dir} is a image goal task")
                # img_path为此文件下以0开头，png结尾的图像（已确保唯一）
                img_in_task = [f for f in os.listdir(os.path.join(goal_dir, task_dir)) if (f.startswith("0") or f.startswith("1")) and f.endswith("0.png")][0]
                img_path = os.path.join(goal_dir, task_dir, img_in_task)
                img = torchvision.io.read_image(img_path).float() / 255.0
                print("image shape: ",img.shape)
                img = torchvision.io.read_image(img_path).float() / 255.0
                feature = open_clip_network.encode_image(img).float()
                feature = F.normalize(feature, dim=-1)
                similarity = torch.matmul(leaf_lang_feat, feature.T).squeeze(1)  # [num_leaf=k1*k2]

            all_modes.append(flag)
            task_id = task_dir[:2]
            # sort the similarity scores
            sorted_indices = torch.argsort(similarity, descending=True)
            
            top_indices = sorted_indices[:k]
            for i in range(k):
                print(f"Top {i+1} index: {top_indices[i].item()}")
                print(f"    similarity: {similarity[top_indices[i]].item()}")
                print(f"    occu_count: {leaf_occu_count[top_indices[i]].item()}")
        
            point_leaf_ind = leaf_ind.cpu().numpy()
            region_id = np.zeros_like(point_leaf_ind, dtype=int)
        
            for region_idx, leaf_idx in enumerate(top_indices.cpu().numpy(), 1):
                mask = (point_leaf_ind == leaf_idx)
                region_id[mask] = region_idx

            region_id[~opacity_mask] = 0

            # calculate centroids for each region
            centroids = []
            for rid in range(1, k+1):
                mask = (region_id == rid)
                if np.any(mask):
                    centroid = np.mean(points[mask], axis=0)
                    centroid = transform_position(centroid, t_0, R_0_inv)
                    centroids.append(centroid)
                    print(f"Region {rid} centroid: {centroid}")
                else:
                    centroids.append(None)
                    print(f"Region {rid} has no points")
            
            print("-" * 20)
        
            region_colors = [
                (255, 0, 0),    # red
                (0, 255, 0),    # green
                (0, 0, 255),    # blue
                (255, 255, 0),  # yellow
                (255, 0, 255)   # purple
            ][:k]
            if args.output_ply:
                output_name = f"{task_id}_{flag}_top{k}.ply"
                ply_output_path = f"/home/sfs/OpenGaussian/query_lh/experiment_wxl9.19/localize_vis/sam_level_3/{model_name.split('/')[-1]}"
                os.makedirs(ply_output_path, exist_ok=True)
                write_ply(vertex_data, os.path.join(ply_output_path, output_name), region_id, region_colors)
            
            pos_path = os.path.join(goal_dir, task_dir, "pos.txt")
            try:
                with open(pos_path, 'r') as f:
                    content = f.read().strip()
                    # 如果有多行，则每行是一个坐标；否则就是一个坐标
                    lines = content.split('\n')
                    gt_positions = []
                    for line in lines:
                        if line.strip():  # 排除空行
                            gt_pos = np.array(eval(line.strip()))
                            gt_pos = np.array([gt_pos[0], gt_pos[2]])
                            gt_positions.append(gt_pos)
                
                distance = np.inf
                for i, pred_centroid in enumerate(centroids):
                    pred_centroid = np.array([pred_centroid[0], pred_centroid[2]])
                    # 计算与每个真值点的最短距离
                    min_distance = np.min([
                        np.linalg.norm(pred_centroid - gt_pos)
                        for gt_pos in gt_positions
                    ])
                    print(f"region {i+1} distance: {min_distance}")
                    if min_distance < distance:
                        distance = min_distance
                        best_index = i
                print(f"Top {best_index + 1} is best")
                # 记录数据
                all_distances.append(distance)
                all_similarities.append(similarity[top_indices[best_index]].item())
                top_ranks.append(best_index)
                all_indexs.append(top_indices[best_index].item())
                
                # valid_samples += 1
                
            except Exception as e:
                print(f"Error processing {pos_path}: {str(e)}")
                continue
        
        
        # 统计结果
        valid_distances = np.array(all_distances)
        # 小于1.5m视为成功，输出总成功率，模态为image的成功率、language的成功率
        success_rate = np.sum(valid_distances < 1.5) / len(valid_distances)
        print(f"Success rate: {success_rate:.2%}")
        language_mask = np.array(all_modes) == "text"
        image_mask = np.array(all_modes) == "image"
        object_mask = np.array(all_modes) == "object"
        language_success_rate = np.sum(valid_distances[language_mask] < 1.5) / np.sum(language_mask)
        image_success_rate = np.sum(valid_distances[image_mask] < 1.5) / np.sum(image_mask)
        object_success_rate = np.sum(valid_distances[object_mask] < 1.5) / np.sum(object_mask)
        print(f"Language success rate: {language_success_rate:.2%}")
        print(f"Image success rate: {image_success_rate:.2%}")
        print(f"Object success rate: {object_success_rate:.2%}")

        stats = {
            "success_rate": success_rate,
            "language_success_rate": language_success_rate,
            "image_success_rate": image_success_rate,
            "object_success_rate": object_success_rate,
            # "average_distance": np.nanmean(valid_distances),
            # "median_distance": np.nanmedian(valid_distances),
            # "min_distance": np.nanmin(valid_distances),
            # "max_distance": np.nanmax(valid_distances),
            # "std_distance": np.nanstd(valid_distances),
            # "valid_samples": valid_samples,
            # "total_samples": len(img_paths)
        }
        # 保存统计结果
        with open(os.path.join(output_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        def add_labels(x_list, y_list, ax, offset=0.05):
            """为每个数据点添加序号标签"""
            for i, (x, y) in enumerate(zip(x_list, y_list)):
                ax.text(x + offset, y, str(i), 
                        fontsize=8, color='darkred', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 修改相似度-距离图的绘制部分
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        scatter = ax.scatter(all_similarities, all_distances, alpha=0.6)
        add_labels(all_similarities, all_distances, ax, offset=0.001)
        plt.xlabel("CLIP Similarity Score")
        plt.ylabel("Distance to GT (m)")
        plt.title("Feature Similarity vs Localization Accuracy (Number indicates image index)")
        plt.savefig(os.path.join(output_path, "similarity_vs_distance_labeled.png"))
        plt.close()
        
        # 在数据统计之后新增以下代码
        plt.figure(figsize=(16, 6))
        ax = plt.gca()

        # 生成颜色映射（根据误差值渐变）
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_distances)))

        # 绘制柱状图（带颜色渐变）
        bars = ax.bar(range(len(all_distances)), all_distances, 
                    color=colors, 
                    edgecolor='black', 
                    linewidth=0.5,
                    width=0.8)

        # 添加平均线
        mean_val = np.nanmean(all_distances)
        ax.axhline(mean_val, color='red', linestyle='--', 
                linewidth=1, label=f'Average ({mean_val:.2f}m)')
        # 添加1.5m线
        ax.axhline(1.5, color='blue', linestyle='--', 
                linewidth=2, label='1.5m Threshold')

        # 优化坐标轴显示
        ax.set_xticks(range(len(all_distances)))
        ax.set_xticklabels([f"{i}\n{mode}" for i, mode in zip(range(len(all_distances)), all_modes)], 
                    rotation=45, fontsize=8, ha='center')
        ax.set_xlabel("Goal Index", fontsize=12)
        ax.set_ylabel("Localization Error (m)", fontsize=12)
        plt.title("Per-Goal Localization Performance", fontsize=14, pad=20)

        # 双标签系统：顶部显示误差值，底部显示预测来源
        for idx, (bar, best_idx) in enumerate(zip(bars, top_ranks)):
            height = bar.get_height()
            x_center = bar.get_x() + bar.get_width()/2
            
            # 顶部标签：红色显示误差值
            ax.text(x_center, height + 0.05, 
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=9, color='darkred')
            
            # 底部标签：白色显示预测来源
            ax.text(x_center, 0.05,  # 放置在柱子底部上方
                f'top{best_idx+1}',
                ha='center', va='bottom',
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='black', 
                            alpha=0.7))

        # 添加图例和颜色条
        plt.legend()
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                norm=plt.Normalize(vmin=min(all_distances), 
                                                vmax=max(all_distances)))
        plt.colorbar(sm, ax=ax, label='Error Magnitude')

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "per_task_errors.png"), dpi=150)
        plt.close()

        # 控制台输出统计结果
        print("\n===== 统计结果 =====")
        # print(f"有效样本数: {stats['valid_samples']}/{stats['total_samples']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"语言任务成功率: {stats['language_success_rate']:.2%}")
        print(f"图像任务成功率: {stats['image_success_rate']:.2%}")
        print(f"物体任务成功率: {object_success_rate:.2%}")

        
        import csv

        csv_path = os.path.join(output_path, "results_table.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "误差", "top", "任务类型", "物体类别"])
            
            for i in range(len(all_distances)):
                row = [
                    all_numbers[i],  # 从路径提取的原始序号（如00）
                    f"{all_distances[i]:.4f}",  # 保留4位小数
                    top_ranks[i] + 1,  # 从0-based转1-based
                    all_modes[i],  # 任务类型
                    all_categories[i]
                ]
                writer.writerow(row)

        
    