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

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  

def write_ply(vertex_data, output_path, region_id, colors, id):
    vertices = []
    color_map = {i+1: color for i, color in enumerate(colors)}  # map id to color
    # color_map2 有更多颜色，随机生成
    color_map2 = {i+1: (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for i in range(40)}
    for i, vertex in enumerate(vertex_data):
        rid = region_id[i]
        if rid == id:
            # region id to color
            # if rid > 5:
            #     r, g, b = color_map2[rid]
            # else if rid == id:
            #     r, g, b = color_map[1]
            
            new_vertex = (vertex['x'], vertex['y'], vertex['z'], 255, 0, 0)
            vertices.append(new_vertex)
        else:
            if i % 5 == 0:
                new_vertex = (vertex['x'], vertex['y'], vertex['z'], 128, 128, 128)
                vertices.append(new_vertex)
    # for i, vertex in enumerate(vertex_data):
    #     rid = region_id[i]
    #     if rid > 0:
    #         # region id to color
    #         if rid > 5:
    #             r, g, b = color_map2[rid]
    #         else:
    #             r, g, b = color_map[rid]
            
    #         new_vertex = (vertex['x'], vertex['y'], vertex['z'], r, g, b)
    #         vertices.append(new_vertex)
    #     else:
    #         if i % 100 == 0:
    #             new_vertex = (vertex['x'], vertex['y'], vertex['z'], 200, 200, 200)
    #             vertices.append(new_vertex)
    
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
    task_id = "02"
    gt_path = "/home/sfs/OpenGaussian/query_lh/3_episode_data/TEEsavR23oF/2"
    # get top k indices
    k = 5
    id = 5  
    for task_path in os.listdir(gt_path):
        if task_path.startswith(task_id):
            gt_goal_path = os.path.join(gt_path, task_path)
            with open(os.path.join(gt_goal_path, "pos.txt"), "r") as f:
                line = f.readline().strip()
                query_gt_pos = np.fromstring(line[1:-1], sep=',')
            if "language.txt" in os.listdir(gt_goal_path):
                mode = "txt"
                with open(os.path.join(gt_goal_path, "language.txt"), "r") as f:
                    text = f.readline().strip()
            else:
                mode = "img"
                # img_path是gt_goal_path下以task_id开头png结尾的文件
                img_path = [os.path.join(gt_goal_path, f) for f in os.listdir(gt_goal_path) if f.endswith("0.png") and f.startswith(task_id)][0]
                img = torchvision.io.read_image(img_path).float() / 255.0
            break
    gt_pose_path = "/home/sfs/lagmemo_scenes/tee/0/local_pos.txt"
    # gt_pose_path = "/home/sfs/OpenGaussian/5cd/local_pos.txt"
    # gt_pose_path = "/home/sfs/SplaTAM/gs_data/0/local_pos.txt"
    _, t_0, R_0_inv = get_first_frame(gt_pose_path)
    model_paths = ["/home/sfs/OpenGaussian/output/4scenes/sam_level_3/tee_32_5_30_60"]
    for model_path in model_paths:
    
        ply_output_path = "./single_task/tee/02"
        if not os.path.exists(ply_output_path):
            os.makedirs(ply_output_path)

    
        mapping_file = os.path.join(model_path, "cluster_lang.npz")
        saved_data = np.load(mapping_file)
        leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=k1*k2, 512] 
        leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=k1*k2] 
        leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=k1*k2] 
        leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] 
        leaf_lang_feat[leaf_occu_count < 2] *= 0.0
        leaf_ind = leaf_ind.clamp(max=319)  # 64*5=320
        
        ply_path = os.path.join(model_path, "point_cloud/iteration_90000/point_cloud.ply")
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data
        points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        
        opacity = sigmoid(vertex_data["opacity"])
        opacity_mask = opacity >= 0.1
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        open_clip_network = OpenCLIPNetwork(device)
        
        if mode == "img":
            feature = open_clip_network.encode_image(img).float()
        else:
            feature = open_clip_network.encode_text(text).float()
        
        all_distances = []
        all_similarities = []
        all_occu_counts = []
        best_indexs = []
        valid_samples = 0
        
        all_categories = []
        all_numbers = []
        
        # print(feature.shape)  # Should print the shape of the encoded text features
        # print(feature.device)  # Print the device of the text features
            
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=-1)
        feature = F.normalize(feature, dim=-1)
        similarity = torch.matmul(leaf_lang_feat, feature.T).squeeze(1)  # [num_leaf=k1*k2]
        # print(similarity)  # Should print the shape of the similarity scores
        # print(similarity.max())  # Print the maximum similarity score
            
        # sort the similarity scores
        sorted_indices = torch.argsort(similarity, descending=True)
        top_indices = sorted_indices[:k]
        
        # print(top_indices)  # Print the top k indices
            
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

        output_name = f"{task_id}_{mode}_top{id}.ply"
        write_ply(vertex_data, os.path.join(ply_output_path, output_name), region_id, region_colors, id)
            
        # 计算误差只考虑x和z坐标
        gt_pos = np.array([query_gt_pos[0], query_gt_pos[2]])
                
        # 计算预测中心与真值的距离
        distance = np.inf
        for i, pred_centroid in enumerate(centroids):
            # 计算误差只考虑x和z坐标
            pred_centroid = np.array([pred_centroid[0], pred_centroid[2]])
            new_distance = np.linalg.norm(pred_centroid - gt_pos)
            
            print(f"region {i+1} distance: {new_distance}")
            
            if new_distance < distance:
                distance = new_distance
                best_index = i
                
        print(f"Top {best_index + 1} is best")
            
        


        
    