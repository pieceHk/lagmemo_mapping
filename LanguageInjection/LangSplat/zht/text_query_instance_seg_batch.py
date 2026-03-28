import os
import argparse
import numpy as np
import torch
import cv2
import open_clip
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
# ================= 配置区域 =================

# 1. 定义你的目标类别 (支持多义词扩展)
TARGET_CLASSES = {
    'chair':           ['chair', 'seat', 'armchair', 'stool'],
    'table':           ['table', 'dining table', 'coffee table', 'desk'],
    'picture':         ['picture', 'painting', 'photo frame', 'poster'],
    'cabinet':         ['cabinet', 'cupboard', 'closet', 'wardrobe', 'drawer'],
    'sofa':            ['sofa', 'couch', 'settee'],
    'bed':             ['bed', 'mattress', 'bed frame'],
    'plant':           ['plant', 'potted plant', 'houseplant', 'flower pot'],
    'sink':            ['sink', 'washbasin'],
    'toilet':          ['toilet', 'wc'],
    'towel':           ['towel', 'hand towel', 'bath towel'],
    'tv monitor':      ['tv', 'television', 'monitor', 'screen'],
    'bathtub':         ['bathtub', 'bath'],
    'counter':         ['counter', 'countertop', 'kitchen island'],
    'oven_stove':      ['oven', 'stove', 'microwave'],
    'clothes':         ['clothes', 'clothing', 'jacket', 'shirt'],
    'refrigerator':    ['refrigerator', 'fridge'],
    'washing_machine': ['washing machine', 'laundry machine']
}
CLASS_COLORS = {
    'chair':           (0, 76, 255),    # 红色系 (鲜艳)
    'table':           (0, 180, 0),     # 绿色系 (深绿)
    'picture':         (255, 105, 180), # 蓝色系 (热粉/紫调，用于区分) -> OpenCV中是 BGR，这是(180,105,255)
    'cabinet':         (0, 140, 255),   # 橙色系
    'sofa':            (180, 0, 180),   # 紫色系
    'bed':             (255, 191, 0),   # 深天蓝 (视觉上的青蓝色)
    'plant':           (34, 139, 34),   # 森林绿
    'sink':            (230, 216, 173), # 浅蓝/白瓷色
    'toilet':          (255, 255, 224), # 浅黄/白
    'towel':           (147, 20, 255),  # 深粉色
    'tv monitor':      (50, 50, 50),    # 深灰/黑
    'bathtub':         (238, 130, 238), # 紫罗兰
    'counter':         (19, 69, 139),   # 棕色/木色
    'oven_stove':      (128, 128, 128), # 灰色
    'clothes':         (255, 144, 30),  # 躲避蓝 (一种深蓝)
    'refrigerator':    (200, 200, 200), # 银灰色
    'washing_machine': (210, 245, 255)  # 象牙白/米色
}

# 2. 定义负样本 (非常重要！防止背景被强行分类为物体)
NEGATIVE_CLASSES = [
    'wall', 'floor', 'ceiling', 'window', 'door', 'curtain', 
    'light', 'lamp', 'rug', 'carpet', 'unknown object', 'background'
]

# 3. CLIP 模型配置 (需与 preprocess.py 一致)
CLIP_MODEL = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"

class OpenCLIPClassifier:
    def __init__(self, target_classes, negative_classes, device='cuda'):
        self.device = device
        print(f"Loading CLIP model: {CLIP_MODEL}...")
        self.model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        
        # 准备文本特征
        self.class_map = []      # 记录索引到最终类名(key)的映射
        self.text_features = self._encode_text(target_classes, negative_classes)

    def _encode_text(self, targets, negatives):
        """将类别描述转换为CLIP特征向量"""
        prompts = []
        
        # 处理目标类别
        for class_key, synonyms in targets.items():
            for w in synonyms:
                prompts.append(f"a photo of a {w}") # 使用 Prompt 模板增强
                self.class_map.append(class_key)    # 记录归属的类别
        
        # 处理负样本
        for w in negatives:
            prompts.append(f"a photo of {w}")
            self.class_map.append("BACKGROUND")     # 标记为背景

        print(f"Encoded {len(prompts)} text prompts.")
        
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def classify(self, image_features):
        """
        输入: image_features (N, 512)
        输出: (N,) best_class_names, (N,) scores
        """
        image_features = torch.tensor(image_features).to(self.device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度: (N_img, 512) @ (N_text, 512).T -> (N_img, N_text)
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        # 获取每个实例得分最高的类别
        values, indices = similarity.topk(1, dim=-1)
        
        results = []
        scores = []
        
        for i in range(len(indices)):
            idx = indices[i].item()
            score = values[i].item()
            class_name = self.class_map[idx]
            results.append(class_name)
            scores.append(score)
            
        return results, scores

def visualize_segmentation(image_path, mask_idx_map, labels, scores, output_path):
    """
    image_path: 原图路径
    mask_idx_map: (H, W) 每个像素的值对应 features 里的行索引
    labels: list, 长度对应 features 的行数
    scores: list, 长度对应 features 的行数
    """
    img = cv2.imread(image_path)
    img_vis = img.copy()

    # 获取所有出现的实例索引 (排除 -1 背景)
    unique_indices = np.unique(mask_idx_map)
    unique_indices = unique_indices[unique_indices != -1]
    
    vis_count = 0
    
    for idx in unique_indices:
        idx = int(idx)
        if idx >= len(labels): continue # 越界保护
        
        label = labels[idx]
        score = scores[idx]
        
        # 1. 过滤背景和低置信度
        if label == "BACKGROUND":
            continue
        if score < 0.4:  # 阈值可调
            continue
        print(f"Visualizing idx {idx}: {label} ({score:.2f})")

        # 2. 生成 Mask 叠加层
        mask = (mask_idx_map == idx).astype(np.uint8)
        
        # 找到 Mask 的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = CLASS_COLORS.get(label, (128, 128, 128)) # 默认灰色
        
        # 绘制半透明填充
        colored_mask = np.zeros_like(img)
        colored_mask[mask == 1] = color
        
        # 叠加混合
        alpha = 0.5
        mask_bool = mask == 1
        img_vis[mask_bool] = cv2.addWeighted(img[mask_bool], 1-alpha, colored_mask[mask_bool], alpha, 0)
        
        # 绘制轮廓
        cv2.drawContours(img_vis, contours, -1, color, 2)
        
        # 3. 绘制标签文字
        # 找 面积最大Mask 的中心点
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            text = f"{label}: {score:.2f}"
            cv2.putText(img_vis, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_vis, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # 描边
        
        vis_count += 1

    print(f"Visualized {vis_count} objects.")
    cv2.imwrite(output_path, img_vis)
    print(f"Saved result to {output_path}")



def process_batch(img_dir, feature_dir, output_dir, scale_idx):
    """
    批量处理逻辑
    img_dir: 原图所在文件夹
    feature_dir: npy文件所在文件夹 (包含 _f.npy 和 _s.npy)
    output_dir: 结果保存文件夹
    scale_idx: 选择的尺度 (0-3)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 扫描所有特征文件 (*_f.npy)
    # 假设文件名格式为 "xxxx_f.npy"
    feature_files = glob.glob(os.path.join(feature_dir, "*_f.npy"))
    feature_files.sort() # 排序保证顺序

    if len(feature_files) == 0:
        print(f"Error: No *_f.npy files found in {feature_dir}")
        return

    print(f"Found {len(feature_files)} files to process.")

    # 2. 初始化分类器 (只初始化一次，避免重复加载模型)
    classifier = OpenCLIPClassifier(TARGET_CLASSES, NEGATIVE_CLASSES)

    # 3. 循环处理
    for f_path in tqdm(feature_files, desc="Processing Batch"):
        # 解析文件名
        # f_path: /path/to/00001_f.npy
        base_name = os.path.basename(f_path).replace("_f.npy", "") # -> 00001
        
        # 构造对应的 mask 路径 (_s.npy)
        mask_path = os.path.join(feature_dir, f"{base_name}_s.npy")
        
        # 构造对应的原图路径 (尝试常见后缀)
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        # 检查文件完整性
        if not os.path.exists(mask_path):
            print(f"[Warn] Mask not found for {base_name}, skipping.")
            continue
        if img_path is None:
            print(f"[Warn] Image not found for {base_name}, skipping.")
            continue

        try:
            # A. 加载数据
            features = np.load(f_path)           # (N, 512)
            masks_all_scales = np.load(mask_path) # (4, H, W)
            current_mask_map = masks_all_scales[scale_idx]

            # B. 推理
            pred_labels, pred_scores = classifier.classify(features)

            # C. 可视化
            save_name = f"{base_name}_scale{scale_idx}_seg.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # 复用之前的 visualize 函数
            visualize_segmentation(
                img_path, 
                current_mask_map, 
                pred_labels, 
                pred_scores, 
                save_path
            )
            
        except Exception as e:
            print(f"[Error] Failed processing {base_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory containing _f.npy and _s.npy")
    parser.add_argument("--output_dir", type=str, default="vis_results", help="Output directory")
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2, 3], 
                        help="0:Default, 1:Small, 2:Medium, 3:Large")
    args = parser.parse_args()

    process_batch(args.img_dir, args.feature_dir, args.output_dir, args.scale)

if __name__ == "__main__":
    main()