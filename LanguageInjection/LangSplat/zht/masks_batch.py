import os
import numpy as np
import argparse
import glob
import tqdm
from matplotlib.colors import ListedColormap
import cv2
import random

def vis_segmentation(image_path, mask_idx_map, output_path):
    mask_num = int(np.max(mask_idx_map))
    # Generate random distinct colors (avoid close hues)
    def generate_distinguishable_colors(mask_num):
        # 在 0-179 之间等距离取点，保证色调尽量分开
        hues = np.linspace(0, 178, mask_num, dtype=np.uint8)
        
        hsv_colors = np.zeros((mask_num, 1, 3), dtype=np.uint8)
        for i in range(mask_num):
            hsv_colors[i, 0, 0] = hues[i]
            hsv_colors[i, 0, 1] = random.randint(100, 255) # 饱和度较高但随机
            hsv_colors[i, 0, 2] = random.randint(100, 255) # 亮度较高但随机
        
        bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)
        return [tuple(map(int, color[0])) for color in bgr_colors]
    colors = generate_distinguishable_colors(mask_num)
    colors = np.vstack([(0, 0, 0), colors])  # Add black for -1
    # print(colors.shape)
    img = cv2.imread(image_path)
    # 绘制半透明的mask
    img_vis = img.copy()
    unique_indices = np.unique(mask_idx_map)
    for idx in unique_indices:
        idx = int(idx)
        mask = (mask_idx_map == idx).astype(np.uint8)
        # print("mask area:", np.sum(mask))
        # 找到 Mask 的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = [int(c) for c in colors[idx]]  # BGR
        if idx == -1:
            color = (0, 0, 0)
        # 绘制半透明填充
        colored_mask = np.zeros_like(img)
        colored_mask[mask == 1] = color
        # 叠加混合
        alpha = 0.9
        mask_bool = mask == 1
        img_vis[mask_bool] = cv2.addWeighted(img[mask_bool], 1-alpha, colored_mask[mask_bool], alpha, 0)
        # 绘制轮廓
        cv2.drawContours(img_vis, contours, -1, color, 2)
    cv2.imwrite(output_path, img_vis)


def vis_batch(img_dir, feature_dir, output_dir, scale_idx=0):
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
    segmentation_files = glob.glob(os.path.join(feature_dir, "*_s.npy"))
    segmentation_files.sort() # 排序保证顺序

    if len(segmentation_files) == 0:
        print(f"Error: No *_f.npy files found in {feature_dir}")
        return

    print(f"Found {len(segmentation_files)} files to process.")

    # 2. 循环处理
    for s_path in tqdm.tqdm(segmentation_files, desc="Processing Batch"):
        # 解析文件名
        base_name = os.path.basename(s_path).replace("_s.npy", "") # -> img0001
        segmentation = np.load(s_path)
        current_mask_map = segmentation[scale_idx]
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        save_name = f"{base_name}_scale{scale_idx}_seg.jpg"
        save_path = os.path.join(output_dir, save_name)
        vis_segmentation(img_path, current_mask_map, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory Path of original images")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory Path of features and segmentations of images")
    parser.add_argument("--output_dir", type=str, default="./vis_masks/", help="Output images path")
    args = parser.parse_args()

    vis_batch(args.img_dir, args.feature_dir, args.output_dir)

if __name__ == "__main__":
    main()