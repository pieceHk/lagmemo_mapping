# ==============================================================================
# 【使用说明】
# 1. 请在 LangSplat 项目根目录下（与 preprocess.py 同级）确保有 ckpts/ 文件夹。
# 2. 将你在能上网的电脑上下载好的 open_clip_pytorch_model.bin 文件，上传到服务器的 ckpts/ 目录中。
# 3. 复制以下全部代码，直接覆盖替换你原有的 preprocess.py 文件。
# 4. 重新运行 process.sh 即可离线加载 CLIP 权重，不再需要连接 Hugging Face 网络。
# ==============================================================================

import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import gc 
from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torchvision
from torch import nn

import open_clip

# 降低多线程与 OpenCL 引发的底层不稳定概率
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        # ==================================================
        # 【修改核心部分】：将 pretrained 改为读取本地文件路径
        # 你也可以根据实际情况将其改为绝对路径，例如: "/home/htz/works/3DGS/LangSplat/ckpts/open_clip_pytorch_model.bin"
        # ==================================================
        local_clip_path = "ckpts/open_clip_pytorch_model.bin"
        
        if not os.path.exists(local_clip_path):
            print(f"\n[ ERROR ] 找不到本地权重文件: {local_clip_path}")
            print("请确认你已经将下载好的 open_clip_pytorch_model.bin 放置在对应路径！\n")

        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # 保持 ViT-B-16 不变
            pretrained=local_clip_path,   # 原本是 self.config.clip_model_pretrained，现强制指定本地离线路径
            precision="fp16",
        )
        # ==================================================

        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
            ).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(
            self.config.clip_model_type, self.config.clip_model_pretrained
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def create(img_folder, data_list, save_folder, args):
    """
    流式处理架构：单张读取 -> 提取特征 -> 直接保存 -> 内存回收
    彻底消除大 Tensor 拼接导致的 OOM 问题
    """
    mask_generator.predictor.model.to("cuda")
    WARNED = False

    for data_path in tqdm(data_list, desc="Processing & Embedding images"):
        save_path = os.path.join(save_folder, data_path.split(".")[0])
        save_path_s = save_path + "_s.npy"
        save_path_f = save_path + "_f.npy"
        if os.path.exists(save_path_s) and os.path.exists(save_path_f):
            continue

        image_path = os.path.join(img_folder, data_path)
        
        # ==========================================
        # 1. 动态读取与调整单张图像 (避免内存囤积)
        # ==========================================
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("\n[ INFO ] Encountered large input images (>1080P), rescaling to 1080P.")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        # 转为 Tensor 并增加 Batch 维度: (1, 3, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # ==========================================
        # 2. 提取 SAM Mask 与 CLIP 特征
        # ==========================================
        try:
            img_embed_dict, seg_map_dict = _embed_clip_sam_tiles(img_tensor, sam_encoder)
        except Exception as e:
            print(f"\n[ ERROR ] Failed to process {data_path}. Error: {e}")
            del image, img_tensor
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ==========================================
        # 3. 整理特征与分割图 (无需繁琐的 Padding)
        # ==========================================
        lengths = [len(v) for k, v in img_embed_dict.items()]
        total_length = sum(lengths)
        if total_length == 0:
            print(f"\n[ WARN ] No valid masks for {data_path}, skipping.")
            del image, img_tensor, img_embed_dict, seg_map_dict
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # 拼接 CLIP 特征
        img_embed = torch.cat([v for k, v in img_embed_dict.items()], dim=0)
        assert img_embed.shape[0] == total_length

        # 整理 Seg Maps 并累加索引
        seg_map_tensor_list = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j - 1]

        for j, (k, v) in enumerate(seg_map_dict.items()):
            if j == 0:
                seg_map_tensor_list.append(torch.from_numpy(v))
                continue
            assert v.max() <= lengths[j] - 1, f"Mask index out of bounds: {j}"
            v[v != -1] += lengths_cumsum[j - 1]
            seg_map_tensor_list.append(torch.from_numpy(v))

        if len(seg_map_tensor_list) == 0:
            print(f"\n[ WARN ] Empty seg map for {data_path}, skipping.")
            del image, img_tensor, img_embed_dict, seg_map_dict, img_embed, seg_map_tensor_list
            torch.cuda.empty_cache()
            gc.collect()
            continue
            
        seg_map = torch.stack(seg_map_tensor_list, dim=0)

        # ==========================================
        # 4. 立即落盘保存
        # ==========================================
        curr = {
            "feature": img_embed, # 纯粹的有效特征，无填充
            "seg_maps": seg_map
        }
        sava_numpy(save_path, curr)

        # ==========================================
        # 5. 强制垃圾回收，释放显存与内存
        # ==========================================
        del image, img_tensor, img_embed_dict, seg_map_dict, img_embed, seg_map_tensor_list, seg_map, curr
        torch.cuda.empty_cache()
        gc.collect()

    # 处理完毕，将模型切回 CPU
    mask_generator.predictor.model.to("cpu")

def sava_numpy(save_path, data):
    save_path_s = save_path + "_s.npy"
    save_path_f = save_path + "_f.npy"
    np.save(save_path_s, data["seg_maps"].numpy())
    np.save(save_path_f, data["feature"].numpy())


def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    valid_seg_map = {}
    for mode in ["default", "s", "m", "l"]:
        if mode not in seg_images:
            continue
        tiles = seg_images[mode]
        if tiles.shape[0] == 0:
            continue
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
        valid_seg_map[mode] = seg_map[mode]

    return clip_embeds, valid_seg_map


def get_seg_img(mask, image):
    image = image.copy()
    image[mask["segmentation"] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask["bbox"])
    seg_img = image[y : y + h, x : x + w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    masks_np = masks.detach().cpu().numpy().astype(bool)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)

    num_masks = masks_np.shape[0]
    if num_masks == 0:
        return torch.empty((0,), dtype=torch.long)

    order = np.argsort(-scores_np)
    masks_ord = masks_np[order]
    scores_ord = scores_np[order]
    masks_area = masks_ord.reshape(num_masks, -1).sum(axis=1).astype(np.float32)

    iou_matrix = np.zeros((num_masks, num_masks), dtype=np.float32)
    inner_iou_matrix = np.zeros((num_masks, num_masks), dtype=np.float32)

    for i in range(num_masks):
        for j in range(i, num_masks):
            inter = np.logical_and(masks_ord[i], masks_ord[j]).sum(dtype=np.int64)
            union = np.logical_or(masks_ord[i], masks_ord[j]).sum(dtype=np.int64)
            iou = float(inter) / float(union) if union > 0 else 0.0
            iou_matrix[i, j] = iou

            ai = float(masks_area[i]) if masks_area[i] > 0 else 1.0
            aj = float(masks_area[j]) if masks_area[j] > 0 else 1.0
            ri = float(inter) / ai
            rj = float(inter) / aj

            if ri < 0.5 and rj >= 0.85:
                inner_iou_matrix[i, j] = 1.0 - rj * ri
            if ri >= 0.85 and rj < 0.5:
                inner_iou_matrix[j, i] = 1.0 - rj * ri

    iou_upper = np.triu(iou_matrix, k=1)
    iou_max = iou_upper.max(axis=0)
    inner_u_max = np.triu(inner_iou_matrix, k=1).max(axis=0)
    inner_l_max = np.tril(inner_iou_matrix, k=-1).max(axis=0)

    keep = iou_max <= iou_thr
    keep_conf = scores_ord > score_thr
    keep_inner_u = inner_u_max <= 1 - inner_thr
    keep_inner_l = inner_l_max <= 1 - inner_thr

    k = min(3, num_masks)
    if not keep_conf.any() and k > 0:
        keep_conf[:k] = True
    if not keep_inner_u.any() and k > 0:
        keep_inner_u[:k] = True
    if not keep_inner_l.any() and k > 0:
        keep_inner_l[:k] = True

    keep = keep & keep_conf & keep_inner_u & keep_inner_l
    selected_idx_np = order[keep]
    return torch.from_numpy(selected_idx_np.astype(np.int64))


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in args:
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
        seg_pred = torch.from_numpy(
            np.stack([m["segmentation"] for m in masks_lvl], axis=0)
        )
        iou_pred = torch.from_numpy(
            np.stack([m["predicted_iou"] for m in masks_lvl], axis=0)
        )
        stability = torch.from_numpy(
            np.stack([m["stability_score"] for m in masks_lvl], axis=0)
        )

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image):
    image = cv2.cvtColor(
        image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = masks_update(
        masks_default,
        masks_s,
        masks_m,
        masks_l,
        iou_thr=0.8,
        score_thr=0.7,
        inner_thr=0.5,
    )

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)

            if seg_img.shape[0] == 0 or seg_img.shape[1] == 0:
                seg_img_list.append(np.zeros((224, 224, 3), dtype=np.uint8))
                seg_map[masks[i]["segmentation"]] = i
                continue  # empty mask, special case

            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]["segmentation"]] = i

        if len(seg_img_list) == 0:
            seg_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32, device="cuda")
            return seg_imgs, seg_map

        seg_imgs = np.stack(seg_img_list, axis=0)  # b,H,W,3
        seg_imgs = (
            torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        ).to("cuda")

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images["default"], seg_maps["default"] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images["s"], seg_maps["s"] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images["m"], seg_maps["m"] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images["l"], seg_maps["l"] = mask2segmap(masks_l, image)

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--sam_ckpt_path", type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, "images")
    
    # 获取图片列表并排序
    data_list = os.listdir(img_folder)
    data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    save_folder = os.path.join(dataset_path, "language_features")
    os.makedirs(save_folder, exist_ok=True)

    # 🚀 调用重写后的流式处理函数
    create(img_folder, data_list, save_folder, args)