#!/bin/bash
# chmod +x scripts/train_lerf.sh
# ./scripts/train_lerf.sh

# !!! Please check the dataset path specified by -s.

# Total training steps: 70k
# 3dgs pre-train: 0~30k
# stage1: 30~40k
# stage2 (coarse-level): 40~50k
# stage2 (fine-level): 50k~70k

# ###############################################
# #              (1/4) figurines
# # Training takes approximately 70 minutes on a 24G 4090 GPU.
# # The object selection effect is better (recommended), the point cloud visualization is poor (not recommended).
# # k1=64, k2=10
# # --pos_weight 0.5
# # --save_memory: Saves memory, but will reduce training speed. If your GPU memory > 24GB, you can omit this flag
# ###############################################
scan="figurines"
gpu_num=3           # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s /gdata/cold1/wuyanmin/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 3 \
    --root_node_num 64 \
    --leaf_node_num 10 \
    --pos_weight 0.5 \
    --save_memory \
    --test_iterations 30000 \
    --eval


# ###############################################
# #              (2/4) waldo_kitchen
# # Training takes approximately 60 minutes on a 24G 4090 GPU.
# # Good point cloud visualization result (recommended), suboptimal object selection effect.
# # k1=64, k2=10
# # --pos_weight 0.5
# # No need to set save_memory, 24G is sufficient.
# ###############################################
scan="waldo_kitchen"
gpu_num=3           # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s /gdata/cold1/wuyanmin/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 3 \
    --root_node_num 64 \
    --leaf_node_num 10 \
    --pos_weight 0.5 \
    --test_iterations 30000 \
    --eval


# ###############################################
# #              (3/4) teatime
# # Training takes approximately 80 minutes on a 24G 4090 GPU.
# # k1=32, k2=10
# # --pos_weight 0.1
# # --save_memory: Saves memory, but will reduce training speed. If your GPU memory > 24GB, you can omit this flag
# ###############################################
scan="teatime"
gpu_num=3       # change
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s /gdata/cold1/wuyanmin/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 3 \
    --root_node_num 32 \
    --leaf_node_num 10 \
    --pos_weight 0.1 \
    --save_memory \
    --test_iterations 30000 \
    --eval


# ###############################################
# #              (4/4) ramen
# # Training takes approximately 40 minutes on a 24G 4090 GPU.
# # The object selection effect is the worst and unstable (not recommended).
# # k1=64, k2=10
# # --pos_weight 0.5
# # --loss_weight 0.01: the weight of intra-mask smooth loss. 0.1 is used for the other scenes.
# # No need to set save_memory, 24G is sufficient.
# ###############################################
scan="ramen"
gpu_num=3
echo "Training for ${scan} ....."
CUDA_VISIBLE_DEVICES=$gpu_num python train.py --port 601$gpu_num \
    -s /gdata/cold1/wuyanmin/OpenGaussian/data/lerf_ovs/${scan} \
    --iterations 70_000 \
    --start_ins_feat_iter 30_000 \
    --start_root_cb_iter 40_000 \
    --start_leaf_cb_iter 50_000 \
    --sam_level 3 \
    --root_node_num 64 \
    --leaf_node_num 10 \
    --pos_weight 0.5 \
    --loss_weight 0.01 \
    --test_iterations 30000 \
    --eval