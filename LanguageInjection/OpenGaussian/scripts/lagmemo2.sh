#!/bin/bash
echo "start exp2:"
dataset_name="0504data"
dataset_path="./5cd"     # dataset path, change
casename="5cd_32_5_30_60_seem_new"
gpu_num=0
mkdir -p "output/${casename}"
cp "$0" "output/${casename}/cfg.sh"
CUDA_VISIBLE_DEVICES=$gpu_num python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --frozen_init_pts \
        --iterations 90_000 \
        --start_ins_feat_iter 30_000 \
        --start_root_cb_iter 50_000 \
        --start_leaf_cb_iter 70_000 \
        --sam_level 0 \
        --root_node_num 32 \
        --leaf_node_num 5 \
        --pos_weight 1.0 \
        --test_iterations 30000 \
        --save_memory \
        --eval \
        --port 6017

# sleep 30
# echo "start exp2:"
# dataset_name="0504data"
# dataset_path="/home/sfs/0504data/result/0"     # dataset path, change
# casename="0504data_32_5_sam3"
# gpu_num=1

# CUDA_VISIBLE_DEVICES=$gpu_num python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --frozen_init_pts \
#         --iterations 90_000 \
#         --start_ins_feat_iter 30_000 \
#         --start_root_cb_iter 50_000 \
#         --start_leaf_cb_iter 70_000 \
#         --sam_level 3 \
#         --root_node_num 32 \
#         --leaf_node_num 5 \
#         --pos_weight 1.0 \
#         --test_iterations 30000 \
#         --save_memory \
#         --eval \
#         --port 601$gpu_num

# sleep 30
# echo "start exp3:"
# dataset_name="0504data"
# dataset_path="/home/sfs/0504data/result/0"     # dataset path, change
# casename="0504data_20_5_sam3"
# gpu_num=1

# CUDA_VISIBLE_DEVICES=$gpu_num python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --frozen_init_pts \
#         --iterations 90_000 \
#         --start_ins_feat_iter 30_000 \
#         --start_root_cb_iter 50_000 \
#         --start_leaf_cb_iter 70_000 \
#         --sam_level 3 \
#         --root_node_num 20 \
#         --leaf_node_num 5 \
#         --pos_weight 1.0 \
#         --test_iterations 30000 \
#         --save_memory \
#         --eval \
#         --port 601$gpu_num

# sleep 30
# echo "start exp4:"
# dataset_name="0504data"
# dataset_path="/home/sfs/0504data/result/0"     # dataset path, change
# casename="0504data_50_4_sam0"
# gpu_num=1

# CUDA_VISIBLE_DEVICES=$gpu_num python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --frozen_init_pts \
#         --iterations 90_000 \
#         --start_ins_feat_iter 30_000 \
#         --start_root_cb_iter 50_000 \
#         --start_leaf_cb_iter 70_000 \
#         --sam_level 0 \
#         --root_node_num 50 \
#         --leaf_node_num 4 \
#         --pos_weight 1.0 \
#         --test_iterations 30000 \
#         --save_memory \
#         --eval \
#         --port 601$gpu_num

# sleep 30
# echo "start exp5:"
# dataset_name="0504data"
# dataset_path="/home/sfs/0504data/result/0"     # dataset path, change
# casename="0504data_50_4_sam3"
# gpu_num=1

# CUDA_VISIBLE_DEVICES=$gpu_num python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --frozen_init_pts \
#         --iterations 90_000 \
#         --start_ins_feat_iter 30_000 \
#         --start_root_cb_iter 50_000 \
#         --start_leaf_cb_iter 70_000 \
#         --sam_level 3 \
#         --root_node_num 50 \
#         --leaf_node_num 4 \
#         --pos_weight 1.0 \
#         --test_iterations 30000 \
#         --save_memory \
#         --eval \
#         --port 601$gpu_num