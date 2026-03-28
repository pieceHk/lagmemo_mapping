#!/bin/bash
echo "start exp1:"
# dataset_path="/home/sfs/lagmemo_scenes/4ok/0" 
# dataset_path="/20TBHDD4/khy/data/Goat-core/dataset/4ok"
dataset_path="/home/htz/works/3DGS/khy_exp_0202/data/Goat-core/dataset/4ok" # 原始数据、特征以及extrinsic/w2c.npz
# dataset_path2="/home/sfs/SplaTAM/4_new_lagmemo_scenes/v2/VBz/0"
# dataset_path2="/20TBHDD4/khy/results/splatam_exp_1770102371/0" # results数据
dataset_path2="/home/htz/works/3DGS/khy_exp_0202/results/splatam_exp_2_4ok/2"
casename="Goat-core"
scene_name="4ok_2"
# dataset_path2="/home/sfs/SplaTAM/tee_30_60/0"     # splatam建立3dgs模型
gpu_num=0
mkdir -p "/home/htz/works/3DGS/khy_exp_0202/results/opengaussian_${scene_name}/${casename}"
cp "$0" "/home/htz/works/3DGS/khy_exp_0202/results/opengaussian_${scene_name}/${casename}/cfg.sh"
CUDA_VISIBLE_DEVICES=$gpu_num python OpenGaussian/train.py -s $dataset_path -m "/home/htz/works/3DGS/khy_exp_0202/results/opengaussian_${scene_name}/${casename}" --start_checkpoint $dataset_path2/chkpnt30000.pth --frozen_init_pts \
        --iterations 90_000 \
        --start_ins_feat_iter 30_000 \
        --start_root_cb_iter 50_000 \
        --start_leaf_cb_iter 70_000 \
        --sam_level 3 \
        --root_node_num 32 \
        --leaf_node_num 5 \
        --pos_weight 1.0 \
        --test_iterations 30000 \
        --save_memory \
        --eval \
        --port 6020 \
        # > "/20TBHDD4/khy/logs/${casename}_train.log" 2>&1

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