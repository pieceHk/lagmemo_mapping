#!/bin/bash

dataset_path="/home/htz/works/3DGS/khy_exp_0313/data/Goat-core/dataset/4ok"

CUDA_VISIBLE_DEVICES=6 python preprocess.py --dataset_path $dataset_path

# dataset_path="/home/htz/works/3DGS/0301data/ori_dataset/nfv_f0/0"

# CUDA_VISIBLE_DEVICES=7 python preprocess.py --dataset_path $dataset_path

# dataset_path="/home/htz/works/3DGS/0301data/ori_dataset/tee_f0/0"

# CUDA_VISIBLE_DEVICES=7 python preprocess.py --dataset_path $dataset_path

# dataset_path="/home/htz/works/3DGS/0301data/ori_dataset/tee_f1/0"

# CUDA_VISIBLE_DEVICES=7 python preprocess.py --dataset_path $dataset_path

# dataset_path="/home/htz/works/3DGS/0301data/ori_dataset/5cd_f0/0"

# CUDA_VISIBLE_DEVICES=7 python preprocess.py --dataset_path $dataset_path

# # train the autoencoder
# cd autoencoder
# python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name $dataset_name
# # e.g. python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

# # get the 3-dims language feature of the scene
# python test.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --dataset_name $dataset_name
# # e.g. python test.py --dataset_path ../data/sofa --dataset_name sofa

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

#cd ..

#for level in 1 2 3
#do
#    python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s data/sofa -m output/sofa --start_checkpoint data/sofa/sofa/chkpnt30000.pth --feature_level 3
#done

#for level in 1 2 3
#do
    # render rgb
#    python render.py -m output/${casename}_${level}
    # render language features
    #python render.py -m output/${casename}_${level} --include_feature
    # e.g. python render.py -m output/sofa_3 --include_feature
#done