<div align="center">

# [NeurIPS2024🔥] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding

<h3>
  <a href="https://arxiv.org/abs/2406.02058"><strong>Paper</strong></a> | 
  <a href="https://3d-aigc.github.io/OpenGaussian/"><strong>Project Page</strong></a>
</h3>

<!-- [**Paper**](https://arxiv.org/abs/2406.02058) | [**Project Page**](https://3d-aigc.github.io/OpenGaussian/) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/abs/2406.02058)
[![Project Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://3d-aigc.github.io/OpenGaussian/) -->

[Yanmin Wu](https://yanmin-wu.github.io/)<sup>1</sup>, [Jiarui Meng](https://scholar.google.com/citations?user=N_pRAVAAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Haijie Li](https://villa.jianzhang.tech/people/haijie-li-%E6%9D%8E%E6%B5%B7%E6%9D%B0/)<sup>1</sup>, [Chenming Wu](https://chenming-wu.github.io/)<sup>2*</sup>, [Yahao Shi](https://scholar.google.com/citations?user=-VJZrUkAAAAJ&hl=en)<sup>3</sup>, [Xinhua Cheng](https://cxh0519.github.io/)<sup>1</sup>, 
[Chen Zhao](https://openreview.net/profile?id=~Chen_Zhao9)<sup>2</sup>, [Haocheng Feng](https://openreview.net/profile?id=~Haocheng_Feng1)<sup>2</sup>, [Errui Ding](https://scholar.google.com/citations?user=1wzEtxcAAAAJ&hl=zh-CN)<sup>2</sup>, [Jingdong Wang](https://jingdongwang2017.github.io/)<sup>2</sup>, [Jian Zhang](https://jianzhang.tech/)<sup>1*</sup>

<sup>1</sup> Peking University, <sup>2</sup> Baidu VIS, <sup>3</sup> Beihang University

</div>

## 0. Installation

The installation of OpenGaussian is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
```
git clone https://github.com/yanmin-wu/OpenGaussian.git
```
Then install the dependencies:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting

# the rasterization lib comes from DreamGaussian
cd OpenGaussian/submodules
unzip ashawkey-diff-gaussian-rasterization.zip
pip install ./ashawkey-diff-gaussian-rasterization
```
+ other additional dependencies: bitarray, scipy, [pytorch3d](https://anaconda.org/pytorch3d/pytorch3d/files)
    ```shell
    pip install bitarray scipy
    
    # install a pytorch3d version compatible with your PyTorch, Python, and CUDA.
    ```
+ `simple-knn` is not required

---

## 1. ToDo list

+ [x] ~~Point feature visualization~~
+ [ ] Data preprocessing
+ [ ] Improved SAM mask extraction (extracting only one layer)
+ [ ] Click to Select 3D Object

---

## 2. Data preparation
The files are as follows:
```
[DATA_ROOT]
├── [1] scannet/
│   │   ├── scene0000_00/
|   |   |   |── color/
|   |   |   |── language_features/
|   |   |   |── points3d.ply
|   |   |   |── transforms_train/test.json
|   |   |   |── *_vh_clean_2.labels.ply
│   │   ├── scene0062_00/
│   │   └── ...
├── [2] lerf_ovs/
│   │   ├── figurines/ & ramen/ & teatime/ & waldo_kitchen/
|   |   |   |── images/
|   |   |   |── language_features/
|   |   |   |── sparse/
│   │   ├── label/
```
+ **[1] Prepare ScanNet Data**
    + You can directly download our pre-processed data: [**OneDrive**](https://onedrive.live.com/?authkey=%21AIgsXZy3gl%5FuKmM&id=744D3E86422BE3C9%2139813&cid=744D3E86422BE3C9). Please unzip the `color.zip` and `language_features.zip` files.
    + The ScanNet dataset requires permission for use, following the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission.
    + The preprocessing script will be updated later.
+ **[2] Prepare lerf_ovs Data**
    + You can directly download our pre-processed data: [**OneDrive**](https://onedrive.live.com/?authkey=%21AIgsXZy3gl%5FuKmM&id=744D3E86422BE3C9%2139815&cid=744D3E86422BE3C9) (re-annotated by LangSplat). Please unzip the `images.zip` and `language_features.zip` files.
+ **Mask and Language Feature Extraction Details**
    + We use the tools provided by LangSplat to extract the SAM mask and CLIP features, but we only use the large-level mask.

---

## 3. Training
### 3.1 ScanNet
```shell
chmod +x scripts/train_scannet.sh
./scripts/train_scannet.sh
```
+ Please ***check*** the script for more details and ***modify*** the dataset path.
+ you will see the following processes during training:
    ```shell
    [Stage 0] Start 3dgs pre-train ... (step 0-30k)
    [Stage 1] Start continuous instance feature learning ... (step 30-50k)
    [Stage 2.1] Start coarse-level codebook discretization ... (step 50-70k)
    [Stage 2.2] Start fine-level codebook discretization ... (step 70-90k)
    [Stage 3] Start 2D language feature - 3D cluster association ... (1 min)
    ```
+ Intermediate results from different stages can be found in subfolders `***/train_process/stage*`. (The intermediate results of stage 3 are recommended to be observed in the LeRF dataset.)

### 3.2 LeRF_ovs
```shell
chmod +x scripts/train_lerf.sh
./scripts/train_lerf.sh
```
+ Please ***check*** the script for more details and ***modify*** the dataset path.
+ you will see the following processes during training:
    ```shell
    [Stage 0] Start 3dgs pre-train ... (step 0-30k)
    [Stage 1] Start continuous instance feature learning ... (step 30-40k)
    [Stage 2.1] Start coarse-level codebook discretization ... (step 40-50k)
    [Stage 2.2] Start fine-level codebook discretization ... (step 50-70k)
    [Stage 3] Start 2D language feature - 3D cluster association ... (1 min)
    ```
+ Intermediate results from different stages can be found in subfolders `***/train_process/stage*`.

### 3.3 Custom data
+ TODO

---

## 4. Render & Eval & Downstream Tasks

### 4.1 3D Instance Feature Visualization
+ Please install `open3d` first, and then execute the following command on a system with UI support:
    ```python
    python scripts/vis_opengs_pts_feat.py
    ```
    + Please specify `ply_path` in the script as the PLY file `output/xxxxxxxx-x/point_cloud/iteration_x0000/point_cloud.ply` saved at different stages.
    + During the training process, we have saved the first three dimensions of the 6D features as colors for visualization; see [here](https://github.com/yanmin-wu/OpenGaussian/blob/2845b9c744c1b06ac6930ffa2d2a6f9167f1b843/scene/gaussian_model.py#L272).

### 4.2 Render 2D Feature Map
+ The same rendering method as the 3DGS rendering colors.
    ```shell
    python render.py -m "output/xxxxxxxx-x"
    ```
    You can find the rendered feature maps in subfolders `renders_ins_feat1` and `renders_ins_feat2`.

### 4.3 ScanNet Evalution (Open-Vocabulary Point Cloud Understanding)
> Due to code optimization and the use of more suitable hyperparameters, the latest evaluation metrics may be higher than those reported in the paper. 
+ Evaluate text-guided segmentation performance on ScanNet for 19, 15, and 10 categories.
    ```shell
    # unzip the pre-extracted text features
    cd assets
    unzip text_features.zip

    # 1. please check the `gt_file_path` and `model_path` are correct
    # 2. specify `target_id` as 19, 15, or 10 categories.
    python scripts/eval_scannet.py
    ```

### 4.4 LeRF Evalution (Open-Vocabulary Object Selection in 3D Space)
+ (1) First, render text-selected 3D Gaussians into multi-view images.
    ```shell
    # unzip the pre-extracted text features
    cd assets
    unzip text_features.zip

    # 1. specify the model path using -m
    # 2. specify the scene name: figurines, teatime, ramen, waldo_kitchen
    python render_lerf_by_text.py -m "output/xxxxxxxx-x" --scene_name "figurines"
    ```
    The object selection results are saved in `output/xxxxxxxx-x/text2obj/ours_70000/renders_cluster`.

+ (2) Then, compute evaluation metrics.
    > Due to code optimization and the use of more suitable hyperparameters, the latest evaluation metrics may be higher than those reported in the paper. 
    > The metrics may be unstable due to the limited evaluation samples of LeRF.
    ```shell
    # 1. change path_gt and path_pred in the script
    # 2. specify the scene name: figurines, teatime, ramen, waldo_kitchen
    python scripts/compute_lerf_iou.py --scene_name "figurines"
    ```

### 4.5 Click to Select 3D Object

+ TODO

---

## 5. Acknowledgements
We are quite grateful for [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [LangSplat](https://github.com/minghanqin/LangSplat), [CompGS](https://github.com/UCDvision/compact3d), [LEGaussians](https://github.com/buaavrcg/LEGaussians), [SAGA](https://github.com/Jumpat/SegAnyGAussians), and [SAM](https://segment-anything.com/).

---

## 6. Citation

```
@article{wu2024opengaussian,
    title={OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding},
    author={Wu, Yanmin and Meng, Jiarui and Li, Haijie and Wu, Chenming and Shi, Yahao and Cheng, Xinhua and Zhao, Chen and Feng, Haocheng and Ding, Errui and Wang, Jingdong and others},
    journal={arXiv preprint arXiv:2406.02058},
    year={2024}
}
```

---

## 7. Contact
If you have any questions about this project, please feel free to contact [Yanmin Wu](https://yanmin-wu.github.io/): wuyanminmax[AT]gmail.com
