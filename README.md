# LagMemo: Open-source Extension for 3D Reconstruction and Language Injection

This repository provides the open-source extension of **LagMemo**, focusing on the previously missing components for:

- **3D Reconstruction**
- **Language Injection**
- **Semantic Query**

The codebase is organized as an independent extension around the existing LagMemo pipeline, with minimal modification to the original released repository.

---

## Overview

The overall pipeline is structured into three stages:

1. **3D Reconstruction**  
   Build a 3D Gaussian Splatting (3DGS) scene representation from RGB-D observations.

2. **Language Injection**  
   Extract 2D semantic features from RGB images, then project and fuse them into the 3D scene representation.

3. **Visual Navigation / Semantic Query**  
   Use the semantic-enhanced scene representation for downstream localization and query.

---

## Repository Structure

```bash
./
├── README.md
├── 3DReconstruction/
├── LanguageInjection/
├── LagMemo/
└── your_experiment/
    ├── data/
    ├── logs/
    ├── scripts/
    │   ├── splatam_i.py
    │   ├── export_ply.py
    │   ├── process.sh
    │   ├── lagmemo.sh
    │   └── localize_new.py
    ├── configs/
    └── results/
```

---

## Environment Setup

This project uses **three environments in total**, corresponding to different stages of the pipeline.

### 1. 3D Reconstruction Environment

`3DReconstruction` has been benchmarked with:

- Python 3.10
- PyTorch 1.12.1
- CUDA 11.6

#### Installation

```bash
git clone <your-repo-url>

cd 3DReconstruction
conda create -n 3DReconstruction python=3.10
conda activate 3DReconstruction

conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

---

### 2. Language Injection Environment

This stage contains two submodules:

- `OpenGaussian`
- `LangSplat`

#### Installation

```bash
git clone <your-repo-url>

cd LanguageInjection/OpenGaussian
conda env create --file environment.yml
conda activate LanguageInjection

cd submodules
unzip ashawkey-diff-gaussian-rasterization.zip
pip install ./ashawkey-diff-gaussian-rasterization

cd ../../LangSplat
conda install -c conda-forge -y "python=3.7" "pytorch=1.12.*" "open-clip-torch<3"
pip install "ftfy" "regex" "tqdm" "huggingface_hub<0.20" "safetensors<0.4" "timm<0.9"
```

You also need to install `segment-anything-langsplat` and download the SAM checkpoints from its official repository into `ckpts/`.

---

## Quick Start

### Step 1. Activate the 3D Reconstruction environment

```bash
conda activate 3DReconstruction
```

### Step 2. Generate the 3DGS model

```bash
python your_experiment/scripts/splatam_i.py your_experiment/configs/splatam_new.py
```

### Step 3. Export the PLY file

```bash
python your_experiment/scripts/export_ply.py your_experiment/configs/splatam_new.py
```

---

### Step 4. Activate the Language Injection environment

```bash
conda activate LanguageInjection
```

### Step 5. Perform 2D-level semantic segmentation and language-feature alignment

```bash
cd LangSplat
bash ../../your_experiment/scripts/process.sh
cd ..
```

### Step 6. Generate semantic modeling results

```bash
cd OpenGaussian
bash ../../your_experiment/scripts/lagmemo.sh
```

### Step 7. Run semantic query / localization

```bash
python query_lh/localize_new.py
```

---

## Common Issues

### 1. GPU memory usage is too high

If GPU memory usage is too high, set:

```python
gt_pose = True
```

Otherwise, memory consumption may become very large.

### 2. Dataset loading errors

Check the dataset path configuration in:

```bash
./exp/configs/splatam_new.py
```

Path misconfiguration is a common source of errors.

### 3. SSL error when downloading AlexNet weights / certificates

Possible solutions:

- Download the required files offline
- Temporarily disable SSL verification if your environment permits it

---

## Notes

- This repository is designed as an **independent extension** to the existing LagMemo release.
- The added code is relatively self-contained and does **not heavily modify** the original open-source codebase.
- In practice, only minimal integration changes may be required, such as README updates or script-path adjustments.

---

## TODO

- [x] Reproduce the released part of the pipeline
- [x] Merge code into GitHub
- [ ] Clean up scripts and configs for public release
- [ ] Add dataset preparation instructions
- [ ] Add checkpoints / model dependency instructions
- [ ] Add example outputs and visualization results
- [ ] Add citation and license information

---

## Citation

If you find this project useful, please consider citing:

```bibtex
@misc{lagmemo_extension,
  title={LagMemo Open-source Extension},
  author={YOUR NAME},
  year={2026},
  howpublished={GitHub repository}
}
```

---

## License

This project is released under the [MIT License](LICENSE) unless otherwise specified.

> Please verify compatibility with the original LagMemo repository and all third-party dependencies before final release.

---

## Acknowledgements

This project builds upon the following components:

- LagMemo
- SplaTAM / 3DGS-related reconstruction tools
- LangSplat
- OpenGaussian
- Segment Anything

We sincerely thank the authors of these open-source projects.
