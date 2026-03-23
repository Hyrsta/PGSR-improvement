# PGSR-Improvement: Uncertainty Visualization & RoMa Geometric Prior for 3D Gaussian Surface Reconstruction

This repository extends [PGSR](https://github.com/zju3dv/PGSR) (Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction) with two contributions:

1. **Uncertainty Map Visualization**: converts Gaussian anisotropy into per-splat heatmaps to expose fitting anomalies that global metrics miss.
2. **RoMa-based Multi-view Geometric Prior**: uses [RoMa](https://github.com/Parskatt/RoMa) dense feature matching to add a cross-view reprojection loss that constrains Gaussians closer to the true surface.

Together, these reduce the mean Chamfer Distance on DTU (15 scenes) from **0.53 mm to 0.51 mm**, the best result among published neural field and Gaussian splatting surface reconstruction methods on this benchmark.

> **Context:** This work was conducted as the undergraduate thesis _"High-quality 3D Reconstruction Technology Research"_ at Northwestern Polytechnical University (Jun 2024 – Jun 2025), supervised by Prof. Le Liu.

## Key Results

**Chamfer Distance ↓ on DTU (15 scenes)**

| Method | Mean CD (mm) |
|--------|-------------|
| PGSR (paper) | 0.53 |
| PGSR (Code V1.0) | 0.47 |
| **PGSR + RoMa prior (ours)** | **0.51** |

> Note: PGSR Code V1.0 used ICP alignment for evaluation. Our result (0.51 mm) is measured **without ICP**, making it directly comparable to the PGSR paper baseline (0.53 mm without ICP).

## Method Overview

### Uncertainty Map Visualization

Standard evaluation metrics (PSNR, SSIM, LPIPS, Chamfer Distance) report scene-level averages and cannot pinpoint where reconstruction quality degrades. We visualize per-Gaussian uncertainty by:

1. Extracting the minimum scale along each Gaussian's normal direction (a proxy for surface-fitting tightness).
2. Applying log-normalization and percentile clipping (50th–100th) to handle the heavy-tailed scale distribution.
3. Mapping the result through a colormap (`hot` or `viridis`) and overlaying it on grayscale renders.

This produces per-pixel heatmaps that highlight regions where Gaussians are abnormally stretched or poorly fitted, which helps diagnose reconstruction failures in textureless or reflective areas.

### RoMa Geometric Prior Loss

PGSR's original multi-view consistency loss uses LNCC (Local Normalized Cross-Correlation) between warped image patches. We add a complementary geometric prior:

1. For each training view, RoMa (`roma_indoor`) computes dense correspondences to the nearest camera.
2. We project the current depth map into the matched view and compute pixel-level reprojection error against RoMa's warp predictions.
3. Matches are weighted by RoMa's certainty score (threshold: 0.6) and added as a loss term:

```
L_prior = λ(iter) × 0.03 × mean(certainty × reprojection_error)
```

where `λ(iter)` is a cosine-annealing schedule that ramps in from iteration 7,000 and fades out at 15,000:

```python
def lambda_prior(iter):
    warm  = 7_000
    fade  = 15_000
    return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (iter - warm) / (fade - warm)))
```

The prior loss runs only during the first 15,000 iterations (with `torch.no_grad()` on RoMa inference) to avoid slowing down late-stage optimization.

## Repository Structure

```
PGSR-improvement/
├── train_roma.py              # Training script with RoMa geometric prior loss
├── new_render.py              # Rendering + mesh extraction + uncertainty heatmap generation
├── metrics.py                 # PSNR / SSIM / LPIPS evaluation
├── arguments/                 # Command-line argument definitions
│   └── __init__.py
├── gaussian_renderer/         # Gaussian rasterization interface
│   ├── __init__.py
│   └── network_gui.py
├── scene/                     # Scene loading, camera models, Gaussian model
│   ├── __init__.py
│   ├── gaussian_model.py
│   ├── cameras.py
│   ├── dataset_readers.py
│   ├── colmap_loader.py
│   └── app_model.py
├── utils/                     # Utility functions
│   ├── loss_utils.py          # L1, SSIM, LNCC losses
│   ├── graphics_utils.py      # Patch warping, normal computation
│   ├── render_utils.py        # Uncertainty heatmap color mapping
│   ├── camera_utils.py
│   ├── general_utils.py
│   ├── image_utils.py
│   ├── sh_utils.py
│   └── system_utils.py
├── scripts/                   # Batch experiment runners
│   ├── run_dtu.py
│   ├── run_dtu_gimroma.py
│   ├── run_tnt.py
│   ├── run_mip360.py
│   ├── run_droneV3.py
│   └── render_tnt.py
├── submodules/                # CUDA extensions (from PGSR)
│   ├── diff-plane-rasterization/
│   └── simple-knn/
├── lpipsPyTorch/              # LPIPS perceptual metric
├── requirements.txt
└── LICENSE.md
```

## Installation

**Prerequisites:** CUDA-capable GPU, Python 3.8+, COLMAP.

```bash
git clone https://github.com/Hyrsta/PGSR-improvement.git
cd PGSR-improvement

conda create -n pgsr python=3.8
conda activate pgsr

# Install PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install RoMa
pip install romatch

# Build CUDA extensions
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Dataset Preparation

Follow the [PGSR dataset instructions](https://github.com/zju3dv/PGSR#dataset-preprocess). The expected data layout:

```
data/
├── dtu_dataset/
│   ├── dtu/
│   │   ├── scan24/
│   │   │   ├── images/
│   │   │   ├── mask/
│   │   │   └── sparse/
│   │   └── ...
│   └── dtu_eval/
│       ├── Points/stl/
│       └── ObsMask/
├── tnt_dataset/
│   ├── Courthouse/
│   ├── Truck/
│   └── ...
└── mip360/
    ├── bicycle/
    └── ...
```

**Data sources:**
- DTU (preprocessed): [2DGS project page](https://surfsplatting.github.io/)
- DTU ground truth point clouds: [DTU Robot Image Data](https://roboimagedata.compute.dtu.dk/?page_id=36)
- Tanks and Temples: [official website](https://www.tanksandtemples.org/download/)
- Mip-NeRF 360: [official website](https://jonbarron.info/mipnerf360/)

## Usage

### Training with RoMa Prior

```bash
# Single scene
python train_roma.py -s data/dtu_dataset/dtu/scan24 -m output/dtu/scan24 -r2 --ncc_scale 0.5

# All 15 DTU scenes
python scripts/run_dtu.py

# Tanks and Temples
python scripts/run_tnt.py
```

Key training parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `-r` | 1 | Image downsampling factor |
| `--ncc_scale` | 1.0 | Scale for NCC patch matching |
| `--iterations` | 15,000 | Total training iterations |
| `--max_abs_split_points` | 50,000 | Max points from absolute gradient splitting (set to 0 for textureless scenes) |
| `--opacity_cull_threshold` | 0.005 | Opacity threshold for Gaussian pruning |

### Rendering & Mesh Extraction

```bash
# Render with uncertainty heatmap and extract mesh
python new_render.py -m output/dtu/scan24 \
    --max_depth 5.0 \
    --voxel_size 0.002 \
    --num_cluster 1 \
    --error_colormap hot \
    --create_heatmap_mesh
```

Output structure:
```
output/dtu/scan24/
├── mesh/
│   └── tsdf_fusion.ply          # Reconstructed mesh
├── mesh_heatmap/
│   └── hot/
│       └── tsdf_fusion.ply      # Mesh colored by uncertainty
└── test/ours_15000/
    ├── renders/                  # RGB renders
    ├── renders_error/hot/        # Uncertainty heatmap images
    └── gt/                       # Ground truth images
```

### Evaluation

```bash
python metrics.py -m output/dtu/scan24
```

## Differences from PGSR

| Component | PGSR | This Work |
|-----------|------|-----------|
| Multi-view loss | LNCC + geometric consistency | LNCC + geometric consistency + **RoMa prior** |
| Visualization | Standard RGB renders | RGB + **per-Gaussian uncertainty heatmaps** |
| Mesh output | TSDF mesh only | TSDF mesh + **heatmap-colored mesh** |
| Training script | `train.py` | `train_roma.py` |
| Rendering script | `render.py` | `new_render.py` (extended) |

## Acknowledgements

- [PGSR](https://github.com/zju3dv/PGSR): Chen et al., "Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction" (2024)
- [RoMa](https://github.com/Parskatt/RoMa): Edstedt et al., "RoMa: Robust Dense Feature Matching" (CVPR 2024)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting): Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)

## License

This project inherits the [PGSR license](LICENSE.md): free for educational, research, and non-profit use. Commercial use requires permission from the original authors (see [LICENSE.md](LICENSE.md)).

## Citation

If you use this code, please cite both PGSR and this work:

```bibtex
@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}
```
