# Fetal Brain ROI Segmentation

3D region-of-interest (ROI) segmentation of the fetal brain in MRI volumes using an **Attention V-Net** built in PyTorch. The model predicts a coarse binary ROI mask around the fetal brain from a low-resolution (64×64×64) copy of the volume. The mask can then be resized back to the original image space, e.g. to crop the brain for downstream processing.

## Overview

| Item | Details |
| --- | --- |
| Task | Binary 3D segmentation of a fetal-brain ROI |
| Input | Fetal MRI volume (NIfTI, `.nii` / `.nii.gz`), resized to 64×64×64 |
| Output | Binary ROI mask in the original image shape and affine (NIfTI) |
| Model | Attention V-Net (3D U-Net-style encoder/decoder with attention gates) |
| Loss | 0.2 · BCE + 0.7 · Dice + 0.1 · 3D total variation |
| Metrics | Dice and IoU (Jaccard) |
| Best validation IoU | 0.9306 (epoch 398, early stopping at epoch 438) |

### How the ROI targets are made

The dataset provides fetal-brain label masks, not ROI masks. [Binary_combined_gaussian.py](Binary_combined_gaussian.py) turns each label into a smooth ROI target:

1. Compute the center of mass of the label.
2. Compute a per-axis standard deviation of the label voxels, scaled by `alpha`, with a floor of `min_sigma`.
3. Build an ellipsoidal 3D Gaussian heatmap from the center and sigmas.
4. Threshold the heatmap (default `0.1`) to get a binary ellipsoidal mask. These masks are the training targets.

## Repository structure

```
.
├── configs/config.py                # Paths, batch size, LR, epochs, device
├── Datasets/dataset.py              # MRIDataset: normalization, resize, split, augmentation
├── models/
│   ├── AttentionVNet.py             # Attention V-Net (used for training/inference)
│   └── VNet.py                      # Plain V-Net baseline
├── utils/
│   ├── loss.py                      # 3D total variation loss
│   └── support.py                   # EarlyStopping helper
├── data_prepration_validation.py    # Pairs images with labels, drops shape mismatches
├── Binary_combined_gaussian.py      # Label -> Gaussian heatmap -> binary ROI mask
├── train_roi.py                     # Training loop
├── final_results_val_test.py        # Dice / IoU on the val and test splits
├── inference.py                     # Single-volume inference
├── lossgraph.py                     # Plots loss/IoU curves from the training log
├── summary.py, check.py             # Model summaries (torchinfo / torchsummary)
├── AV_net_noDNN_model_roi_best.pth  # Trained checkpoint (best validation IoU)
├── run_avnet_noDNN_1122983.log      # Training log (HPC run, A100 40GB)
├── loss_curves.png, iou_curves.png  # Training curves
└── MRI_data/                        # Data (see "Data layout")
```

## Installation

Requires Python 3 and, for reasonable training speed, a CUDA GPU. The checkpoint was trained with PyTorch 2.5.1 (CUDA 11.8).

```bash
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate

# PyTorch with CUDA 11.8 (adjust to your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Remaining dependencies
pip install monai==1.4.0 torchio==0.20.1 torchmetrics==1.7.0 torchinfo==1.8.0 \
            nibabel==5.3.2 numpy==1.26.3 scipy==1.14.1 matplotlib==3.10.0 torchsummary==1.5.1
```

[requirements.txt](requirements.txt) is the full pinned environment (`pip freeze`); [clean_requirements.txt](clean_requirements.txt) lists only a few top-level packages and is incomplete.

## Data layout

Place data under `MRI_data/`:

```
MRI_data/
├── original-images/        # Raw fetal MRI volumes (.nii / .nii.gz)
├── original-labels/        # Matching label volumes (.nii.gz)
├── new_images/             # Created by data_prepration_validation.py  (image_001.nii, ...)
├── new_labels/             # Created by data_prepration_validation.py  (label_001.nii, ...)
├── new_global_heatmaps/    # Created by Binary_combined_gaussian.py
└── new_global_masks/       # Created by Binary_combined_gaussian.py (training targets)
```

Training reads `MRI_data/new_images` and `MRI_data/new_global_masks`, matched by sorted filename order. The data path is set in [configs/config.py](configs/config.py).

The dataset is split with a fixed seed (`246`) into **80% train / 10% validation / 10% test**. Training volumes are repeated 4× per epoch (`augmentation_factor=4`).

## Usage

### 1. Prepare the data

```bash
python data_prepration_validation.py
```

Matches raw images to labels by filename prefix, copies pairs with identical shapes to `new_images/` and `new_labels/`, and writes `size_mismatched_pairs.csv`.

Then generate the ROI targets:

```bash
python Binary_combined_gaussian.py
```

> **Note:** the `__main__` block of this script only sets the folder paths and hyperparameters. It does not call `main(...)`. Add this line at the end of the block before running it:
>
> ```python
> main(train_images_dir, train_labels_dir, out_heatmaps_dir, out_binary_dir, alpha, min_sigma, threshold)
> ```

### 2. Train

```bash
python train_roi.py
```

Settings from [configs/config.py](configs/config.py) and [train_roi.py](train_roi.py):

| Setting | Value |
| --- | --- |
| Batch size | 1 |
| Optimizer | Adam, lr 1e-5, weight decay 1e-4 |
| Scheduler | Cosine annealing (`eta_min=1e-8`) |
| Max epochs | 500 |
| Early stopping | patience of 40 epochs on validation IoU |
| Checkpoint | Best validation IoU saved to `AV_net_noDNN_model_roi_best.pth` |

If more than one GPU is available, the model is wrapped in `DataParallel`. Checkpoints with the `module.` prefix are handled when loading. The TorchIO augmentation pipeline (flips, affine, noise) is defined in `train_roi.py` but not passed to the dataset (`transform=None`).

### 3. Evaluate

```bash
python final_results_val_test.py
```

Prints mean ± standard deviation of Dice and IoU for the validation and test splits using `AV_net_noDNN_model_roi_best.pth`.

### 4. Run inference on one volume

Edit the paths in the `__main__` block of [inference.py](inference.py) (`sample_image_path`, `image`, `model_path`, `output_mask_path`). They currently point to the author's machine. Then run:

```bash
python inference.py
```

Pipeline: percentile normalization → resize to 64³ → forward pass → sigmoid and 0.5 threshold → keep the largest connected component → nearest-neighbour resize to the original shape → save NIfTI with the original affine.

To use it from Python:

```python
from inference import run_inference_single_image

run_inference_single_image("MRI_data/new_images/image_335.nii",
                           "AV_net_noDNN_model_roi_best.pth",
                           "predicted_image_335.nii")
```

### 5. Plot training curves and inspect the model

```bash
python lossgraph.py   # parses run_avnet_noDNN_1122983.log -> loss_curves.png, iou_curves.png
python summary.py     # torchinfo summary (plain VNet)
python check.py       # torchsummary of AttentionVNet
```

## Model

[models/AttentionVNet.py](models/AttentionVNet.py) is a 3D encoder–decoder with four downsampling stages (16 → 32 → 64 → 128 channels), a 256-channel bottleneck, and four decoder stages. Each skip connection passes through an attention gate before concatenation. Convolution blocks are two 3×3×3 convs with instance norm, ReLU and 3D dropout (p=0.1). A 1×1×1 conv produces the single-channel logit map. An optional bottleneck DNN is left commented out in the code, which the checkpoint name `noDNN` refers to.

## Preprocessing

- Intensities are clipped to the 1st–99th percentile, then min-max scaled to [0, 1] (`robust_normalize`).
- Volumes are resized to 64×64×64 (trilinear for images, nearest for masks).

## Results

Best validation IoU during training was **0.9306** at epoch 398. Training stopped early at epoch 438. See [loss_curves.png](loss_curves.png) and [iou_curves.png](iou_curves.png). Run `final_results_val_test.py` to reproduce the validation and test Dice/IoU.

## Notes

- Paths in [inference.py](inference.py) are hard-coded and must be edited.
- The dataset is not included in the repository's tracked files. Provide your own fetal MRI volumes and labels.
- No license file is present. Add one before redistributing.
