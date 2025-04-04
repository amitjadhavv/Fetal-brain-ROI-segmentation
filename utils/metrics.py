import torch

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()

def dice_coefficient_thresholded(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes a threshold-based Dice coefficient for predicted and target
    heatmaps (both in [0..1], but not necessarily strictly binary).

    1. Threshold pred and target at `threshold` to get binary masks.
    2. Compute standard Dice overlap on those binary masks.

    Args:
        pred:  shape [N, 1, D, H, W], each voxel in [0..1]
        target: shape [N, 1, D, H, W], each voxel in [0..1]
                (e.g. a Gaussian heatmap).
        threshold: voxel values >= threshold become 1, else 0
        eps: small constant to avoid division by zero

    Returns:
        A scalar Tensor with mean Dice across the batch.
    """
    # 1) Binarize both predicted and target heatmaps
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    # 2) Flatten batch + spatial dims so we can do sum easily
    pred_bin_flat = pred_bin.view(pred_bin.size(0), -1)
    target_bin_flat = target_bin.view(target_bin.size(0), -1)

    intersection = (pred_bin_flat * target_bin_flat).sum(dim=1)
    union = pred_bin_flat.sum(dim=1) + target_bin_flat.sum(dim=1)

    dice_per_sample = (2.0 * intersection + eps) / (union + eps)
    return dice_per_sample.mean()