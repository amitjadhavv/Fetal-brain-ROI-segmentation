import torch

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()

def multiclass_dice_coefficient(pred, target, num_classes, smooth=1e-6):
    """
    Computes the multi-class Dice coefficient.

    Args:
        pred (torch.Tensor): Model predictions with shape [B, C, D, H, W].
        target (torch.Tensor): Ground truth with shape [B, D, H, W].
        num_classes (int): Number of classes.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Mean Dice coefficient across all classes.
    """
    dice_scores = []

    # Ensure target tensor has no extra channel dimension
    if target.ndim == 5 and target.size(1) == 1:
        target = target.squeeze(1)  # Convert [B, 1, D, H, W] to [B, D, H, W]

    for c in range(num_classes):
        # Extract predictions for class c
        pred_c = pred[:, c]  # Shape: [B, D, H, W]
        target_c = (target == c).float()  # Binary mask for class c, Shape: [B, D, H, W]

        # Compute Dice for class c
        intersection = (pred_c * target_c).sum(dim=(1, 2, 3))  # Intersection over batch
        dice = (2. * intersection + smooth) / (
                pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3)) + smooth
        )
        dice_scores.append(dice.mean().item())  # Average Dice for the class

    # Return mean Dice score across all classes
    return sum(dice_scores) / num_classes
