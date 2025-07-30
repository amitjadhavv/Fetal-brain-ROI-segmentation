import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from torchmetrics.functional import jaccard_index
from monai.transforms import Activations, AsDiscrete

from Datasets.dataset import MRIDataset
from configs.config import Config
from models.AttentionVNet import AttentionVNet


def evaluate_split(split="val", model_path="AV_net_noDNN_model_roi_best.pth"):
    """
    Evaluate Dice and IoU on a given dataset split ("val" or "test").
    """
    image_paths = Config.get_image_paths()
    mask_paths = Config.get_mask_paths()
    dataset = MRIDataset(image_paths, mask_paths, split=split)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    dice_metric = DiceMetric(include_background=False, get_not_nans=True)
    post_pred = Activations(sigmoid=True)
    post_label = AsDiscrete(threshold=0.5)

    # Load model
    model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): val for key, val in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            outputs = model(images)
            outputs = post_pred(outputs)
            outputs = post_label(outputs)

            # Dice (per batch)
            dice = dice_metric(y_pred=outputs, y=masks).mean().item()
            dice_scores.append(dice)

            # IoU (Jaccard index)
            iou = jaccard_index(
                outputs.int(), masks.int(),
                task="binary", num_classes=Config.NUM_CLASSES
            ).item()
            iou_scores.append(iou)

    return dice_scores, iou_scores


def summarize_results(values, name="Metric"):
    """Return mean ± std in thesis-friendly format."""
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    print(f"{name}: {mean_val:.4f} (± {std_val:.4f})")
    return mean_val, std_val


if __name__ == "__main__":
    # --- Validation evaluation
    val_dice, val_iou = evaluate_split(split="val")
    summarize_results(val_dice, "Validation Dice")
    summarize_results(val_iou, "Validation IoU")

    # --- Test evaluation
    test_dice, test_iou = evaluate_split(split="test")
    summarize_results(test_dice, "Test Dice")
    summarize_results(test_iou, "Test IoU")
