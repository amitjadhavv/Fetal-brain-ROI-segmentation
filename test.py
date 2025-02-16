import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from configs.config import Config
from models.VNet import VNet

image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
class_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
# Load the test dataset
test_dataset = MRIDataset(image_paths, mask_paths, split="test",class_mapping=class_mapping)
test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

model = VNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
model_path = "V_net_model_cropped.pth"

state_dict = torch.load(model_path)
# Remove 'module.' prefix if present
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
val_loss = 0.0
val_metric = 0.0

with torch.no_grad():
    for images, masks in test_dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

        # Forward pass
        outputs = model(images)
        # one hot encoding
        masks = masks.squeeze(1)
        masks = masks.to(torch.long)
        print("Mask shape:", masks.shape)
        print("Unique values in mask:", torch.unique(masks))
        one_hot = torch.nn.functional.one_hot(masks, num_classes=Config.NUM_CLASSES)  # Shape: (N, D, H, W, C)
        masks = one_hot.permute(0, 4, 1, 2, 3)

        dice = dice_metric(y_pred=outputs, y=masks)
        if dice.ndim > 0:
            dice = dice.mean()

    # Aggregate Dice scores
    mean_dice = dice.item()
    dice_metric.reset()

print(f"Mean Dice Coefficient: {mean_dice:.4f}")
