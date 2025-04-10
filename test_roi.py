import torch
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from configs.config import Config
from models.VNet import VNet
from monai.transforms import Activations, AsDiscrete

image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
# Load the test dataset
test_dataset = MRIDataset(image_paths, mask_paths, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

dice_metric = DiceMetric(include_background=False, get_not_nans=True)
post_pred = Activations(sigmoid=True)
post_label = AsDiscrete(threshold=0.5)
model = VNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
model_path = "V_net_model_roi_best.pth"

state_dict = torch.load(model_path)
# Remove 'module.' prefix if present
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
val_metric = 0.0
dice_metric.reset()
with torch.no_grad():
    for images, masks in test_dataloader:
        images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
        outputs = model(images)
        # Convert logits → probabilities → binary predictions
        outputs = post_pred(outputs)
        outputs = post_label(outputs)

        dice = dice_metric(y_pred=outputs, y=masks)
        dice = dice.mean()
        val_metric += dice.item()
    # Aggregate Dice scores

print(f"Mean Dice Coefficient: {val_metric:.4f}")
