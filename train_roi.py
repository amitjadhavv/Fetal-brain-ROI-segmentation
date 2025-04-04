import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
from configs.config import Config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import torch.nn.functional as F
import warnings

# This ignores *all* warnings of any category
warnings.simplefilter('ignore')
# Define augmentations using torchio
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),          # Randomly flip along axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Apply random scaling and rotation
    tio.RandomNoise(mean=0.0, std=0.1)     # Add random noise
])
image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, augmentation_factor=4)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))

model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
dice_loss_fn = DiceLoss(include_background=False, squared_pred=True, reduction="mean")
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
train_loss_history = []
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()  # Start time tracking
    train_loss = 0
    train_metric = 0
    for images, heatmaps in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        outputs = model(images)
        pred_map = F.relu(outputs)
        map_sum = pred_map.sum(dim=1, keepdim=True) + 1e-8
        pred_probs = pred_map / map_sum
        # Forward pass
        pred_probs = torch.log(torch.clamp(pred_probs, min=1e-8))
        # target_probs = heatmaps / (heatmaps.sum() + 1e-8)
        # target_probs = torch.clamp(target_probs, min=1e-8)
        # target_probs = target_probs / target_probs.sum()
        kl_loss = kl_loss_fn(input=pred_probs, target=heatmaps)
        outputs = torch.sigmoid(outputs)
        probs = torch.sigmoid(outputs)
        probs = probs.clamp(min=1e-8, max=1.0 - 1e-8)
        threshold = 0.1
        pred_bin = (probs >= threshold).float()
        heatmaps_bin = (heatmaps >= threshold).float()
        print("pred_bin unique:", torch.unique(pred_bin))
        print("heatmaps_bin unique:", torch.unique(heatmaps_bin))
        dice_loss = dice_loss_fn(pred_bin, pred_bin)
        print("diceloss: ",dice_loss)
        loss = kl_loss + dice_loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        dice = dice_metric(y_pred=pred_bin, y=heatmaps_bin)
        if dice.ndim > 0:
            dice = dice.mean()
        train_metric += dice.item()
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}, Time: {epoch_time:.2f} seconds, Epoch {epoch+1} , Current LR: {current_lr}")
    scheduler.step()
model_save_path = "V_net_model_roi.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")
loss_history = {
    "train_loss": train_loss_history
}
with open("loss_history_roi.json", "w") as f:
    json.dump(loss_history, f)
