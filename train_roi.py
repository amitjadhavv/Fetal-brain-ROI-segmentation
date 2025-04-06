import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
from configs.config import Config
# from monai.losses import DiceLoss
# from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import torch.nn.functional as F
import warnings
from utils.metrics import dice_coefficient
from utils.loss import dice_loss
from torchmetrics.functional import jaccard_index


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
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=None, augmentation_factor=4)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))

model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)

# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
train_loss_history = []
max_train_metric  = 0
# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()  # Start time tracking
    train_loss = 0
    train_metric = 0
    for images, heatmaps in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        outputs = model(images)
        loss = dice_loss(outputs, heatmaps)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        iou = jaccard_index(outputs, heatmaps.int(),task="binary", num_classes=Config.NUM_CLASSES)
        train_metric += iou.item()
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, IoU Score: {train_metric:.4f}, {epoch_time:.2f} seconds, Epoch {epoch + 1} , Current LR: {current_lr}")
    scheduler.step()
    if epoch >900:
        if train_metric > max_train_metric:
            max_train_metric = train_metric
            torch.save(model.state_dict(), "V_net_model_roi_best.pth")
            print(f"Model state dictionary saved to V_net_model_roi_best.pth at Epoch: {epoch + 1} with IoU Score: {train_metric:.4f}")
loss_history = {
    "train_loss": train_loss_history
}
with open("loss_history_roi.json", "w") as f:
    json.dump(loss_history, f)
