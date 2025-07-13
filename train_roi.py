import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.AttentionVNet import AttentionVNet
import torchio as tio
from configs.config import Config
from monai.losses import DiceLoss
# from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import torch.nn.functional as F
import warnings
from utils.loss import total_variation_loss_3d
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
val_dataset = MRIDataset(image_paths, mask_paths, split="val", transform=None)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset,batch_size=Config.BATCH_SIZE,num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print(len(train_dataloader))

model = AttentionVNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
# Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
# Use sigmoid for binary logits, and keep to_onehot_y=False
dice_loss = DiceLoss(sigmoid=True, to_onehot_y=False)
bce_loss = nn.BCEWithLogitsLoss()
def combined_loss(pred, target):
    return 0.2 * bce_loss(pred, target) + 0.7 * dice_loss(pred, target) + 0.1 * total_variation_loss_3d(F.sigmoid(pred))

# Early stopping setup
early_stop_patience = 20
epochs_without_improvement = 0
best_val_metric = 0
train_loss_history = []
val_loss_history = []
best_model_path = "AV_net_minDNN_model_roi_best.pth"

# # Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    start_time = time.time()  # Start time tracking
    train_loss = 0
    train_metric = 0
    for images, heatmaps in train_dataloader:
        images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
        outputs = model(images)
        loss = combined_loss(outputs, heatmaps)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        outputs = (outputs > 0.5).int()
        iou = jaccard_index((outputs > 0.5), heatmaps.int(),task="binary", num_classes=Config.NUM_CLASSES)
        train_metric += iou.item()
    train_loss /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_metric /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    val_metric = 0
    with torch.no_grad():
        for images, heatmaps in val_loader:
            images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
            outputs = model(images)
            loss = combined_loss(outputs, heatmaps)
            val_loss += loss.item()
            preds = (outputs > 0.5).int()
            iou = jaccard_index(preds, heatmaps.int(), task="binary", num_classes=Config.NUM_CLASSES)
            val_metric += iou.item()

    val_loss /= len(val_loader)
    val_metric /= len(val_loader)
    val_loss_history.append(val_loss)
    end_time = time.time()  # End time tracking
    epoch_time = end_time - start_time
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train IoU Score: {train_metric:.4f}, Val Loss: {val_loss:.4f},  Val IoU: {val_metric:.4f}, Time: {epoch_time:.2f} seconds, Epoch {epoch + 1} , Current LR: {current_lr}")
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model at epoch {epoch + 1} with Val IoU: {val_metric:.4f}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f" No improvement for {epochs_without_improvement} epochs.")

    if epochs_without_improvement >= early_stop_patience:
        print(f" Early stopping triggered at epoch {epoch + 1}. Best Val IoU: {best_val_metric:.4f}")
        break
loss_history = {
    "train_loss": train_loss_history
}
with open("AVloss_history_roi.json", "w") as f:
    json.dump(loss_history, f)
