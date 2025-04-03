import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
from models.CNN import ROI_CNN
import torchio as tio
import numpy as np
from configs.config import Config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import  json
from torch.optim.lr_scheduler import CosineAnnealingLR

def remap_labels(labels, mapping):
    # Remap the labels based on the mapping
    remapped_labels = labels.clone()
    for old, new in mapping.items():
        remapped_labels[labels == old] = new
    return remapped_labels

# Define augmentations using torchio
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),          # Randomly flip along axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=30),  # Apply random scaling and rotation
    tio.RandomNoise(mean=0.0, std=0.1)     # Add random noise
])
class_mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
# Load dataset
image_paths = Config.get_image_paths()
mask_paths = Config.get_mask_paths()
train_dataset = MRIDataset(image_paths, mask_paths, split="train",class_mapping = class_mapping)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
print(len(train_dataloader))
# class_weight calculation
epsilon = 1e-6
class_weights = np.zeros(Config.NUM_CLASSES)

# Compute class frequencies across batches
for batch in train_dataloader:
    masks = torch.flatten(batch[1]).to(torch.long)  # Flatten masks
    bincounts = torch.bincount(masks, minlength=Config.NUM_CLASSES).float()
    class_weights += bincounts.numpy()

# Normalize class frequencies
class_weights = class_weights / class_weights.sum()  # Convert to probability distribution

# Compute inverse class weights (higher for rare classes)
inverse_weights = 1.0 / (class_weights + epsilon)  # Avoid division by zero

# Reduce background class weight (index 0) to prevent over-segmentation
inverse_weights[0] *= 0.2  # Reduce background influence

# Normalize so all weights sum to 1
norm_class_weights = torch.tensor(inverse_weights / inverse_weights.sum(), dtype=torch.float32)

print("Normalized Class Weights:", norm_class_weights)

# Initialize CNN model
# cnn_model = ROI_CNN(input_channels=1, output_channels=1).to(Config.DEVICE)
# criterion1 = DiceLoss()
# dice_metric1 = DiceMetric(include_background=True, reduction="mean")
# optimizer = torch.optim.Adam(cnn_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
#
# # Initialize model
# model = VNet(num_classes=Config.NUM_CLASSES)
# if torch.cuda.device_count()>1:
#     model = DataParallel(model)
# model = model.to(Config.DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
# criterion2 = DiceLoss(include_background=False, softmax=True, squared_pred=True,weight=norm_class_weights, reduction="mean")
# dice_metric = DiceMetric(include_background=False)
# # Learning Rate Scheduler (Cosine Annealing for smooth decay)
# scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
# for epoch in range(Config.NUM_EPOCHS):
#     model.train()
#     train_loss = 0
#     train_metric = 0
#     for images, masks, heatmaps in train_dataloader:
#         images, heatmaps = images.to(Config.DEVICE), heatmaps.to(Config.DEVICE)
#
#
# train_loss_history = []
# # # Training loop
# for epoch in range(Config.NUM_EPOCHS):
#     model.train()
#     train_loss = 0
#     train_metric = 0
#     for images, masks in train_dataloader:
#         images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
#         masks = masks.squeeze(1)
#         masks = masks.to(torch.long)
#         one_hot = torch.nn.functional.one_hot(masks, num_classes=Config.NUM_CLASSES)  # Shape: (N, D, H, W, C)
#         masks = one_hot.permute(0, 4, 1, 2, 3)
#         # Forward pass
#         outputs = model(images)
#         loss = criterion1(outputs, masks)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         dice = dice_metric(y_pred=outputs, y=masks)
#         if dice.ndim > 0:
#             dice = dice.mean()
#         train_metric += dice.item() * images.size(0)
#     train_loss /= len(train_dataloader)
#     train_loss_history.append(train_loss)
#     train_metric /= len(train_dataloader)
#     print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}")
#     scheduler.step()
# model_save_path = "V_net_model_cropped.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"Model state dictionary saved to {model_save_path}")
# loss_history = {
#     "train_loss": train_loss_history
# }
#
# with open("loss_history.json", "w") as f:
#     json.dump(loss_history, f)
