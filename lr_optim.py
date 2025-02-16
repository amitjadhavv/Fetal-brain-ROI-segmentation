import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from Datasets.dataset import MRIDataset
from models.VNet import VNet
import torchio as tio
import numpy as np
from configs.config import Config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt
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
train_dataset = MRIDataset(image_paths, mask_paths, split="train", transform=transform, class_mapping=class_mapping, augmentation_factor=2)
train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
print(len(train_dataloader))
# class_weight calculation
epsilon = 1e-6
norm_class_weights = None
class_weights=np.zeros(Config.NUM_CLASSES)
for batch in train_dataloader:
    masks = torch.flatten(batch[1]).to(torch.long)
    a=list(torch.bincount(masks).numpy())
    for j in range(Config.NUM_CLASSES):
        class_weights[j]+=a[j]/(sum(a)+epsilon)
class_weights = class_weights/len(train_dataloader)
foreground_weights=(1-class_weights)[1:]
inverse_values = 1 / foreground_weights
norm_class_weights = torch.tensor(inverse_values / inverse_values.sum())
print(norm_class_weights)

# Initialize model
model = VNet(num_classes=Config.NUM_CLASSES)
if torch.cuda.device_count()>1:
    model = DataParallel(model)
model = model.to(Config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
criterion = DiceLoss(include_background=False, softmax=True, squared_pred=True,weight=norm_class_weights, reduction="mean")

#Learning Rate Scheduler (Cosine Annealing for smooth decay)
scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)

# Adaptive Learning Rate Finder
initial_lr = 1e-6
final_lr = 1e-1
num_steps = 100
lrs = []
losses = []

for i, (images, masks) in enumerate(train_dataloader):
    if i >= num_steps:
        break
    images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
    masks = masks.squeeze(1)
    masks = masks.to(torch.long)
    one_hot = torch.nn.functional.one_hot(masks, num_classes=Config.NUM_CLASSES)  # Shape: (N, D, H, W, C)
    masks = one_hot.permute(0, 4, 1, 2, 3)
    optimizer.param_groups[0]['lr'] = initial_lr * (final_lr / initial_lr) ** (i / num_steps)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    lrs.append(optimizer.param_groups[0]['lr'])
    losses.append(loss.item())

# Find the best LR
best_lr = lrs[losses.index(min(losses))]
optimizer.param_groups[0]['lr'] = best_lr
print(f"Optimal Learning Rate Found: {best_lr:.6f}")

# Plot and save the LR finder graph
plt.figure(figsize=(8, 8))
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True)
plt.savefig('learning_rate_finder.png')
plt.show()
