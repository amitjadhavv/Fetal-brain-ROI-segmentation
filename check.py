from torchsummary import summary
import torch
import torch.nn as nn
from configs.config import Config
from models.AttentionVNet import AttentionVNet
# Create model
model = AttentionVNet(num_classes=Config.NUM_CLASSES)  # n_channels for grayscale MRI input

# Move to device
model = model.to(Config.DEVICE)

# If using multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Dummy input for 3D input of shape [B, C, D, H, W]
dummy_input = torch.randn(1, 1, 64, 64, 64).to(Config.DEVICE)

# Get model summary

summary(model.module if isinstance(model, nn.DataParallel) else model, input_size=(1, 64, 64, 64))
