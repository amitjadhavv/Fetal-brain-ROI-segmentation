import torch
import torch.nn as nn
import torch.nn.functional as F


class ROI_CNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        """
        Simple CNN model for predicting a Gaussian heatmap indicating the ROI.

        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale MRI).
            output_channels (int): Number of output channels (1 for heatmap prediction).
        """
        super(ROI_CNN, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(32, output_channels, kernel_size=1)

        self.pool = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # Conv + ReLU
        x2 = self.pool(F.relu(self.conv2(x1)))  # Downsample
        x3 = self.pool(F.relu(self.conv3(x2)))  # Downsample
        x4 = self.upsample(F.relu(self.conv4(x3)))  # Upsample
        x5 = self.upsample(x4)  # Upsample again to restore original size
        x6 = torch.sigmoid(self.conv5(x5))  # Sigmoid to normalize to [0,1]
        return x6  # Output has same spatial dimensions as input
