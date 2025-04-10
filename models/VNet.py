import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck_conv = self.conv_block(128, 256)

        # Small DNN to add in the bottleneck
        self.dnn = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Linear(512, 16384),
            nn.ReLU()
        )

        # Decoder
        self.decoder4 = self.conv_block(256 + 128, 128)
        self.decoder3 = self.conv_block(128 + 64, 64)
        self.decoder2 = self.conv_block(64 + 32, 32)
        self.decoder1 = self.conv_block(32 + 16, 16)

        # Final output layer
        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck_conv(F.max_pool3d(enc4, kernel_size=2, stride=2))

        # DNN injection
        B, C, D, H, W = bottleneck.shape
        flat = bottleneck.view(B, -1)  # Flatten

        # Ensure the tensor matches the device
        device = bottleneck.device
        flat = flat.to(device)

        flat = self.dnn(flat)  # Pass through DNN
        bottleneck = flat.view(B, C, D, H, W)  # Reshape back

        # Decoder
        dec4 = self.decoder4(
            torch.cat([F.interpolate(bottleneck, scale_factor=2, mode="trilinear", align_corners=True), enc4], dim=1))
        dec3 = self.decoder3(
            torch.cat([F.interpolate(dec4, scale_factor=2, mode="trilinear", align_corners=True), enc3], dim=1))
        dec2 = self.decoder2(
            torch.cat([F.interpolate(dec3, scale_factor=2, mode="trilinear", align_corners=True), enc2], dim=1))
        dec1 = self.decoder1(
            torch.cat([F.interpolate(dec2, scale_factor=2, mode="trilinear", align_corners=True), enc1], dim=1))

        output = self.final_conv(dec1)
        return output