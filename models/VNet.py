import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self, num_classes=1):
        super(VNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(1, 16, kernel_size=3, dilation=1)
        self.encoder2 = self.conv_block(16, 32, kernel_size=3, dilation=1)
        self.encoder3 = self.conv_block(32, 64, kernel_size=3, dilation=2)
        self.encoder4 = self.conv_block(64, 128, kernel_size=5, dilation=2)
        self.encoder5 = self.conv_block(128, 256, kernel_size=5, dilation=2)  # Extra encoder layer

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, kernel_size=5, dilation=2)

        # Decoder
        self.decoder5 = self.conv_block(512 + 256, 256, kernel_size=5, dilation=2)  # Additional decoder layer
        self.decoder4 = self.conv_block(256 + 128, 128, kernel_size=5, dilation=2)
        self.decoder3 = self.conv_block(128 + 64, 64, kernel_size=3, dilation=2)
        self.decoder2 = self.conv_block(64 + 32, 32, kernel_size=3, dilation=1)
        self.decoder1 = self.conv_block(32 + 16, 16, kernel_size=3, dilation=1)

        # Final output layer
        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Binary segmentation

    def conv_block(self, in_channels, out_channels, kernel_size=3, dilation=1):
        padding = (kernel_size // 2) * dilation  # Adjust padding for dilation
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))
        enc5 = self.encoder5(F.max_pool3d(enc4, kernel_size=2, stride=2))  # Extra encoding step

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc5, kernel_size=2, stride=2))

        # Decoder
        dec5 = self.decoder5(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode="trilinear", align_corners=True), enc5], dim=1))
        dec4 = self.decoder4(torch.cat([F.interpolate(dec5, scale_factor=2, mode="trilinear", align_corners=True), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2, mode="trilinear", align_corners=True), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode="trilinear", align_corners=True), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode="trilinear", align_corners=True), enc1], dim=1))

        output = self.final_conv(dec1)
        return self.sigmoid(output)