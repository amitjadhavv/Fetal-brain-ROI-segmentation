import torch

def total_variation_loss_3d(x):
    # x: [B, C, D, H, W], typically sigmoid(output)
    tv_z = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    tv_y = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
    tv_x = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
    return tv_x + tv_y + tv_z
