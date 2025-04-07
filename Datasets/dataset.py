import torch
import random
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader

def robust_normalize(img, lower_percentile=1, upper_percentile=99):
    """
    Apply robust clipping using the given percentiles, then min-max normalize to [0, 1].
    """
    # 1) Robust Clipping
    low_val = np.percentile(img, lower_percentile)
    high_val = np.percentile(img, upper_percentile)
    clipped = np.clip(img, low_val, high_val)

    # 2) Min-Max Normalization
    normalized = (clipped - low_val) / (high_val - low_val)
    return normalized

class MRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, split="train", train_ratio=0.85, val_ratio=0.0,
                 seed=123, transform=None, augmentation_factor=1,
                 lower_percentile=1, upper_percentile=99):
        """
        Args:
            image_paths (list): List of paths to MRI images.
            mask_paths (list): List of paths to segmentation masks.
            split (str): 'train', 'val', or 'test'.
            train_ratio (float): Proportion of data for training.
            val_ratio (float): Proportion of data for validation.
            seed (int): Random seed for shuffling.
            transform (callable, optional): Optional transform for data augmentation.
            augmentation_factor (int): Factor by which to artificially expand dataset.
            lower_percentile (float): Lower percentile for robust clipping (default=1).
            upper_percentile (float): Upper percentile for robust clipping (default=99).
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.split = split
        self.transform = transform
        self.augmentation_factor = augmentation_factor
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        # Shuffle data with seed
        data = list(zip(image_paths, mask_paths))
        random.seed(seed)
        random.shuffle(data)
        self.image_paths, self.mask_paths = zip(*data)

        # Compute split indices
        total_files = len(self.image_paths)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)

        self.train_indices = range(0, train_count)
        self.val_indices = range(train_count, train_count + val_count)
        self.test_indices = range(train_count + val_count, total_files)

        # Select the appropriate split
        if split == "train":
            self.indices = self.train_indices
        elif split == "val":
            self.indices = self.val_indices
        elif split == "test":
            self.indices = self.test_indices
        else:
            raise ValueError("Invalid split! Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        # Multiply by augmentation_factor so each sample is repeated
        return len(self.indices) * self.augmentation_factor

    def __getitem__(self, idx):
        # Map the index to the correct file index
        actual_idx = self.indices[idx // self.augmentation_factor]
        image_path = self.image_paths[actual_idx]
        mask_path = self.mask_paths[actual_idx]

        # Load the MRI image and mask
        img_npy = nib.load(image_path).get_fdata()
        mask_npy = nib.load(mask_path).get_fdata()

        # Preprocess the image using the separate robust_normalize function
        img_npy = robust_normalize(img_npy, self.lower_percentile, self.upper_percentile)

        # Expand channel dimension
        img_npy = np.expand_dims(img_npy, axis=0)  # shape: (1, D, H, W)
        mask_npy = np.expand_dims(mask_npy, axis=0)

        # Convert to torch tensors
        img = torch.tensor(img_npy, dtype=torch.float32)
        mask = torch.tensor(mask_npy, dtype=torch.long)

        # Resize both to 64x64x64
        target_size = (64, 64, 64)
        img = F.interpolate(img.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0)

        # Apply TorchIO transforms (if any)
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img),
                mask=tio.LabelMap(tensor=mask)
            )
            subject = self.transform(subject)
            img = subject['image'].data
            mask = subject['mask'].data

        return img, mask
