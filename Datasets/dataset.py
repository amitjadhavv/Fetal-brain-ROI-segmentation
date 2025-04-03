import torch
import random
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio  # For optional data augmentation

from torch.utils.data import DataLoader
def remap_labels(labels, mapping):
    remapped_labels = labels.clone()
    for old, new in mapping.items():
        remapped_labels[labels == old] = new
    return remapped_labels

class MRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, split="train",class_mapping=None, train_ratio=0.9, val_ratio=0.0, test_ratio=0.10,
                 seed=123, transform=None, augmentation_factor=1):
        """
        Args:
            image_paths (list): List of paths to MRI images.
            mask_paths (list): List of paths to segmentation masks.
            transform (callable, optional): Optional transform to apply on images and masks.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping
        self.augmentation_factor = augmentation_factor

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
        return len(self.indices)* self.augmentation_factor

    def __getitem__(self, idx):
        # Load the MRI image and mask
        actual_idx = self.indices[idx//self.augmentation_factor]
        img = nib.load(self.image_paths[actual_idx]).get_fdata()
        mask = nib.load(self.mask_paths[actual_idx]).get_fdata()

        # Normalize the image
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Add channel dimension to both image and mask
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)  # Use long for segmentation labels

        # Resize to 64x64x64
        target_size = (64, 64, 64)
        img = F.interpolate(img.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0)

        # Apply transformations (if any)
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img),
                mask=tio.LabelMap(tensor=mask)
            )
            subject = self.transform(subject)
            img = subject['image'].data
            mask = subject['mask'].data  # Add channel dimension back
        if self.class_mapping is not None:
            mask = remap_labels(mask, self.class_mapping)

        return img, mask
