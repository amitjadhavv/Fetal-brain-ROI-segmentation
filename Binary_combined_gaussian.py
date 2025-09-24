import os
import argparse
import numpy as np
import nibabel as nib
import torch

def load_nifti(file_path):
    """
    Loads a NIfTI image and returns its data as a NumPy array along with its affine matrix.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def save_nifti(data, filename, affine):
    """
    Saves a 3D numpy array as a NIfTI file.
    """
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, filename)

def get_center_of_mass(mask):
    """
    Computes the center of mass of a given binary mask.
    """
    indices = np.argwhere(mask > 0)
    if len(indices) == 0:
        raise ValueError("No landmarks found in the mask!")
    center = np.mean(indices, axis=0)
    return tuple(int(c) for c in center)

def compute_axiswise_sigmas(mask, alpha=1.0, min_sigma=1.0):
    """
    Computes an axiswise sigma (stdev) for x, y, z based on the standard deviation of nonzero voxels.
    :param mask: 3D numpy array (D, H, W) with the segmentation mask.
    :param alpha: Additional multiplier for the computed std devs.
    :param min_sigma: Minimum value for each sigma to avoid extremely small Gaussians.
    :return: (sigma_x, sigma_y, sigma_z)
    """
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        # fallback if mask is empty
        return (3.0, 3.0, 3.0)

    # Standard deviation along each axis
    stdev_x = np.std(coords[:, 0])
    stdev_y = np.std(coords[:, 1])
    stdev_z = np.std(coords[:, 2])

    # Scale by alpha
    sigma_x = max(alpha * stdev_x, min_sigma)
    sigma_y = max(alpha * stdev_y, min_sigma)
    sigma_z = max(alpha * stdev_z, min_sigma)

    return (sigma_x, sigma_y, sigma_z)

def create_ellipsoidal_heatmap(image_shape, center, sigma_x, sigma_y, sigma_z):
    """
    Generates an ellipsoidal Gaussian heatmap centered at 'center' with separate
    sigma for each axis.
    """
    cx, cy, cz = center
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(image_shape[0]),
        torch.arange(image_shape[1]),
        torch.arange(image_shape[2]),
        indexing='ij'
    )

    # Ellipsoidal distance function
    distance = ((grid_x - cx) ** 2) / (2.0 * sigma_x ** 2) \
             + ((grid_y - cy) ** 2) / (2.0 * sigma_y ** 2) \
             + ((grid_z - cz) ** 2) / (2.0 * sigma_z ** 2)

    heatmap = torch.exp(-distance)
    heatmap = heatmap / heatmap.max()  # Normalize
    return heatmap.numpy()

def create_binary_mask(heatmap, threshold=0.1):
    """
    Creates a binary mask from a Gaussian heatmap by thresholding values.
    """
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    return binary_mask

def process_one_case(input_image, input_mask, heatmap_path, binary_mask_path,
                     alpha=1.0, min_sigma=1.0, threshold=0.1):
    """
    Loads an input image and mask, generates an ellipsoidal Gaussian heatmap
    based on the spread of non-zero mask voxels in each axis, and saves both
    the heatmap and binary mask as NIfTI.
    """
    # Load input image and mask
    image_data, affine = load_nifti(input_image)
    mask_data, _ = load_nifti(input_mask)

    # Ensure mask is binary
    mask_data = (mask_data > 0).astype(np.float32)

    # Compute center of mass
    center = get_center_of_mass(mask_data)

    # Compute axiswise sigmas
    sigma_x, sigma_y, sigma_z = compute_axiswise_sigmas(mask_data, alpha=alpha, min_sigma=min_sigma)
    print(f"Sigmas for {input_mask}: (x={sigma_x:.2f}, y={sigma_y:.2f}, z={sigma_z:.2f})")

    # Generate ellipsoidal Gaussian heatmap
    heatmap = create_ellipsoidal_heatmap(image_data.shape, center, sigma_x, sigma_y, sigma_z)

    # Threshold to get binary mask
    binary_mask = create_binary_mask(heatmap, threshold)

    # Save NIfTI files
    save_nifti(heatmap, heatmap_path, affine)
    save_nifti(binary_mask, binary_mask_path, affine)

def main(train_images_dir, train_labels_dir, out_heatmaps_dir, out_binary_dir,
         alpha=1.0, min_sigma=1.0, threshold=0.1):
    """
    Reads all NIfTI images and masks in the specified training directories,
    creates elliptical Gaussian heatmaps and binary masks for each pair,
    and saves them in separate directories.
    """
    # Create output dirs if they don't exist
    os.makedirs(out_heatmaps_dir, exist_ok=True)
    os.makedirs(out_binary_dir, exist_ok=True)

    # List all files in train_images_dir
    all_images = sorted([
        f for f in os.listdir(train_images_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    # List all files in train_labels_dir
    all_labels = sorted([
        f for f in os.listdir(train_labels_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])

    # Quick check to ensure we have the same number of images and labels
    if len(all_images) != len(all_labels):
        print("Warning: The number of images and labels differs. Make sure they match by name!")

    # Process each image & mask pair
    for img_name, mask_name in zip(all_images, all_labels):
        img_path = os.path.join(train_images_dir, img_name)
        mask_path = os.path.join(train_labels_dir, mask_name)

        heatmap_path = os.path.join(out_heatmaps_dir, f"heatmap_{img_name}")
        binary_path  = os.path.join(out_binary_dir,  f"binary_{img_name}")

        process_one_case(
            input_image=img_path,
            input_mask=mask_path,
            heatmap_path=heatmap_path,
            binary_mask_path=binary_path,
            alpha=alpha,
            min_sigma=min_sigma,
            threshold=threshold
        )

        print(f"Processed {img_name} → {heatmap_path} and {binary_path}")

if __name__ == "__main__":
    # ✅ Default: Original MRI input folders
    train_images_dir = "MRI_data/new_images"  # Original images
    train_labels_dir = "MRI_data/new_labels"  # Original labels

    # ✅ Default: Generated folders for outputs
    out_heatmaps_dir = "MRI_data/new_global_heatmaps"  # Created folder for Gaussian heatmaps
    out_binary_dir = "MRI_data/new_global_masks"  # Created folder for binary masks

    # Hyperparameters
    alpha = 1.0  # Global scaling factor for std dev
    min_sigma = 1.0  # Minimum sigma allowed
    threshold = 0.1  # Threshold for binary mask generation