import torch
import nibabel as nib
import numpy as np
import argparse
import os


def load_nifti(file_path):
    """
    Loads a NIfTI image and returns its data as a NumPy array along with its affine matrix.

    Args:
        file_path (str): Path to the NIfTI file.

    Returns:
        tuple: (numpy.ndarray, numpy.ndarray) Image data and affine transformation matrix.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine


def save_nifti(data, filename, affine):
    """
    Saves a 3D numpy array as a NIfTI file.

    Args:
        data (numpy.ndarray): 3D array containing the image data.
        filename (str): Output file path.
        affine (numpy.ndarray): Affine transformation matrix.
    """
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, filename)


def get_center_of_mass(mask):
    """
    Computes the center of mass of a given binary mask.

    Args:
        mask (numpy.ndarray): 3D binary mask.

    Returns:
        tuple: (x, y, z) coordinates of the center of mass.
    """
    indices = np.argwhere(mask > 0)
    if len(indices) == 0:
        raise ValueError("No landmarks found in the mask!")

    center = np.mean(indices, axis=0)  # Compute mean position
    return tuple(int(c) for c in center)


def create_global_gaussian_heatmap(image_shape, center, sigma=15):
    """
    Generates a single Gaussian heatmap centered at the brain region.

    Args:
        image_shape (tuple): Shape of the image (D, H, W).
        center (tuple): (x, y, z) coordinates of the Gaussian center.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        numpy.ndarray: A 3D heatmap mask of the same shape as the image.
    """
    mask = torch.zeros(image_shape, dtype=torch.float32)
    cx, cy, cz = center

    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(image_shape[0]),
        torch.arange(image_shape[1]),
        torch.arange(image_shape[2]),
        indexing='ij'
    )

    distance = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 + (grid_z - cz) ** 2
    mask = torch.exp(-distance / (2 * sigma ** 2))

    mask = mask / mask.max()  # Normalize the heatmap
    return mask.numpy()


def create_binary_mask(heatmap, threshold=0.1):
    """
    Creates a binary mask from a Gaussian heatmap by thresholding values.

    Args:
        heatmap (numpy.ndarray): The generated Gaussian heatmap.
        threshold (float): Minimum value for the mask (default 10% intensity).

    Returns:
        numpy.ndarray: A binary mask where values >= threshold are 1.
    """
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    return binary_mask


def main(input_image, input_mask, output_heatmap, output_mask, sigma=15, threshold=0.1):
    """
    Loads an input image and mask, generates a single Gaussian heatmap centered at the detected brain region,
    and saves both the heatmap and its binary mask.

    Args:
        input_image (str): Path to the input MRI image file (.nii or .nii.gz).
        input_mask (str): Path to the mask file (.nii or .nii.gz).
        output_heatmap (str): Output filename for the Gaussian heatmap.
        output_mask (str): Output filename for the binary mask.
        sigma (float): Standard deviation for Gaussian heatmap generation.
        threshold (float): Threshold to create the binary mask.
    """
    # Load input image and mask
    image_data, affine = load_nifti(input_image)
    mask_data, _ = load_nifti(input_mask)

    # Ensure mask is binary
    mask_data = (mask_data > 0).astype(np.float32)

    # Compute center of mass
    center = get_center_of_mass(mask_data)

    # Generate Gaussian heatmap centered at this region
    heatmap_mask = create_global_gaussian_heatmap(image_data.shape, center, sigma)

    # Generate binary mask from heatmap
    binary_mask = create_binary_mask(heatmap_mask, threshold)

    # Save heatmap and binary mask as NIfTI files
    save_nifti(heatmap_mask, output_heatmap, affine)
    save_nifti(binary_mask, output_mask, affine)

    print(f"Gaussian heatmap saved as {output_heatmap}")
    print(f"Binary mask saved as {output_mask}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a global Gaussian heatmap mask centered at detected brain landmarks and its binary mask.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input NIfTI image (.nii/.nii.gz).")
    parser.add_argument("--input_mask", type=str, required=True, help="Path to the input NIfTI mask (.nii/.nii.gz).")
    parser.add_argument("--output_heatmap", type=str, default="roi_heatmap.nii.gz",
                        help="Output filename for the Gaussian heatmap.")
    parser.add_argument("--output_mask", type=str, default="roi_mask.nii.gz",
                        help="Output filename for the binary mask.")
    parser.add_argument("--sigma", type=float, default=15, help="Standard deviation for Gaussian kernel.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for binary mask generation.")

    args = parser.parse_args()

    main(args.input_image, args.input_mask, args.output_heatmap, args.output_mask, args.sigma, args.threshold)
