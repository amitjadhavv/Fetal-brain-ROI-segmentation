import torch
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label

# Define input and output file paths here
INPUT_IMAGE = "/home/amit/PycharmProjects/fetalMRI2/MRI_data/training_data/09-15-10-gadgetron-fetal-brain-localisation-img_initial_seg_brain.nii.gz"  # Change this to your actual image file
INPUT_MASK = "/home/amit/PycharmProjects/fetalMRI2/MRI_data/training_labels/09-15-10-gadgetron-fetal-brain-localisation-img__labels_cropped.nii.gz"  # Change this to your actual mask file
OUTPUT_4D_FILENAME = "./heatmaps/heatmap_4d.nii.gz"  # 4D output file path
LANDMARK_TYPES = [1, 3, 4, 6]  # Only these landmark labels are processed
ALPHA = 1.0  # Initial scaling factor for adaptive sigma
SIGMA_SCALING_FACTOR = 0.60  # Additional scaling factor to reduce all sigma values


def load_nifti(file_path):
    """Loads a NIfTI image and returns its data as a NumPy array along with its affine matrix."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine


def save_nifti(data, filename, affine):
    """Saves a 4D NIfTI file."""
    nii_img = nib.Nifti1Image(data, affine)
    nib.save(nii_img, filename)


def get_landmarks_by_type(mask, landmark_types):
    """
    Extracts landmarks for specific landmark types and separates multiple clusters for 1 & 4.

    Args:
        mask (numpy.ndarray): 3D mask array with different landmark types.
        landmark_types (list): List of landmark labels to process.

    Returns:
        dict: Dictionary where keys are landmark types and values are lists of landmark clusters.
    """
    landmarks_dict = {}
    for landmark_type in landmark_types:
        coords = np.argwhere(mask == landmark_type)
        if len(coords) > 0:
            if landmark_type in [1, 4]:  # Split into two separate clusters for IDs 1 & 4
                labeled_array, num_clusters = label(mask == landmark_type)
                clusters = [np.argwhere(labeled_array == i + 1) for i in range(num_clusters)]
                landmarks_dict[landmark_type] = clusters  # Store multiple clusters
            else:
                landmarks_dict[landmark_type] = [coords]  # Store as a single cluster

    return landmarks_dict


def compute_adaptive_sigma(landmarks, alpha=1.0, half_sigma=False, scaling_factor=0.75):
    """
    Computes an adaptive sigma value based on the spatial spread of landmarks.

    Args:
        landmarks (list of tuples): List of (x, y, z) coordinates of landmarks.
        alpha (float): Initial scaling factor to adjust sigma.
        half_sigma (bool): Whether to halve the sigma value (for IDs 1 and 4).
        scaling_factor (float): Factor to further reduce the computed sigma.

    Returns:
        float: Scaled adaptive sigma value.
    """
    if len(landmarks) == 1:
        sigma = 3.0
    else:
        landmarks = np.array(landmarks)
        centroid = np.mean(landmarks, axis=0)
        variance = np.mean(np.sum((landmarks - centroid) ** 2, axis=1))  # Average squared distance
        sigma = alpha * np.sqrt(variance)

    sigma *= scaling_factor  # Apply additional reduction
    return sigma / 2 if half_sigma else sigma  # Halve sigma if needed


def create_gaussian_heatmap(image_shape, landmarks, sigma):
    """Generates a 3D Gaussian heatmap mask for a given image shape and landmark locations."""
    mask = torch.zeros(image_shape, dtype=torch.float32)

    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(image_shape[0]),
        torch.arange(image_shape[1]),
        torch.arange(image_shape[2]),
        indexing='ij'
    )

    for landmark in landmarks:
        x, y, z = landmark
        distance = (grid_x - x) ** 2 + (grid_y - y) ** 2 + (grid_z - z) ** 2
        mask += torch.exp(-distance / (2 * sigma ** 2))

    return mask / mask.max()  # Normalize the heatmap


def main():
    """Main function to generate and save a 4D Gaussian heatmap mask."""
    os.makedirs(os.path.dirname(OUTPUT_4D_FILENAME), exist_ok=True)  # Ensure output directory exists

    # Load input image and mask
    image_data, affine = load_nifti(INPUT_IMAGE)
    mask_data, _ = load_nifti(INPUT_MASK)

    # Get landmarks for selected landmark types, separating multiple clusters for 1 & 4
    landmarks_dict = get_landmarks_by_type(mask_data, LANDMARK_TYPES)

    # Initialize 4D heatmap storage
    heatmap_4d = np.zeros((len(LANDMARK_TYPES), *image_data.shape), dtype=np.float32)

    # Generate and store a heatmap for each landmark type in a separate channel
    for i, landmark_type in enumerate(LANDMARK_TYPES):
        heatmap_mask = torch.zeros(image_data.shape, dtype=torch.float32)  # Initialize combined mask

        if landmark_type in landmarks_dict:
            # Process multiple clusters separately but merge into the same channel
            for cluster in landmarks_dict[landmark_type]:
                half_sigma = landmark_type in [1, 4]  # Halve sigma for IDs 1 and 4
                adaptive_sigma = compute_adaptive_sigma(cluster, ALPHA, half_sigma, SIGMA_SCALING_FACTOR)
                cluster_heatmap = create_gaussian_heatmap(image_data.shape, cluster, adaptive_sigma)
                heatmap_mask += cluster_heatmap  # Merge both heatmaps into one

        heatmap_4d[i] = heatmap_mask.numpy()  # Store in 4D array

    # Save the final 4D heatmap NIfTI file
    save_nifti(heatmap_4d, OUTPUT_4D_FILENAME, affine)

    print(f"4D heatmap NIfTI file saved as {OUTPUT_4D_FILENAME}")


if __name__ == "__main__":
    main()


