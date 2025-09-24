# sample_inference.py

import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from models.AttentionVNet import AttentionVNet
from configs.config import Config
from monai.transforms import Activations, AsDiscrete
from Datasets.dataset import robust_normalize
import time
from scipy import ndimage

def keep_largest_connected_component(mask):
    """
    Keep only the largest connected component from a binary 3D mask.
    Handles both NumPy arrays and PyTorch tensors.
    Returns a torch tensor (int).
    """
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)

    # Connected component labeling
    labeled, num_features = ndimage.label(mask_np)
    if num_features < 2:  # nothing to clean
        cleaned = mask_np
    else:
        sizes = ndimage.sum(mask_np, labeled, range(1, num_features + 1))
        largest_cc = (sizes.argmax() + 1)
        cleaned = (labeled == largest_cc).astype(np.uint8)

    # Back to torch tensor, shape [1,1,D,H,W]
    cleaned_t = torch.tensor(cleaned, dtype=torch.int)
    return cleaned_t

def load_model(model_path, num_classes):
    """
    Loads the trained VNet model from a .pth checkpoint.
    """
    model = AttentionVNet(num_classes=num_classes).to(Config.DEVICE)

    # Load state dict
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    # If trained with DataParallel, remove "module." prefix
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): val for key, val in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(64, 64, 64)):
    """
    Load and preprocess a single volume from a NIfTI file:
      - Normalizes the intensities to [0,1].
      - Resizes to target_size via trilinear interpolation.
    Returns: a PyTorch Tensor of shape (1, 1, D, H, W).
    """
    # Load the image
    nib_img = nib.load(image_path)
    image_data = nib_img.get_fdata()
    image_data = robust_normalize(image_data)

    # Add channel dimension => (1, D, H, W)
    image_data = np.expand_dims(image_data, axis=0)

    # Convert to torch tensor
    image_tensor = torch.tensor(image_data, dtype=torch.float32)

    # Resize to (1, 1, 64, 64, 64)
    # NOTE: unsqueeze(0) => (1, 1, D, H, W) to feed F.interpolate
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = F.interpolate(
        image_tensor,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )
    return image_tensor  # shape: (1, 1, 64, 64, 64)


def postprocess_mask(pred_mask, original_shape, inverse_mapping=None, threshold=0.1):
    """
    - pred_mask: PyTorch tensor of shape (1, D, H, W) or (D, H, W) with label indices.
    - Resizes it back to the original shape of the input image using nearest interpolation.
    - Applies inverse label mapping if provided.
    - Saves a binary heatmap mask based on a threshold, if binary_mask_path is provided.
    - Returns numpy array with final segmentation.
    """

    post_pred = Activations(sigmoid=True)
    post_label = AsDiscrete(threshold=0.5)
    pred_mask = post_pred(pred_mask)
    pred_mask = post_label(pred_mask)
    pred_mask = keep_largest_connected_component(pred_mask.cpu().numpy()) #keep only 1 mask cluster and remove smaller clusters if there are any

    # Ensure shape is (1, 1, D, H, W) for interpolation
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif len(pred_mask.shape) == 4:
        pred_mask = pred_mask.unsqueeze(0)

    # Resize with nearest neighbor to match original shape
    resized_mask = F.interpolate(
        pred_mask.float(),
        size=original_shape,
        mode='nearest'
    )
    # Remove batch and channel dims => (D, H, W)
    resized_mask = resized_mask.squeeze(0).squeeze(0)
    final_mask = resized_mask.cpu().numpy().astype(np.int16)

    return final_mask #binary_mask

def save_nifti(volume, affine, save_path):
    """
    Save a 3D volume (numpy array) to NIfTI format using the given affine.
    """
    nib_obj = nib.Nifti1Image(volume, affine)
    nib.save(nib_obj, save_path)


def run_inference_single_image(image_path, model_path, output_path):
    """
    End-to-end function to:
      1) load model,
      2) preprocess single NIfTI image,
      3) predict mask,
      4) post-process mask (resize + inverse mapping),
      5) save mask file to output_path
    """
    # Load model
    model = load_model(model_path, num_classes=Config.NUM_CLASSES)

    # Load original image
    nib_img = nib.load(image_path)
    original_shape = nib_img.shape  # (D, H, W)
    affine = nib_img.affine

    # Preprocess
    input_tensor = preprocess_image(image_path, target_size=(64, 64, 64))
    input_tensor = input_tensor.to(Config.DEVICE)
    #print(input_tensor)
 
    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)  # shape: (1, num_classes, 64, 64, 64)

    # Post-process (resize + map labels back)
    final_mask = postprocess_mask(
        logits,
        original_shape
    )
    #Print unique values of the final mask
    print(f"Unique values in final mask: {np.unique(final_mask)}")
    #Save mask
    save_nifti(final_mask, affine, output_path)
    print(f"Saved segmentation mask to: {output_path}")


if __name__ == "__main__":
    # Example usage:
    # Adjust these paths as needed
    sample_image_path = "/home/amit/PycharmProjects/fetalMRI2/MRI_data/new_images/" #input folder lcoation
    image = "image_335.nii" # input file name
    model_path = "AV_net_noDNN_model_roi_best.pth" # model file name
    # The path to your trained model
    output_mask_path = "/home/amit/PycharmProjects/fetalMRI2/MRI_data/output/" # output directory to save the predicted mask
    output = "predicted_"+image # output file name
    start_time = time.time()
    # Make sure the model_path and sample_image_path exist
    if not os.path.exists(sample_image_path):
        raise FileNotFoundError(f"Sample image not found: {sample_image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    run_inference_single_image(sample_image_path+image, model_path,output_mask_path+output)
    end_time = time.time()  # End time tracking
    inference_time = end_time - start_time
    print(str(inference_time) + " seconds") # prints the time taken to process the file.