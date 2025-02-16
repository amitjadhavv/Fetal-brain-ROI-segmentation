import os
import torch
class Config:
    #path to the data directory
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MRI_data'))

    @classmethod
    def get_image_paths(cls):
        images = sorted(os.listdir(os.path.join(cls.dir_path, "cropped-images")))
        return [os.path.join(cls.dir_path, "cropped-images", i) for i in images]

    @classmethod
    def get_mask_paths(cls):
        masks = sorted(os.listdir(os.path.join(cls.dir_path, "cropped-labels")))
        return [os.path.join(cls.dir_path, "cropped-labels", i) for i in masks]

# Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 5
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
