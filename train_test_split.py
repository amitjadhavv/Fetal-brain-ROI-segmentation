import os
import random
import shutil

from configs.config import Config


def split_dataset(
    source_dir_images,
    source_dir_labels,
    train_dir_images,
    train_dir_labels,
    test_dir_images,
    test_dir_labels,
    split_ratio=0.8,
    seed=123
):
    """
    Randomly splits data from `source_dir_images` & `source_dir_labels` into training
    and testing folders, preserving filename alignment (image-mask pairs).
    """

    # Make sure these output directories exist
    os.makedirs(train_dir_images, exist_ok=True)
    os.makedirs(train_dir_labels, exist_ok=True)
    os.makedirs(test_dir_images,  exist_ok=True)
    os.makedirs(test_dir_labels,  exist_ok=True)

    # Get sorted lists of images & labels
    image_files = sorted([
        f for f in os.listdir(source_dir_images)
        if os.path.isfile(os.path.join(source_dir_images, f))
    ])
    label_files = sorted([
        f for f in os.listdir(source_dir_labels)
        if os.path.isfile(os.path.join(source_dir_labels, f))
    ])

    # Check that the number of images matches the number of labels
    assert len(image_files) == len(label_files), "Images and labels count mismatch."

    # Pair them up so we always move the matching image & label
    file_pairs = list(zip(image_files, label_files))

    # Shuffle
    random.seed(seed)
    random.shuffle(file_pairs)

    # Compute split index
    split_index = int(len(file_pairs) * split_ratio)

    # Separate into train/test
    train_pairs = file_pairs[:split_index]
    test_pairs  = file_pairs[split_index:]

    # Move training pairs
    for img_name, lbl_name in train_pairs:
        src_img = os.path.join(source_dir_images, img_name)
        src_lbl = os.path.join(source_dir_labels, lbl_name)
        dst_img = os.path.join(train_dir_images, img_name)
        dst_lbl = os.path.join(train_dir_labels, lbl_name)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    # Move testing pairs
    for img_name, lbl_name in test_pairs:
        src_img = os.path.join(source_dir_images, img_name)
        src_lbl = os.path.join(source_dir_labels, lbl_name)
        dst_img = os.path.join(test_dir_images, img_name)
        dst_lbl = os.path.join(test_dir_labels, lbl_name)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    print(f"Training set size: {len(train_pairs)}")
    print(f"Testing set size : {len(test_pairs)}")


if __name__ == "__main__":
    # Example usage:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'MRI_data'))
    print(path)
    source_imgs = os.path.join(path, "cropped-images")
    source_lbls = os.path.join(path, "cropped-labels")
    train_imgs  = os.path.join(path, "training_data")
    train_lbls  = os.path.join(path, "training_labels")
    test_imgs   = os.path.join(path, "testing_data")
    test_lbls   = os.path.join(path, "testing_labels")

    split_dataset(
        source_dir_images=source_imgs,
        source_dir_labels=source_lbls,
        train_dir_images=train_imgs,
        train_dir_labels=train_lbls,
        test_dir_images=test_imgs,
        test_dir_labels=test_lbls,
        split_ratio=0.8,  # 80% train, 20% test
        seed=42
    )
