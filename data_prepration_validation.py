
import os
import nibabel as nib

def read_original_images():
    """
    Read all original images from both the original_images folder and 
    any original_images folders inside the MRI_data subdirectory.
    """
    mri_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'MRI_data'))
    # Paths for original_images in root and original_images inside MRI_data
    original_images_dir = os.path.join(mri_data_dir, 'original-images')
    original_labels_dir = os.path.join(mri_data_dir, 'original-labels')
    new_images_dir = os.path.join(mri_data_dir, 'new_images')
    new_labels_dir = os.path.join(mri_data_dir, 'new_labels')

    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)

    matching_pairs = []
    label_files = []
    matching_pairs_csv_path = os.path.join(mri_data_dir, 'size_mismatched_pairs.csv')
    with open(matching_pairs_csv_path, 'w') as csv_file:
        csv_file.write('counter,image,label,type\n')  # Write CSV header

        if os.path.exists(original_labels_dir):
            for root, _, files in os.walk(original_labels_dir):
                label_files.extend(
                    [os.path.join(root, f) for f in files if f.endswith('.nii.gz')]
                )

        counter = 1
        match_count = 0
        size_mismatch_count = 0
        if os.path.exists(original_images_dir):
            for root, _, files in os.walk(original_images_dir):
                for f in files:
                    if f.endswith('.nii') or f.endswith('.nii.gz'):
                        original_file = os.path.join(root, f)

                    img_shape = nib.load(original_file).shape
                    original_filename = os.path.basename(original_file)[:30]

                    for label_file in label_files:
                        label_filename = os.path.basename(label_file)[:30]
                        if original_filename == label_filename:
                            match_count += 1
                            label_shape = nib.load(label_file).shape

                            if img_shape == label_shape:
                                # Copy matched pairs to new directories
                                new_image_name = f"image_{counter:03d}.nii"
                                new_label_name = f"label_{counter:03d}.nii"
                                new_image_path = os.path.join(new_images_dir, new_image_name)
                                new_label_path = os.path.join(new_labels_dir, new_label_name)

                                nib.save(nib.load(original_file), new_image_path)
                                nib.save(nib.load(label_file), new_label_path)

                                # Write matched files into the CSV
                                csv_file.write(
                                    f"{counter},{os.path.basename(original_file)},{os.path.basename(label_file)},match\n")
                                counter += 1
                            else:
                                size_mismatch_count += 1
                                # Write mismatched files into the CSV
                                csv_file.write(
                                    f"{counter},{os.path.basename(original_file)},{os.path.basename(label_file)},mismatch\n")

    print(f"Total matched files: {match_count}")
    print(f"Total size mismatches: {size_mismatch_count}")


def main():
    """
    Main function to call the read_original_images function.
    """
    read_original_images()


if __name__ == "__main__":
    main()
