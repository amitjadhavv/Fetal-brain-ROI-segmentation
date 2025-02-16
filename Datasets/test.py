import nibabel as nib
import torch
import os

labels_folder = "/home/amit/Downloads/OneDrive_1_2-14-2025/cropped-labels/"
image_folder = "/home/amit/Downloads/OneDrive_1_2-14-2025/cropped-images/"
labels = sorted(os.listdir(labels_folder))
images = sorted(os.listdir(image_folder))
outfile = "/home/amit/PycharmProjects/fetalMRI/MRI_data/corrupted_labels_list.txt"

for i in labels:
    mask = nib.load(labels_folder + i).get_fdata()
    masks = torch.unique(torch.flatten(torch.from_numpy(mask)).to(torch.long))
    print(mask.shape)
    print(masks)
    if len(masks)<5:
        print(labels_folder + i)

# with open(outfile, 'w') as f:
#     for i in range(len(labels)):
#         mask = nib.load(labels_folder+labels[i]).get_fdata()
#         unique = torch.unique(torch.flatten(torch.from_numpy(mask)).to(torch.long))
#         if len(unique) > 5:
#             print(labels[i])
#         if len(unique) != 5:
#             f.write(images[i] + '\n')  # Write each file name to the text file
#             print(f"Wrote: {images[i]}")
#
# with open(outfile, 'r') as f:
#     files_to_delete = [line.strip() for line in f.readlines()]
# for file_name in files_to_delete:
#     file_path = os.path.join(labels_folder, file_name)
#     try:
#         if os.path.isfile(file_path):  # Check if it's a file
#             os.remove(file_path)
#             print(f"Deleted file: {file_path}")
#         else:
#             print(f"File not found or not a file: {file_path}")
#     except Exception as e:
#         print(f"Error deleting file {file_path}: {e}")
#
# print("File deletion process complete.")
# norm_class_weights=[0.24982204 0.25049243 0.24995651 0.24972902]
# norm_class_weights.view(1, -1, *([1] * (loss.ndim - 2)))