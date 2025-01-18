import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the paths to the folders
ground_truth_folder = 'MosaicsUCSD/MosaicsUCSD_gt/test/labels'
gt_folder = 'MosaicsUCSD/test_predictions_gt'
sam_folder = 'MosaicsUCSD/test_predictions_sam_100'
spx_folder = 'MosaicsUCSD/test_predictions_superpixel_100'
mix_folder = 'MosaicsUCSD/test_predictions_mixed_100'

output_dir = 'MosaicsUCSD/eval_mosaics'

input_df = pd.read_csv('../Datasets/MosaicsUCSD/MosaicsUCSD_annotations_100.csv')

# List all image files in each folder
ground_truth_files = sorted(os.listdir(ground_truth_folder))
gt_files = sorted(os.listdir(gt_folder))
sam_files = sorted(os.listdir(sam_folder))
spx_files = sorted(os.listdir(spx_folder))
mix_files = sorted(os.listdir(mix_folder))

# Ensure all folders have the same number of images
assert len(ground_truth_files) == len(sam_files) == len(spx_files) == len(mix_files), \
    f"Mismatch in number of images: Ground Truth: {len(ground_truth_files)}, Method 1: {len(sam_files)}, Method 2: {len(spx_files)}, Method 3: {len(mix_files)}"

def get_color_hsv(index, total_colors):
    hue = (index / total_colors) * 360  # Vary hue from 0 to 360 degrees
    saturation = 1.0 if index % 2 == 0 else 0.7  # Alternate saturation levels
    value = 1.0 if index % 3 == 0 else 0.8  # Alternate value levels
    return hue, saturation, value

def hsv_to_rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    hi = int(h / 60.0) % 6
    f = (h / 60.0) - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)

# get unique labels in 
unique_labels = input_df['Label'].unique()

total_colors = len(unique_labels)
colors_hsv = [get_color_hsv(i, total_colors) for i in range(total_colors)]
colors_rgb = [hsv_to_rgb(*color) for color in colors_hsv]

# Sort colors by HSV values to get similar colors together
sorted_colors_hsv = sorted(colors_hsv, key=lambda x: (x[0], x[1], x[2]))
sorted_colors_rgb = [hsv_to_rgb(*color) for color in sorted_colors_hsv]

# Sort the labels
sorted_labels = sorted(unique_labels)

# Create a dictionary to store the color for each unique label
label_colors = {label: sorted_colors_rgb[i % total_colors] for i, label in enumerate(sorted_labels)}

# Include the class 0 and assign it a specific color (e.g., black)
label_colors[0] = (0, 0, 0)

# Iterate over the images and plot them with a progress bar
for i in tqdm(range(len(ground_truth_files)), desc="Processing images"):
    ground_truth_path = os.path.join(ground_truth_folder, ground_truth_files[i])
    image_name = os.path.basename(ground_truth_path)
    gt_path = os.path.join(gt_folder, gt_files[i])
    sam_path = os.path.join(sam_folder, sam_files[i])
    spx_path = os.path.join(spx_folder, spx_files[i])
    mix_path = os.path.join(mix_folder, mix_files[i])

    # Load the images in grayscale
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    sam = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
    spx = cv2.imread(spx_path, cv2.IMREAD_GRAYSCALE)
    mix = cv2.imread(mix_path, cv2.IMREAD_GRAYSCALE)

    color_mask_ground_truth = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)
    color_mask_gt = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    color_mask_sam = np.zeros((sam.shape[0], sam.shape[1], 3), dtype=np.uint8)
    color_mask_spx = np.zeros((spx.shape[0], spx.shape[1], 3), dtype=np.uint8)
    color_mask_mix = np.zeros((mix.shape[0], mix.shape[1], 3), dtype=np.uint8)

    for label in sorted_labels:
        color = label_colors[label]
        color_mask_ground_truth[ground_truth == label] = color
        color_mask_gt[gt == label] = color
        color_mask_sam[sam == label] = color
        color_mask_spx[spx == label] = color
        color_mask_mix[mix == label] = color

    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    im = axs[0].imshow(color_mask_ground_truth)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')
    axs[1].imshow(color_mask_gt)
    axs[1].set_title("Ground Truth Pred")
    axs[1].axis('off')
    axs[2].imshow(color_mask_sam)
    axs[2].set_title("SAM")
    axs[2].axis('off')
    axs[3].imshow(color_mask_spx)
    axs[3].set_title("Superpixels")
    axs[3].axis('off')
    axs[4].imshow(color_mask_mix)
    axs[4].set_title("Mixed")
    axs[4].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(os.path.join(output_dir, os.path.splitext(image_name)[0] + '.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()