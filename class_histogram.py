import os
import cv2
import numpy as np
from petrographic_image_utils import rgb_to_2d_label

root_directory = 'petrographic_image_dataset/'

mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'masks':  # Find all 'masks' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".tiff"):  # Only read png images... (masks in this dataset)
                mask = cv2.imread(path + "/" + mask_name, 1)  # Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask_dataset.append(mask)

mask_dataset = np.array(mask_dataset)

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
print("Unique labels in label dataset are: ", np.unique(labels))

histogram, edges = np.histogram(labels, 5)
total_pixels = np.sum(histogram)
percentages = np.around(histogram / total_pixels * 100, decimals=1)
print('Occurences per class: ', histogram)
print('Total pixels: ', total_pixels)
print('Occurence percentages: ', percentages)
