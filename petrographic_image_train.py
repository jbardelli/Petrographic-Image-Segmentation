import os
import cv2
import random
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from petrographic_image_utils import image_crop
from petrographic_image_utils import process_img_patch
# import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from simple_multi_unet_model import multi_unet_model, jacard_coef

scaler = MinMaxScaler()
patch_size = 256
SIZE_X = 256
SIZE_Y = 256
n_classes = 4  # Number of classes for segmentation
root_directory = 'petrographic_image_dataset/'
image_dataset_p = []
image_dataset_c = []

for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'paralelos':             # Find all 'images' directories
        images = os.listdir(path)        # List of all image names in this subdirectory
        for i, image_name in enumerate(images):
            if image_name.endswith(".tiff"):
                print(image_name)
                path_cruzados = root_directory + "cruzados/" + image_name.split('_')[0] + "_cruzados.tiff"
                print(path_cruzados)

                image_p = cv2.imread(path + "/" + image_name, 1)        # Read parallel nicols image as BGR
                image_c = cv2.imread(path_cruzados, 1)                  # Read crossed nicols image as BGR

                # Crop images to to the nearest multiple of patch_size
                image_p = image_crop(image_p, patch_size)
                image_c = image_crop(image_c, patch_size)

                # Extract patches from each image
                print("Now patchifying image:", path + "/" + image_name)
                patches_p = patchify(image_p, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
                patches_c = patchify(image_c, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap

                for j in range(patches_p.shape[0]):
                    for k in range(patches_p.shape[1]):
                        single_patch_p = process_img_patch(patches_p[j, k, :, :], scaler)
                        single_patch_c = process_img_patch(patches_c[j, k, :, :], scaler)
                        # stacked = np.dstack((single_patch_p, single_patch_c))                 # Stack the two images into a 6 channel array
                        image_dataset_p.append(single_patch_p)
                        image_dataset_c.append(single_patch_c)

# Now do the same as above for masks
# For this specific dataset we could have added masks to the above code as masks have extension png
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'masks':  # Find all 'masks' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".tiff"):  # Only read png images... (masks in this dataset)
                mask = cv2.imread(path + "/" + mask_name, 1)  # Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size

                mask = image_crop(mask, patch_size)

                # Extract patches from each image
                print("Now patchifying mask:", path + "/" + mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap
                # print(patches_img.shape[0], patches_img.shape[1])
                for j in range(patches_mask.shape[0]):
                    for k in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[j, k, :, :]
                        single_patch_mask = single_patch_mask[0]  # Drop the extra unnecessary dimension that patchify adds.
                        mask_dataset.append(single_patch_mask)

image_dataset_p = np.array(image_dataset_p)
image_dataset_c = np.array(image_dataset_c)
mask_dataset = np.array(mask_dataset)


# image_number = random.randint(0, len(image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
# plt.subplot(122)
# plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
# plt.show()

def rgb_to_2d_label(label_):
    """
    Supply our label masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label_.shape, dtype=np.uint8)
    label_seg[np.all(label_ == 0, axis=-1)] = 0
    label_seg[np.all(label_ == 1, axis=-1)] = 1
    label_seg[np.all(label_ == 2, axis=-1)] = 2
    label_seg[np.all(label_ == 3, axis=-1)] = 3
    label_seg[np.all(label_ == 4, axis=-1)] = 4
    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels
    return label_seg


labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)

print("Unique labels in label dataset are: ", np.unique(labels))

# Another Sanity check, view few mages

image_number = random.randint(0, len(image_dataset_p))
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image_dataset_p[image_number])
plt.subplot(132)
plt.imshow(image_dataset_c[image_number])
plt.subplot(133)
plt.imshow(labels[image_number])
plt.show()

############################################################################
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

#######################################
# Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss
# weights = compute_class_weight('balanced', np.unique(np.ravel(labels, order='C')),
#                                np.ravel(labels, order='C'))
# print(weights)

weights = [0.4, 0.6]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
metrics = ['accuracy', jacard_coef]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

history1 = model.fit(X_train, y_train,
                     batch_size=16,
                     verbose=1,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     shuffle=False)

# Minmaxscaler
# With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
# With focal loss only, after 100 epochs val jacard is: 0.62  (Mean IoU: 0.6)
# With dice loss only, after 100 epochs val jacard is: 0.74 (Reached 0.7 in 40 epochs)
# With dice + 5 focal, after 100 epochs val jacard is: 0.711 (Mean IoU: 0.611)
# With dice + 1 focal, after 100 epochs val jacard is: 0.75 (Mean IoU: 0.62)
# Using categorical crossentropy as loss: 0.71

# With calculated weights in Dice loss.
# With dice loss only, after 100 epochs val jacard is: 0.672 (0.52 iou)


# Standardscaler
# Using categorical crossentropy as loss: 0.677

model.save('models/satellite_standard_unet_100epochs_7May2021.hdf5')
############################################################
