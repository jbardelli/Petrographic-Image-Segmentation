import os
import cv2
import random
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from petrographic_image_utils import image_crop, process_img_patch, rgb_to_2d_label
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from skimage import exposure
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from simple_multi_unet_model import multi_unet_model, jacard_coef

scaler = MinMaxScaler()
patch_size = 400
n_classes = 5  # Number of classes for segmentation
root_directory = 'petrographic_image_dataset/'
image_dataset = []

for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'paralelos':  # Find all 'images' directories
        images = os.listdir(path)  # List of all image names in this subdirectory
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"):
                path_paralelos = path + "/" + image_name
                path_cruzados = root_directory + "cruzados/" + image_name.split('_')[0] + "_cruzados.jpg"
                image_p = cv2.imread(path_paralelos, 1)  # Read parallel nicols image as BGR
                image_c = cv2.imread(path_cruzados, 1)  # Read crossed nicols image as BGR
                image_p = exposure.equalize_hist(image_p)  # Histogram Equalization
                image_c = exposure.equalize_hist(image_c)  # Histogram Equalization
                # Extract patches from each image
                print("Now patchifying image:", path_paralelos)
                patches_p = patchify(image_p, (patch_size, patch_size, 3), step=patch_size)
                print("Now patchifying image:", path_cruzados)
                patches_c = patchify(image_c, (patch_size, patch_size, 3), step=patch_size)
                # print('Patches shape: ', patches_c.shape)

                for j in range(patches_p.shape[0]):
                    for k in range(patches_p.shape[1]):
                        single_patch_p = process_img_patch(patches_p[j, k, :, :], scaler)
                        single_patch_c = process_img_patch(patches_c[j, k, :, :], scaler)
                        stacked = np.dstack((single_patch_p, single_patch_c))
                        image_dataset.append(stacked)

# Now do the same as above for masks
# For this specific dataset we could have added masks to the above code as masks have extension png
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'masks':  # Find all 'masks' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".tiff"):  # Only read png images... (masks in this dataset)
                mask = cv2.imread(path + "/" + mask_name,
                                  1)  # Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # Extract patches from each image
                print("Now patchifying mask:", path + "/" + mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3),
                                        step=patch_size)  # Step=256 for 256 patches means no overlap
                # print(patches_img.shape[0], patches_img.shape[1])
                for j in range(patches_mask.shape[0]):
                    for k in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[j, k, :, :]
                        single_patch_mask = single_patch_mask[
                            0]  # Drop the extra unnecessary dimension that patchify adds.
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)
print("Unique labels in label dataset are: ", np.unique(labels))

# Sanity check, view few mages
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image_dataset[image_number, :, :, 0:3])
plt.subplot(132)
plt.imshow(image_dataset[image_number, :, :, 3:7])
plt.subplot(133)
plt.imshow(labels[image_number])
plt.show()

############################################################################
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

# -------- LOSS -----------
# weights = [0.05, 0.35,  0.10, 0.35, 0.15]                     # Set FIXED class weights for dice_loss
weights = compute_class_weight('balanced', np.unique(np.ravel(labels, order='C')), np.ravel(labels, order='C'))  # Calculated Weights
dice_loss = sm.losses.DiceLoss(class_weights=weights)           # Use Dice loss from Segmentation Models package
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = (dice_loss * 0.5) + (focal_loss * 0.5)           # Losses can be combined

# ------- METRICS ---------
metrics = ['accuracy', jacard_coef]
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


# --- MODEL PARAMETERS ---
model_folder = 'models/'
model_name = 'pis_unet_30SEP2021'
batch_size = 10
learning_rate = 0.01
epochs = 100
loss = dice_loss
optimizer = Adam(lr=learning_rate)

# --- PRINT INFORMATION ---
print("=== MODEL INFORMATION ===")
print("Model Name: ", model_name)
print("Loss Weights: ", weights)
print("Batch Size: ", batch_size)
print("Epochs to train: ", epochs)
print("Learning Rate: ", learning_rate)
print("Image (height, width): ", IMG_HEIGHT, ', ', IMG_WIDTH)
print('Channels: ', IMG_CHANNELS)
print("=========================")

# --- LOAD AND TRAIN MODEL ---
model = get_model()                                                      # Use when creating model from scratch
# model = load_model(model_folder + model_name + '.hdf5', compile=False)   # Use when retraining an existing model
model.compile(optimizer=optimizer, loss=dice_loss, metrics=metrics)
print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())
model.summary()
csv_logger = CSVLogger(model_folder + model_name + '.log', append=True)  # Save loss and metrics to a log file, append if already exists

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    shuffle=False,
                    callbacks=[csv_logger])

model.save(model_folder + model_name + '.hdf5')
############################################################
