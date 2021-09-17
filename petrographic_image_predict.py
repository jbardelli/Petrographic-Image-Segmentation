import numpy as np
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from petrographic_image_utils import rgb_to_2d_label

scaler = MinMaxScaler()
patch_size = 400
n_classes = 5  # Number of classes for segmentation
image_p_path = 'petrographic_image_dataset/paralelos/9a_paralelos.jpg'
image_c_path = 'petrographic_image_dataset/cruzados/9a_cruzados.jpg'
mask_path = 'petrographic_image_dataset/masks/9a_mascara.ome.tiff'

image_p = cv2.imread(image_p_path, 1)           # Read parallel nicols image as BGR
image_c = cv2.imread(image_c_path, 1)           # Read crossed nicols image as BGR
mask = cv2.imread(mask_path, 1)    # Read mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
mask = rgb_to_2d_label(mask)

# Load model
model = load_model("models/petrography_standard_unet_100epochs.hdf5", compile=False)

###################################################################################
#Predict using smooth blending
input_img_p = scaler.fit_transform(image_p.reshape(-1, image_p.shape[-1])).reshape(image_p.shape)
input_img_c = scaler.fit_transform(image_c.reshape(-1, image_c.shape[-1])).reshape(image_c.shape)
input_img = np.dstack((input_img_p, input_img_c))
# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv)))
)

final_prediction = np.argmax(predictions_smooth, axis=2)

########################
#Plot and save results
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Image Plain Polarized Light')
plt.imshow(image_p)
plt.subplot(222)
plt.title('Image Cross Polarized Light')
plt.imshow(image_c)
plt.subplot(223)
plt.title('Testing Label')
plt.imshow(mask)
plt.subplot(224)
plt.title('Prediction')
plt.imshow(final_prediction)
plt.show()
#############################