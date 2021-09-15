from PIL import Image
import numpy as np


def image_crop(image, patch_size):
    SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
    SIZE_Y = (image.shape[0] // patch_size) * patch_size  # Nearest size divisible by our patch size
    print(SIZE_Y, SIZE_X)
    image = Image.fromarray(image)
    image = image.crop((0, 0, SIZE_X, SIZE_Y))  # Crop from top left corner
    return np.array(image)


def process_img_patch(single_patch_img, scaler):
    # Use minmaxscaler instead of just dividing by 255.
    single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
    # single_patch_img = (single_patch_img.astype('float32')) / 255.
    single_patch_img = single_patch_img[0]  # Drop the extra unnecessary dimension that patchify adds.
    return single_patch_img


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
    label_seg[np.all(label_ == 5, axis=-1)] = 5
    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels
    return label_seg
