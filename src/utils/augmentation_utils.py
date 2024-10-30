import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
import logging

def augment_image(image_path, output_dir, augmentation_params, num_augmented=20):
    """
    Apply data augmentation techniques to increase dataset diversity.
    """
    try:
        image = load_img(image_path)
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)
        datagen = ImageDataGenerator(**augmentation_params)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= num_augmented:
                break
        logging.info(f"Generated {num_augmented} augmented images for {image_path}")
    except Exception as e:
        logging.error(f"Error augmenting image {image_path}: {e}")
