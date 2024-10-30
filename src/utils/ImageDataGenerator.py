from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_data_generator(augmentation_params=None):
    """
    Return an ImageDataGenerator with predefined or custom augmentation parameters.
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }
    return ImageDataGenerator(**augmentation_params)
