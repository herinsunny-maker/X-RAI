
import cv2
import numpy as np
import tensorflow as tf

def apply_clahe(image):
    """
    Applies CLAHE to a single image (H, W, 3).
    Ensures output is always (H, W, 3).
    """
    try:
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        # Ensure uint8
        img_uint8 = image.astype(np.uint8)
        
        # Get dimensions
        h, w = img_uint8.shape[:2]
        
        # Convert to gray
        if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        elif len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
            gray = img_uint8[:, :, 0]
        else:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if len(img_uint8.shape) == 3 else img_uint8
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        
        # Merge back to 3 channels (RGB)
        res = cv2.merge([cl1, cl1, cl1])
        
        # Final safety resize to ensure exactly H, W
        if res.shape[:2] != (h, w):
            res = cv2.resize(res, (w, h))
            
        return res.astype(np.float32)
    except Exception as e:
        # If anything fails, return original image as float32 to avoid crashing the dataset
        return image.astype(np.float32)

def tf_apply_clahe(image, label):
    """
    TensorFlow wrapper for apply_clahe.
    """
    # Wrap the python function. specify the output type.
    image_processed = tf.py_function(apply_clahe, [image], tf.float32)
    # CRITICAL: Re-assert the shape so TF knows the dimensions
    image_processed.set_shape(image.shape)
    return image_processed, label

def get_augmentation_model():
    """
    Returns a Sequential model for on-the-fly data augmentation using Keras layers.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.1),
    ])
