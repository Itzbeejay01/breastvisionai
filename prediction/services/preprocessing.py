"""
Image preprocessing for the 3 PSO-selected CNN models.

Supports two modes:
  - "raw":         load image, resize, apply model-specific preprocessing
  - "processed":   load image, resize, skip model-specific preprocessing

Model-specific preprocessing functions are adapted from:
  model_training/generate_gradcam.py (preprocess_image)
  model_training/ensembles.py (preprocess_input functions)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

IMAGE_SIZE = (224, 224)


def preprocess_for_model(image_array, model_name):
    """
    Apply model-specific preprocessing to an image array.

    EfficientNet: keep 0–255 range (internal rescaling in modern tf.keras)
    ResNet:       tf.keras.applications.resnet.preprocess_input (BGR, zero-centered)
    VGG16:        tf.keras.applications.vgg16.preprocess_input (BGR, zero-centered)
    """
    image_array = image_array.astype(np.float32)

    if model_name == "EfficientNet":
        pass

    elif model_name == "ResNet":
        image_array = resnet_preprocess(image_array)

    elif model_name == "VGG16":
        image_array = vgg16_preprocess(image_array)

    else:
        image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


def load_image_as_array(image_source):
    """
    Load an image from a file path or an in-memory file (UploadedFile)
    and return it as a (224, 224, 3) numpy array in 0–255 range.
    """
    if hasattr(image_source, "read"):
        # Django UploadedFile or BytesIO
        image_source.seek(0)
        img = tf.keras.utils.load_img(image_source, target_size=IMAGE_SIZE)
    else:
        img = tf.keras.utils.load_img(image_source, target_size=IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(img)
    return image_array


def preprocess_image(image_source, model_name, image_type="raw"):
    """
    Full preprocessing pipeline for a single image.

    Args:
        image_source: file path or Django UploadedFile
        model_name:   one of "EfficientNet", "ResNet", "VGG16"
        image_type:   "raw" (apply model-specific preprocessing) or
                      "processed" (skip preprocessing, return normalized array)

    Returns:
        tuple: (batched_array shape (1, 224, 224, 3), original_array (224, 224, 3))
    """
    original_array = load_image_as_array(image_source)

    if image_type == "processed":
        processed = original_array.astype(np.float32)
        processed = processed / 255.0
        batched = np.expand_dims(processed, axis=0)
    else:
        batched = preprocess_for_model(original_array, model_name)

    return batched, original_array


def preprocess_image_for_all_models(image_source, image_type="raw"):
    """
    Preprocess an image for ALL 3 PSO-selected models.

    Used when running the full ensemble pipeline — each model may need
    different preprocessing.

    Args:
        image_source: file path or Django UploadedFile
        image_type:   "raw" or "processed"

    Returns:
        dict: {model_name: (batched_array, original_array)}
    """
    # Load original once (shared across models)
    original_array = load_image_as_array(image_source)

    results = {}
    for model_name in ["EfficientNet", "ResNet", "VGG16"]:
        if image_type == "processed":
            processed = original_array.astype(np.float32) / 255.0
            batched = np.expand_dims(processed, axis=0)
        else:
            batched = preprocess_for_model(original_array.copy(), model_name)

        results[model_name] = (batched, original_array)

    return results
