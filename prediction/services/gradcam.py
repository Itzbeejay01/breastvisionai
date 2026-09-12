"""
Grad-CAM heatmap generation for single images.

Adapted from model_training/generate_gradcam.py to work on single images
(either raw or already preprocessed) without requiring batch generation.

Generates a heatmap overlay showing where the CNN model "looks" when
making its prediction, and returns it as a base64-encoded PNG string
for embedding in API responses.
"""

import io
import base64
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from prediction.services.model_registry import PSOModelRegistry

IMAGE_SIZE = (224, 224)


def find_cnn_base_model(model):
    """
    Finds the nested CNN base model inside the outer Sequential model.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def find_last_conv_layer(model):
    """
    Finds the last convolutional layer in the model.
    Works with EfficientNet, ResNet, VGG16.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (
            tf.keras.layers.Conv2D,
            tf.keras.layers.SeparableConv2D,
            tf.keras.layers.DepthwiseConv2D
        )):
            return model, layer

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.SeparableConv2D,
                    tf.keras.layers.DepthwiseConv2D
                )):
                    return layer, nested_layer

    raise ValueError(f"Could not find a convolutional layer in model: {model.name}")


def build_gradcam_model(model, cnn_base_model, last_conv_layer):
    """
    Builds a Grad-CAM model that outputs both the last conv layer
    activations and the final prediction.
    """
    if cnn_base_model is not None:
        base_input = cnn_base_model.input
        conv_output = last_conv_layer.output
        x = cnn_base_model.output
        base_index = model.layers.index(cnn_base_model)
        for layer in model.layers[base_index + 1:]:
            x = layer(x)
        predictions = x
        grad_model = tf.keras.models.Model(
            inputs=base_input,
            outputs=[conv_output, predictions]
        )
        return grad_model

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    return grad_model


def make_gradcam_heatmap(image, grad_model):
    """
    Generate Grad-CAM heatmap from a preprocessed image batch.
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)

        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]

    gradients = tape.gradient(class_channel, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients are None. The selected convolutional layer "
            "is not connected to the prediction."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_gradients
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)

    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap):
    """
    Overlay a heatmap on the original image and return as a combined
    RGB array.

    Args:
        original_image: (H, W, 3) array in 0–255 range
        heatmap: (H, W) normalized 0–1 array

    Returns:
        (H, W, 3) uint8 array with heatmap overlay
    """
    heatmap_resized = tf.image.resize(
        np.expand_dims(heatmap, axis=-1),
        IMAGE_SIZE,
        method="bilinear"
    ).numpy().squeeze()

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = tf.keras.utils.array_to_img(
        tf.stack([heatmap_uint8] * 3, axis=-1),
        data_format="channels_last"
    )
    # Use matplotlib colormap for jet-style coloring
    import matplotlib.cm as cm
    jet = cm.get_cmap("jet")
    colored = jet(heatmap_resized)[:, :, :3] * 255.0

    original_float = original_image.astype(np.float32)
    overlay = 0.6 * original_float + 0.4 * colored
    return np.uint8(np.clip(overlay, 0, 255))


def image_to_base64(image_array):
    """Convert an RGB uint8 array to a base64-encoded PNG string."""
    img = tf.keras.utils.array_to_img(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def generate_gradcam_for_image(preprocessed_batch, original_image, model_name):
    """
    Generate a Grad-CAM heatmap overlay for a single image.

    Args:
        preprocessed_batch: (1, 224, 224, 3) preprocessed image for inference
        original_image:     (224, 224, 3) original image in 0–255 range
        model_name:         one of the PSO-selected models

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    registry = PSOModelRegistry.initialize()
    model = registry.get_models()[model_name]

    cnn_base_model = find_cnn_base_model(model)
    conv_parent, last_conv_layer = find_last_conv_layer(model)
    grad_model = build_gradcam_model(model, cnn_base_model, last_conv_layer)

    heatmap = make_gradcam_heatmap(preprocessed_batch, grad_model)
    overlay = overlay_heatmap_on_image(original_image, heatmap)

    return image_to_base64(overlay)
