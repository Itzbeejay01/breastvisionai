"""
Grad-CAM utilities for BreastVisionAI.

The functions in this module explain the visual evidence used by an individual
CNN. They support class-targeted explanations for the binary benign/malignant
task and can return both the normalized heatmap and the rendered overlay.
"""

import base64
import io

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf

from prediction.services.model_registry import PSOModelRegistry

IMAGE_SIZE = (224, 224)
VALID_TARGETS = {"benign", "malignant"}


def find_cnn_base_model(model):
    """Find the nested CNN backbone inside an outer Keras model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def find_last_conv_layer(model):
    """Find the last convolutional layer, including nested backbones."""
    convolution_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )

    for layer in reversed(model.layers):
        if isinstance(layer, convolution_types):
            return model, layer

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, convolution_types):
                    return layer, nested_layer

    raise ValueError(f"Could not find a convolutional layer in model: {model.name}")


def build_gradcam_model(model, cnn_base_model, last_conv_layer):
    """Build a model that returns convolution activations and prediction."""
    if cnn_base_model is not None:
        base_input = cnn_base_model.input
        conv_output = last_conv_layer.output
        x = cnn_base_model.output
        base_index = model.layers.index(cnn_base_model)
        for layer in model.layers[base_index + 1:]:
            x = layer(x)
        return tf.keras.models.Model(
            inputs=base_input,
            outputs=[conv_output, x],
        )

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )


def make_gradcam_heatmap(image, grad_model, target_class="malignant"):
    """
    Generate a normalized Grad-CAM heatmap for the requested class.

    For a single sigmoid output, the model output represents malignancy
    probability. A benign explanation therefore differentiates (1 - p),
    rather than reusing the malignant target.
    """
    target_class = str(target_class).lower()
    if target_class not in VALID_TARGETS:
        raise ValueError(f"target_class must be one of {sorted(VALID_TARGETS)}")

    with tf.GradientTape() as tape:
        if len(grad_model.inputs) == 1:
            input_name = grad_model.inputs[0].name.split(":", 1)[0]
            conv_outputs, predictions = grad_model(
                {input_name: image},
                training=False,
            )
        else:
            conv_outputs, predictions = grad_model(image, training=False)

        if predictions.shape[-1] == 1:
            malignant_score = predictions[:, 0]
            class_channel = (
                malignant_score
                if target_class == "malignant"
                else 1.0 - malignant_score
            )
        else:
            target_index = 1 if target_class == "malignant" else 0
            class_channel = predictions[:, target_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    if gradients is None:
        raise ValueError(
            "Gradients are None. The selected convolutional layer is not "
            "connected to the requested prediction."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap):
    """Blend a normalized heatmap with the original 224x224 RGB image."""
    heatmap_resized = tf.image.resize(
        np.expand_dims(heatmap, axis=-1),
        IMAGE_SIZE,
        method="bilinear",
    ).numpy().squeeze()

    jet = cm.get_cmap("jet")
    colored = jet(heatmap_resized)[:, :, :3] * 255.0

    original_float = original_image.astype(np.float32)
    overlay = 0.6 * original_float + 0.4 * colored
    return np.uint8(np.clip(overlay, 0, 255))


def image_to_base64(image_array):
    """Convert an RGB uint8 array to a base64 PNG data URL."""
    img = tf.keras.utils.array_to_img(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_gradcam_artifacts(
    preprocessed_batch,
    original_image,
    model_name,
    target_class="malignant",
):
    """Return the raw normalized heatmap and rendered overlay for one model."""
    registry = PSOModelRegistry.initialize()
    model = registry.get_models()[model_name]

    cnn_base_model = find_cnn_base_model(model)
    _, last_conv_layer = find_last_conv_layer(model)
    grad_model = build_gradcam_model(model, cnn_base_model, last_conv_layer)

    heatmap = make_gradcam_heatmap(
        preprocessed_batch,
        grad_model,
        target_class=target_class,
    )
    overlay = overlay_heatmap_on_image(original_image, heatmap)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "overlay_base64": image_to_base64(overlay),
        "target_class": target_class,
        "last_conv_layer": last_conv_layer.name,
    }


def generate_gradcam_for_image(
    preprocessed_batch,
    original_image,
    model_name,
    target_class="malignant",
):
    """Backward-compatible helper returning only the overlay data URL."""
    return generate_gradcam_artifacts(
        preprocessed_batch,
        original_image,
        model_name,
        target_class=target_class,
    )["overlay_base64"]
