"""
generate_gradcam.py

Grad-CAM visualization for the 3 CNN models selected by PSO.

Pipeline:

    PSO selection
          ↓
    Selected 3 CNNs
          ↓
    Load trained .keras models
          ↓
    Test images
          ↓
    Grad-CAM
          ↓
    Heatmap + original image overlay

Dataset:
    datasets_split/test/
        benign/
        malignant/

Models:
    models/
        densenet_final_model.keras
        efficientnet_final_model.keras
        resnet_final_model.keras
        vgg16_final_model.keras
        xception_final_model.keras

Output:
    GradCAM_Result/
        ModelName/
            benign/
            malignant/
"""


import os
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"

DATASET_DIR = os.path.join(
    "datasets_split",
    "test"
)

PSO_RESULT_DIR = "PSO_Result"

SELECTION_FILE = os.path.join(
    PSO_RESULT_DIR,
    "pso_selected_models.json"
)

OUTPUT_DIR = "GradCAM_Result"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (
    224,
    224
)

# Number of images generated per class for each model.
#
# 10 benign + 10 malignant = 20 images/model
#
# Set to None to process every test image.
IMAGES_PER_CLASS = 10

CLASS_NAMES = [
    "benign",
    "malignant"
]


# ============================================================
# MODEL FILE MAPPING
# ============================================================

MODEL_FILES = {

    "EfficientNet":
        "efficientnet_final_model.keras",

    "DenseNet":
        "densenet_final_model.keras",

    "ResNet":
        "resnet_final_model.keras",

    "VGG16":
        "vgg16_final_model.keras",

    "Xception":
        "xception_final_model.keras",
}


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_image(
    image_array,
    model_name
):
    """
    Apply the same preprocessing expected by the
    corresponding ImageNet CNN architecture.

    VGG16:
        tf.keras.applications.vgg16.preprocess_input

    ResNet:
        tf.keras.applications.resnet.preprocess_input

    EfficientNet:
        EfficientNet preprocessing is handled internally
        by modern tf.keras EfficientNet models, so the
        image is scaled to 0-255 float values.

    DenseNet:
        tf.keras.applications.densenet.preprocess_input

    Xception:
        tf.keras.applications.xception.preprocess_input
    """

    image_array = image_array.astype(
        np.float32
    )

    if model_name == "VGG16":

        image_array = (
            tf.keras.applications.vgg16.preprocess_input(
                image_array
            )
        )

    elif model_name == "ResNet":

        image_array = (
            tf.keras.applications.resnet.preprocess_input(
                image_array
            )
        )

    elif model_name == "DenseNet":

        image_array = (
            tf.keras.applications.densenet.preprocess_input(
                image_array
            )
        )

    elif model_name == "Xception":

        image_array = (
            tf.keras.applications.xception.preprocess_input(
                image_array
            )
        )

    elif model_name == "EfficientNet":

        # Modern tf.keras EfficientNet models include
        # rescaling internally.
        #
        # Therefore the input should remain in the
        # 0-255 range.

        pass

    else:

        # Safe fallback
        image_array = (
            image_array / 255.0
        )

    return np.expand_dims(
        image_array,
        axis=0
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    image_path
):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    return image_array


# ============================================================
# FIND TEST IMAGES
# ============================================================

def get_test_images():

    image_paths = []

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(
            class_dir
        ):

            raise FileNotFoundError(
                f"\nTest directory not found:\n"
                f"{class_dir}"
            )

        files = []

        for filename in os.listdir(
            class_dir
        ):

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff"
                )
            ):

                files.append(
                    os.path.join(
                        class_dir,
                        filename
                    )
                )

        files.sort()

        if (
            IMAGES_PER_CLASS is not None
            and len(files) > IMAGES_PER_CLASS
        ):

            random.seed(42)

            files = random.sample(
                files,
                IMAGES_PER_CLASS
            )

        for path in files:

            image_paths.append(
                (
                    class_name,
                    path
                )
            )

    return image_paths


# ============================================================
# FIND CNN BASE MODEL
# ============================================================

def find_cnn_base_model(
    model
):
    """
    Finds the nested CNN base model inside the outer
    Sequential model.

    Expected structure:

        Sequential
            ↓
        CNN base model
            ↓
        GlobalAveragePooling
            ↓
        Dense
            ↓
        Dense(1)
    """

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            return layer

    return None


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(
    model
):
    """
    Finds the last convolutional layer.

    Works with:
        EfficientNet
        DenseNet
        ResNet
        VGG16
        Xception

    Also handles nested CNN models.
    """

    # --------------------------------------------------------
    # First search the outer model
    # --------------------------------------------------------

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.SeparableConv2D,
                tf.keras.layers.DepthwiseConv2D
            )
        ):

            return model, layer

    # --------------------------------------------------------
    # Search nested model
    # --------------------------------------------------------

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            tf.keras.Model
        ):

            for nested_layer in reversed(
                layer.layers
            ):

                if isinstance(
                    nested_layer,
                    (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.SeparableConv2D,
                        tf.keras.layers.DepthwiseConv2D
                    )
                ):

                    return layer, nested_layer

    raise ValueError(
        f"Could not find a convolutional "
        f"layer in model: {model.name}"
    )


# ============================================================
# BUILD GRAD-CAM MODEL
# ============================================================

def build_gradcam_model(
    model,
    cnn_base_model,
    last_conv_layer
):
    """
    Builds a Grad-CAM model.

    The important part is that the convolutional layer
    and prediction are connected through the actual model
    computation graph.

    This avoids:

        The layer sequential has never been called
    """

    # --------------------------------------------------------
    # Create a model whose outputs are:
    #
    # 1. Last convolutional feature maps
    # 2. Final classification prediction
    # --------------------------------------------------------

    if cnn_base_model is not None:

        # The outer model is:
        #
        # Sequential(
        #     CNN,
        #     GAP,
        #     Dense,
        #     Dropout,
        #     Dense
        # )
        #
        # We need to explicitly run the CNN and then
        # propagate its output through the classifier.

        base_input = cnn_base_model.input

        conv_output = last_conv_layer.output

        # Start from convolutional output and manually
        # pass through the layers after the CNN base.

        x = cnn_base_model.output

        # Find the CNN base position
        base_index = model.layers.index(
            cnn_base_model
        )

        # Pass through all layers after the base model.
        for layer in model.layers[
            base_index + 1:
        ]:

            x = layer(x)

        predictions = x

        grad_model = tf.keras.models.Model(
            inputs=base_input,
            outputs=[
                conv_output,
                predictions
            ]
        )

        return grad_model

    # --------------------------------------------------------
    # Fallback for non-nested models
    # --------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    return grad_model


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image,
    grad_model
):
    """
    Generate Grad-CAM heatmap.
    """

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(image)
        )

        # ----------------------------------------------------
        # Binary classification
        # ----------------------------------------------------

        if predictions.shape[-1] == 1:

            class_channel = (
                predictions[:, 0]
            )

        # ----------------------------------------------------
        # Multi-class classification
        # ----------------------------------------------------

        else:

            predicted_class = tf.argmax(
                predictions[0]
            )

            class_channel = (
                predictions[
                    :,
                    predicted_class
                ]
            )

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_channel,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "The selected convolutional layer "
            "is not connected to the prediction."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Weight feature maps
    # --------------------------------------------------------

    heatmap = (
        conv_outputs
        * pooled_gradients
    )

    # --------------------------------------------------------
    # Sum across channels
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        heatmap,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    return heatmap.numpy()


# ============================================================
# SAVE GRAD-CAM IMAGE
# ============================================================

def save_gradcam(
    original_image,
    heatmap,
    save_path,
    title
):

    plt.figure(
        figsize=(8, 4)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_image.astype(
            np.uint8
        )
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        original_image.astype(
            np.uint8
        )
    )

    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis(
        "off"
    )

    plt.suptitle(
        title
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("GRAD-CAM GENERATION")
print("=" * 70)


# ============================================================
# LOAD PSO SELECTION
# ============================================================

if not os.path.exists(
    SELECTION_FILE
):

    raise FileNotFoundError(
        f"\nPSO selection file not found:\n"
        f"{SELECTION_FILE}"
    )


with open(
    SELECTION_FILE,
    "r"
) as f:

    selection = json.load(f)


selected_models = selection[
    "selected_models"
]


print(
    "\nPSO-selected models:"
)

for model_name in selected_models:

    print(
        f"  - {model_name}"
    )


if len(selected_models) != 3:

    raise ValueError(
        "Expected exactly 3 PSO-selected models."
    )


# ============================================================
# GET TEST IMAGES
# ============================================================

test_images = get_test_images()


print(
    f"\nImages selected for Grad-CAM: "
    f"{len(test_images)}"
)


# ============================================================
# CHECK IMAGE DISTRIBUTION
# ============================================================

for class_name in CLASS_NAMES:

    count = sum(
        1
        for label, _ in test_images
        if label == class_name
    )

    print(
        f"  {class_name}: {count}"
    )


# ============================================================
# PROCESS EACH SELECTED MODEL
# ============================================================

for model_name in selected_models:

    print("\n" + "=" * 70)

    print(
        f"PROCESSING: {model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Check model mapping
    # --------------------------------------------------------

    if model_name not in MODEL_FILES:

        print(
            f"WARNING: No model file mapping "
            f"for {model_name}. Skipping."
        )

        continue


    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        MODEL_FILES[model_name]
    )


    if not os.path.exists(
        model_path
    ):

        print(
            f"WARNING: Model not found:\n"
            f"{model_path}"
        )

        continue


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"\nLoading model:\n"
        f"{model_path}"
    )

    model = load_model(
        model_path
    )

    print(
        "Model loaded successfully."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Call the model once before inspecting
    # its computational graph.
    # --------------------------------------------------------

    dummy_input = tf.zeros(
        (
            1,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        dtype=tf.float32
    )

    try:

        model(dummy_input)

    except Exception as error:

        print(
            "\nWARNING: Initial model call failed:"
        )

        print(
            error
        )


    # --------------------------------------------------------
    # Find CNN base model
    # --------------------------------------------------------

    cnn_base_model = (
        find_cnn_base_model(
            model
        )
    )


    if cnn_base_model is not None:

        print(
            f"CNN base model: "
            f"{cnn_base_model.name}"
        )

    else:

        print(
            "CNN base model: "
            "Not nested"
        )


    # --------------------------------------------------------
    # Find last convolutional layer
    # --------------------------------------------------------

    conv_parent_model, last_conv_layer = (
        find_last_conv_layer(
            model
        )
    )


    print(
        f"Last convolutional layer: "
        f"{last_conv_layer.name}"
    )


    # --------------------------------------------------------
    # Build Grad-CAM model
    # --------------------------------------------------------

    try:

        grad_model = build_gradcam_model(
            model,
            cnn_base_model,
            last_conv_layer
        )

    except Exception as error:

        print(
            "\nERROR building Grad-CAM model:"
        )

        print(
            error
        )

        tf.keras.backend.clear_session()

        continue


    print(
        "Grad-CAM model created successfully."
    )


    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    model_output_dir = os.path.join(
        OUTPUT_DIR,
        model_name
    )

    os.makedirs(
        model_output_dir,
        exist_ok=True
    )


    for class_name in CLASS_NAMES:

        os.makedirs(
            os.path.join(
                model_output_dir,
                class_name
            ),
            exist_ok=True
        )


    # ========================================================
    # PROCESS TEST IMAGES
    # ========================================================

    successful = 0
    failed = 0


    for class_name, image_path in test_images:

        try:

            # ------------------------------------------------
            # Load original image
            # ------------------------------------------------

            original_image = load_image(
                image_path
            )


            # ------------------------------------------------
            # Apply model-specific preprocessing
            # ------------------------------------------------

            processed_image = (
                preprocess_image(
                    original_image,
                    model_name
                )
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                processed_image,
                verbose=0
            )


            # ------------------------------------------------
            # Binary classification
            # ------------------------------------------------

            if prediction.shape[-1] == 1:

                probability = float(
                    prediction[0][0]
                )

                predicted_class = (
                    "malignant"
                    if probability >= 0.5
                    else "benign"
                )

                confidence = (
                    probability
                    if predicted_class == "malignant"
                    else 1.0 - probability
                )


            # ------------------------------------------------
            # Multi-class classification
            # ------------------------------------------------

            else:

                predicted_index = int(
                    np.argmax(
                        prediction[0]
                    )
                )

                predicted_class = (
                    CLASS_NAMES[
                        predicted_index
                    ]
                )

                confidence = float(
                    prediction[
                        0,
                        predicted_index
                    ]
                )


            # ------------------------------------------------
            # Generate Grad-CAM
            # ------------------------------------------------

            heatmap = (
                make_gradcam_heatmap(
                    processed_image,
                    grad_model
                )
            )


            # ------------------------------------------------
            # Create output filename
            # ------------------------------------------------

            filename = os.path.basename(
                image_path
            )

            filename_without_ext = (
                os.path.splitext(
                    filename
                )[0]
            )


            output_filename = (
                f"{filename_without_ext}"
                f"_gradcam_"
                f"pred_{predicted_class}"
                f".png"
            )


            output_path = os.path.join(
                model_output_dir,
                class_name,
                output_filename
            )


            # ------------------------------------------------
            # Title
            # ------------------------------------------------

            title = (
                f"{model_name} | "
                f"True: {class_name} | "
                f"Pred: {predicted_class} "
                f"({confidence:.3f})"
            )


            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            save_gradcam(
                original_image,
                heatmap,
                output_path,
                title
            )


            successful += 1


            print(
                f"Saved: {output_path}"
            )


        except Exception as error:

            failed += 1

            print(
                f"\nERROR processing:"
                f"\n  {image_path}"
                f"\n  {error}"
            )


    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    print(
        f"\n{model_name} Grad-CAM summary:"
    )

    print(
        f"  Successful: {successful}"
    )

    print(
        f"  Failed:     {failed}"
    )


    # --------------------------------------------------------
    # Clear memory
    # --------------------------------------------------------

    del grad_model
    del model

    tf.keras.backend.clear_session()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAD-CAM GENERATION COMPLETE")
print("=" * 70)


print(
    f"\nResults saved to:"
)

print(
    f"  {OUTPUT_DIR}/"
)


print(
    "\nSelected models:"
)

for model_name in selected_models:

    print(
        f"  - {model_name}"
    )


print(
    "\nExpected output structure:"
)

for model_name in selected_models:

    print(
        f"  {OUTPUT_DIR}/{model_name}/benign/"
    )

    print(
        f"  {OUTPUT_DIR}/{model_name}/malignant/"
    )


print(
    "\nDone."
)