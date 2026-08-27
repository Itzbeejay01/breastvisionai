# ============================================================
# computational_efficiency_memory.py
#
# Measures actual CPU RAM usage and inference time for:
#   1. All 5 CNNs
#   2. PSO-selected 3 CNNs
#
# No estimated values are used.
# ============================================================

import os
import gc
import time
import json
import psutil
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"

TEST_DIR = "datasets_split/test"

OUTPUT_DIR = "Efficiency_Result"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

MODEL_PATHS = {
    "EfficientNet": os.path.join(
        MODEL_DIR,
        "efficientnet_final_model.keras"
    ),

    "DenseNet": os.path.join(
        MODEL_DIR,
        "densenet_final_model.keras"
    ),

    "ResNet": os.path.join(
        MODEL_DIR,
        "resnet_final_model.keras"
    ),

    "VGG16": os.path.join(
        MODEL_DIR,
        "vgg16_final_model.keras"
    ),

    "Xception": os.path.join(
        MODEL_DIR,
        "xception_final_model.keras"
    ),
}


ALL_MODELS = [
    "EfficientNet",
    "DenseNet",
    "ResNet",
    "VGG16",
    "Xception"
]


# PSO result from your actual experiment
PSO_SELECTED_MODELS = [
    "EfficientNet",
    "ResNet",
    "VGG16"
]


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32


# ============================================================
# MEMORY FUNCTIONS
# ============================================================

process = psutil.Process(
    os.getpid()
)


def get_memory_mb():

    return process.memory_info().rss / (
        1024 * 1024
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_images():

    images = []

    labels = []

    class_names = [
        "benign",
        "malignant"
    ]

    for label, class_name in enumerate(
        class_names
    ):

        class_dir = os.path.join(
            TEST_DIR,
            class_name
        )

        if not os.path.exists(class_dir):

            raise FileNotFoundError(
                f"Directory not found: {class_dir}"
            )

        for filename in sorted(
            os.listdir(class_dir)
        ):

            filepath = os.path.join(
                class_dir,
                filename
            )

            if not filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            ):
                continue

            image = load_img(
                filepath,
                target_size=IMAGE_SIZE
            )

            image = img_to_array(
                image
            )

            image = image / 255.0

            images.append(
                image
            )

            labels.append(
                label
            )

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    return images, labels


# ============================================================
# MEASURE CONFIGURATION
# ============================================================

def measure_configuration(
    model_names,
    images
):

    print("\n" + "=" * 70)

    print(
        "MEASURING:",
        ", ".join(model_names)
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Force garbage collection
    # --------------------------------------------------------

    gc.collect()


    # --------------------------------------------------------
    # Memory before loading
    # --------------------------------------------------------

    memory_before = get_memory_mb()

    print(
        f"\nMemory before loading: "
        f"{memory_before:.2f} MB"
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = []

    for name in model_names:

        print(
            f"\nLoading {name}..."
        )

        model_path = MODEL_PATHS[name]

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        model = load_model(
            model_path,
            compile=False
        )

        models.append(
            model
        )

        current_memory = get_memory_mb()

        print(
            f"Memory after loading "
            f"{name}: "
            f"{current_memory:.2f} MB"
        )


    # --------------------------------------------------------
    # Memory after loading all models
    # --------------------------------------------------------

    memory_after_loading = get_memory_mb()

    print(
        f"\nMemory after loading all models: "
        f"{memory_after_loading:.2f} MB"
    )


    memory_increase_loading = (
        memory_after_loading
        -
        memory_before
    )


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        "\nRunning warm-up inference..."
    )

    warmup_images = images[
        :min(32, len(images))
    ]

    for model in models:

        model.predict(
            warmup_images,
            batch_size=BATCH_SIZE,
            verbose=0
        )


    gc.collect()


    # --------------------------------------------------------
    # Start inference measurement
    # --------------------------------------------------------

    print(
        "\nStarting measured inference..."
    )

    start_time = time.perf_counter()

    peak_memory = get_memory_mb()


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    predictions = []

    for model in models:

        prediction = model.predict(
            images,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions.append(
            prediction
        )

        current_memory = get_memory_mb()

        peak_memory = max(
            peak_memory,
            current_memory
        )


    end_time = time.perf_counter()


    # --------------------------------------------------------
    # Final measurements
    # --------------------------------------------------------

    total_time = (
        end_time
        -
        start_time
    )

    average_time = (
        total_time
        /
        len(images)
    )

    memory_after_inference = (
        get_memory_mb()
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {

        "models": model_names,

        "number_of_models": len(
            model_names
        ),

        "number_of_test_images": len(
            images
        ),

        "memory_before_loading_mb":
            memory_before,

        "memory_after_loading_mb":
            memory_after_loading,

        "memory_increase_from_loading_mb":
            memory_increase_loading,

        "memory_after_inference_mb":
            memory_after_inference,

        "peak_ram_memory_mb":
            peak_memory,

        "total_inference_time_seconds":
            total_time,

        "total_inference_time_minutes":
            total_time / 60,

        "average_inference_time_per_image_seconds":
            average_time,

        "average_inference_time_per_image_ms":
            average_time * 1000
    }


    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------

    del predictions

    del models

    gc.collect()


    return result


# ============================================================
# MAIN
# ============================================================

print(
    "\nLoading test dataset..."
)

images, labels = load_test_images()

print(
    f"\nTest images loaded: "
    f"{len(images)}"
)

print(
    f"Image shape: "
    f"{images.shape}"
)


# ============================================================
# MEASURE ALL 5
# ============================================================

all_5_result = measure_configuration(
    ALL_MODELS,
    images
)


# ============================================================
# CLEAN MEMORY BEFORE SECOND TEST
# ============================================================

gc.collect()

time.sleep(3)


# ============================================================
# MEASURE PSO 3
# ============================================================

pso_3_result = measure_configuration(
    PSO_SELECTED_MODELS,
    images
)


# ============================================================
# EFFICIENCY COMPARISON
# ============================================================

time_5 = (
    all_5_result[
        "total_inference_time_seconds"
    ]
)

time_3 = (
    pso_3_result[
        "total_inference_time_seconds"
    ]
)


memory_5 = (
    all_5_result[
        "peak_ram_memory_mb"
    ]
)

memory_3 = (
    pso_3_result[
        "peak_ram_memory_mb"
    ]
)


time_reduction = (
    (
        time_5
        -
        time_3
    )
    /
    time_5
) * 100


memory_reduction = (
    (
        memory_5
        -
        memory_3
    )
    /
    memory_5
) * 100


# ============================================================
# FINAL RESULT
# ============================================================

final_results = {

    "all_5_cnn": all_5_result,

    "pso_selected_3_cnn": pso_3_result,

    "efficiency_comparison": {

        "inference_time_reduction_percent":
            time_reduction,

        "peak_memory_reduction_percent":
            memory_reduction
    }
}


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "memory_efficiency_results.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPUTATIONAL EFFICIENCY RESULTS")
print("=" * 70)


print("\nALL 5 CNNs")

print(
    f"Models: {len(ALL_MODELS)}"
)

print(
    f"Total inference time: "
    f"{time_5:.2f} seconds "
    f"({time_5 / 60:.2f} minutes)"
)

print(
    f"Average/image: "
    f"{all_5_result['average_inference_time_per_image_ms']:.2f} ms"
)

print(
    f"Peak RAM: "
    f"{memory_5:.2f} MB"
)


print("\nPSO-SELECTED 3 CNNs")

print(
    f"Models: {len(PSO_SELECTED_MODELS)}"
)

print(
    f"Total inference time: "
    f"{time_3:.2f} seconds "
    f"({time_3 / 60:.2f} minutes)"
)

print(
    f"Average/image: "
    f"{pso_3_result['average_inference_time_per_image_ms']:.2f} ms"
)

print(
    f"Peak RAM: "
    f"{memory_3:.2f} MB"
)


print("\nEFFICIENCY IMPROVEMENT")

print(
    f"Inference-time reduction: "
    f"{time_reduction:.2f}%"
)

print(
    f"Peak-memory reduction: "
    f"{memory_reduction:.2f}%"
)


print("\nResults saved to:")

print(
    json_path
)

print(
    "\n" + "=" * 70
)