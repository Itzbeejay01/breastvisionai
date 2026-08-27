"""
evaluate_efficiency.py

Deployment efficiency comparison between:

    1. All 5 CNN models
    2. PSO-selected 3 CNN models

Measures:
    - Total inference time
    - Average inference time per image
    - Peak memory usage

The purpose is to determine whether PSO-based model selection
reduces deployment cost while maintaining diagnostic performance.

Dataset:
    datasets_split/test/
        benign/
        malignant/

Models:
    models/

PSO selection:
    PSO_Result/pso_selected_models.json

Output:
    Efficiency_Result/
"""

import os
import json
import time
import gc

import numpy as np
import tensorflow as tf

try:
    import tracemalloc
except ImportError:
    tracemalloc = None

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

OUTPUT_DIR = "Efficiency_Result"

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

BATCH_SIZE = 32

RANDOM_SEED = 42


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {

    "EfficientNet": (
        "efficientnet_final_model.keras"
    ),

    "DenseNet": (
        "densenet_final_model.keras"
    ),

    "ResNet": (
        "resnet_final_model.keras"
    ),

    "VGG16": (
        "vgg16_final_model.keras"
    ),

    "Xception": (
        "xception_final_model.keras"
    ),
}


# ============================================================
# LOAD PSO SELECTION
# ============================================================

print("=" * 70)
print("DEPLOYMENT EFFICIENCY EVALUATION")
print("=" * 70)


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


if len(selected_models) != 3:

    raise ValueError(
        "Expected exactly 3 PSO-selected models."
    )


print(
    "\nPSO-selected models:"
)

for model_name in selected_models:

    print(
        f"  - {model_name}"
    )


# ============================================================
# GET TEST IMAGES
# ============================================================

def get_test_images():

    image_paths = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    for class_name in [
        "benign",
        "malignant"
    ]:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(
            class_dir
        ):

            raise FileNotFoundError(
                f"Test directory not found:\n"
                f"{class_dir}"
            )

        for filename in sorted(
            os.listdir(class_dir)
        ):

            if filename.lower().endswith(
                valid_extensions
            ):

                image_paths.append(
                    os.path.join(
                        class_dir,
                        filename
                    )
                )

    return image_paths


image_paths = get_test_images()


if len(image_paths) == 0:

    raise ValueError(
        "No test images were found."
    )


print(
    f"\nTotal test images: "
    f"{len(image_paths)}"
)


# ============================================================
# LOAD TEST IMAGES
# ============================================================

def load_test_dataset(
    image_paths
):

    images = []

    for image_path in image_paths:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image_array = (
            tf.keras.utils.img_to_array(
                image
            )
        )

        # Match the assumed CNN preprocessing
        image_array = (
            image_array.astype(
                np.float32
            ) / 255.0
        )

        images.append(
            image_array
        )

    return np.asarray(
        images,
        dtype=np.float32
    )


print(
    "\nLoading test images..."
)

X_test = load_test_dataset(
    image_paths
)


print(
    f"Test tensor shape: "
    f"{X_test.shape}"
)


# ============================================================
# MEMORY FUNCTION
# ============================================================

def get_process_memory_mb():

    """
    Return current process memory in MB.

    Uses psutil if available.
    """

    try:

        import psutil

        process = (
            psutil.Process(
                os.getpid()
            )
        )

        return (
            process.memory_info().rss
            / (1024 ** 2)
        )

    except ImportError:

        return None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(
    model_names
):

    models = []

    print(
        "\nLoading models:"
    )

    for model_name in model_names:

        if model_name not in MODEL_FILES:

            raise ValueError(
                f"Unknown model: "
                f"{model_name}"
            )

        model_path = os.path.join(
            MODEL_DIR,
            MODEL_FILES[model_name]
        )

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Model not found:\n"
                f"{model_path}"
            )

        print(
            f"  Loading {model_name}..."
        )

        model = load_model(
            model_path
        )

        models.append(
            model
        )

    return models


# ============================================================
# WARM-UP
# ============================================================

def warm_up_models(
    models,
    X
):

    """
    Run one small prediction before timing.

    This prevents model initialization and graph
    tracing from affecting the timing measurement.
    """

    warmup_batch = X[
        :min(
            BATCH_SIZE,
            len(X)
        )
    ]

    for model in models:

        model.predict(
            warmup_batch,
            verbose=0
        )


# ============================================================
# MEASURE INFERENCE
# ============================================================

def measure_inference(
    model_names,
    X
):

    """
    Load the specified CNNs and measure:

        - model loading memory
        - inference time
        - average inference time per image
        - peak process memory

    Returns a dictionary of results.
    """

    tf.keras.backend.clear_session()

    gc.collect()

    memory_before = (
        get_process_memory_mb()
    )

    models = load_models(
        model_names
    )

    memory_after_loading = (
        get_process_memory_mb()
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        "\nWarming up models..."
    )

    warm_up_models(
        models,
        X
    )

    # --------------------------------------------------------
    # Memory tracing
    # --------------------------------------------------------

    if tracemalloc is not None:

        tracemalloc.start()

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    print(
        "\nMeasuring inference..."
    )

    start_time = time.perf_counter()

    predictions = []

    for model in models:

        prediction = model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions.append(
            prediction
        )

    # Force synchronization where possible
    if tf.config.list_physical_devices(
        "GPU"
    ):

        try:

            for device in tf.config.list_physical_devices(
                "GPU"
            ):

                tf.config.experimental.set_memory_growth(
                    device,
                    True
                )

        except Exception:

            pass

    end_time = time.perf_counter()

    total_time = (
        end_time -
        start_time
    )

    # --------------------------------------------------------
    # Peak Python memory
    # --------------------------------------------------------

    peak_traced_memory_mb = None

    if tracemalloc is not None:

        current, peak = (
            tracemalloc.get_traced_memory()
        )

        peak_traced_memory_mb = (
            peak /
            (1024 ** 2)
        )

        tracemalloc.stop()

    memory_after_inference = (
        get_process_memory_mb()
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    average_time = (
        total_time /
        len(X)
    )

    memory_increase = None

    if (
        memory_before is not None
        and memory_after_loading is not None
    ):

        memory_increase = (
            memory_after_loading -
            memory_before
        )

    result = {

        "models": model_names,

        "number_of_models": len(
            model_names
        ),

        "number_of_test_images": len(
            X
        ),

        "total_inference_time_seconds": (
            float(total_time)
        ),

        "average_inference_time_per_image_seconds": (
            float(average_time)
        ),

        "average_inference_time_per_image_ms": (
            float(average_time * 1000)
        ),

        "memory_before_loading_mb": (
            memory_before
        ),

        "memory_after_loading_mb": (
            memory_after_loading
        ),

        "memory_increase_from_loading_mb": (
            memory_increase
        ),

        "memory_after_inference_mb": (
            memory_after_inference
        ),

        "peak_traced_memory_mb": (
            peak_traced_memory_mb
        )
    }

    # --------------------------------------------------------
    # Release models
    # --------------------------------------------------------

    del predictions
    del models

    tf.keras.backend.clear_session()

    gc.collect()

    return result


# ============================================================
# CONFIGURATIONS
# ============================================================

all_models = list(
    MODEL_FILES.keys()
)

pso_models = selected_models


# ============================================================
# EVALUATE ALL 5 MODELS
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION 1: ALL 5 CNN MODELS")
print("=" * 70)

all_5_results = measure_inference(
    all_models,
    X_test
)


print(
    "\nAll 5 CNN results:"
)

print(
    f"  Total time: "
    f"{all_5_results['total_inference_time_seconds']:.4f} s"
)

print(
    f"  Average/image: "
    f"{all_5_results['average_inference_time_per_image_ms']:.4f} ms"
)

if (
    all_5_results[
        "memory_increase_from_loading_mb"
    ]
    is not None
):

    print(
        f"  Model memory increase: "
        f"{all_5_results['memory_increase_from_loading_mb']:.2f} MB"
    )


# ============================================================
# EVALUATE PSO 3 MODELS
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION 2: PSO-SELECTED 3 CNN MODELS")
print("=" * 70)

pso_3_results = measure_inference(
    pso_models,
    X_test
)


print(
    "\nPSO 3-CNN results:"
)

print(
    f"  Total time: "
    f"{pso_3_results['total_inference_time_seconds']:.4f} s"
)

print(
    f"  Average/image: "
    f"{pso_3_results['average_inference_time_per_image_ms']:.4f} ms"
)

if (
    pso_3_results[
        "memory_increase_from_loading_mb"
    ]
    is not None
):

    print(
        f"  Model memory increase: "
        f"{pso_3_results['memory_increase_from_loading_mb']:.2f} MB"
    )


# ============================================================
# CALCULATE REDUCTION
# ============================================================

time_5 = (
    all_5_results[
        "total_inference_time_seconds"
    ]
)

time_3 = (
    pso_3_results[
        "total_inference_time_seconds"
    ]
)


time_reduction_percent = (
    (
        time_5 -
        time_3
    )
    / time_5
    * 100
)


memory_5 = (
    all_5_results[
        "memory_increase_from_loading_mb"
    ]
)

memory_3 = (
    pso_3_results[
        "memory_increase_from_loading_mb"
    ]
)


memory_reduction_percent = None

if (
    memory_5 is not None
    and memory_3 is not None
    and memory_5 > 0
):

    memory_reduction_percent = (
        (
            memory_5 -
            memory_3
        )
        / memory_5
        * 100
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "all_5_cnn": all_5_results,

    "pso_selected_3_cnn": pso_3_results,

    "efficiency_comparison": {

        "inference_time_reduction_percent": (
            float(
                time_reduction_percent
            )
        ),

        "memory_reduction_percent": (
            None
            if memory_reduction_percent is None
            else float(
                memory_reduction_percent
            )
        )
    }
}


results_path = os.path.join(
    OUTPUT_DIR,
    "efficiency_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE HUMAN-READABLE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "efficiency_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "DEPLOYMENT EFFICIENCY COMPARISON\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Configuration 1: All 5 CNN models\n"
    )

    f.write(
        f"Models: {', '.join(all_models)}\n"
    )

    f.write(
        f"Total inference time: "
        f"{time_5:.6f} seconds\n"
    )

    f.write(
        f"Average inference time per image: "
        f"{all_5_results['average_inference_time_per_image_ms']:.6f} ms\n"
    )

    if memory_5 is not None:

        f.write(
            f"Model memory increase: "
            f"{memory_5:.4f} MB\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "Configuration 2: PSO-selected 3 CNN models\n"
    )

    f.write(
        f"Models: {', '.join(pso_models)}\n"
    )

    f.write(
        f"Total inference time: "
        f"{time_3:.6f} seconds\n"
    )

    f.write(
        f"Average inference time per image: "
        f"{pso_3_results['average_inference_time_per_image_ms']:.6f} ms\n"
    )

    if memory_3 is not None:

        f.write(
            f"Model memory increase: "
            f"{memory_3:.4f} MB\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "Efficiency improvement\n"
    )

    f.write(
        f"Inference time reduction: "
        f"{time_reduction_percent:.4f}%\n"
    )

    if memory_reduction_percent is not None:

        f.write(
            f"Memory reduction: "
            f"{memory_reduction_percent:.4f}%\n"
        )

    else:

        f.write(
            "Memory reduction: Not available\n"
        )


# ============================================================
# FINAL DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("EFFICIENCY EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nInference time reduction: "
    f"{time_reduction_percent:.2f}%"
)

if memory_reduction_percent is not None:

    print(
        f"Memory reduction: "
        f"{memory_reduction_percent:.2f}%"
    )

else:

    print(
        "Memory reduction: "
        "Not available"
    )

print(
    "\nResults saved to:"
)

print(
    f"  {OUTPUT_DIR}/"
)

print(
    "\nFiles:"
)

print(
    "  - efficiency_results.json"
)

print(
    "  - efficiency_report.txt"
)