"""
Save predictions for all models.

Features:
- Runs all models automatically
- Skips models whose prediction files already exist
- Resumes safely after interruption
- Logs progress to pso_cache/_progress.log
- Shows images/sec
- Shows elapsed time
- Shows estimated remaining time
- Continues to next model if one fails
"""

import os
import sys
import time
import warnings
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# SETUP
# ============================================================

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

os.makedirs("pso_cache", exist_ok=True)

LOG_PATH = "pso_cache/_progress.log"

LOG = open(LOG_PATH, "a", buffering=1)


def log(msg):
    """Write to both log file and terminal."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {msg}"

    LOG.write(line + "\n")
    LOG.flush()

    print(line, flush=True)


# ============================================================
# MODELS
# ============================================================

models = [
    "EfficientNet",
    "DenseNet",
    "ResNet",
    "VGG16",
    "Xception",
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocess_map = {

    "EfficientNet":
        tf.keras.applications.efficientnet.preprocess_input,

    "DenseNet":
        tf.keras.applications.densenet.preprocess_input,

    "ResNet":
        tf.keras.applications.resnet.preprocess_input,

    "VGG16":
        tf.keras.applications.vgg16.preprocess_input,

    "Xception":
        tf.keras.applications.xception.preprocess_input,
}


# ============================================================
# MODEL PATHS
# ============================================================

model_paths = {

    "EfficientNet":
        "models/efficientnet_final_model.keras",

    "DenseNet":
        "models/densenet_final_model.keras",

    "ResNet":
        "models/resnet_final_model.keras",

    "VGG16":
        "models/vgg16_final_model.keras",

    "Xception":
        "models/xception_final_model.keras",
}


# ============================================================
# OUTPUT PATHS
# ============================================================

def val_output(name):
    return f"pso_cache/{name}_val.npy"


def test_output(name):
    return f"pso_cache/{name}_test.npy"


# ============================================================
# COMPLETION CHECK
# ============================================================

def model_is_complete(name):
    """
    A model is considered complete only when BOTH
    validation and test prediction files exist.
    """

    val_file = val_output(name)
    test_file = test_output(name)

    if not os.path.exists(val_file):
        return False

    if not os.path.exists(test_file):
        return False

    # Also make sure files aren't empty/corrupt.
    try:
        val = np.load(val_file)
        test = np.load(test_file)

        if val.size == 0:
            return False

        if test.size == 0:
            return False

        return True

    except Exception:
        return False


# ============================================================
# TIME FORMAT
# ============================================================

def format_time(seconds):

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"

    if minutes > 0:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


# ============================================================
# START
# ============================================================

overall_start = time.time()

log("")
log("=" * 70)
log("PREDICTION RUN STARTED")
log("=" * 70)

log(f"Models: {', '.join(models)}")
log(f"Cache directory: pso_cache")
log("")


# ============================================================
# CHECK EXISTING MODELS
# ============================================================

completed = []
remaining = []

for name in models:

    if model_is_complete(name):

        completed.append(name)

        log(
            f"SKIP {name} - predictions already exist"
        )

    else:

        remaining.append(name)

        log(
            f"QUEUE {name} - predictions missing"
        )


log("")

log(
    f"Completed: {len(completed)}/{len(models)}"
)

log(
    f"Remaining: {len(remaining)}/{len(models)}"
)

log("")


# ============================================================
# NOTHING TO DO
# ============================================================

if not remaining:

    log("=" * 70)
    log("ALL MODELS ARE ALREADY COMPLETE")
    log("=" * 70)

    LOG.close()
    sys.exit(0)


# ============================================================
# PROCESS MODELS
# ============================================================

model_times = []


for model_index, name in enumerate(remaining, start=1):

    model_start = time.time()

    log("")
    log("=" * 70)
    log(
        f"STARTING {name} "
        f"({model_index}/{len(remaining)} remaining)"
    )
    log("=" * 70)

    try:

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        preprocess = preprocess_map[name]
        model_path = model_paths[name]

        log(f"Loading model: {model_path}")

        load_start = time.time()

        model = tf.keras.models.load_model(
            model_path
        )

        load_time = time.time() - load_start

        log(
            f"{name} loaded in "
            f"{format_time(load_time)}"
        )


        # ----------------------------------------------------
        # DATA GENERATOR
        # ----------------------------------------------------

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        log(f"{name}: preparing validation generator...")

        val_gen = datagen.flow_from_directory(

            "datasets_split/val",

            target_size=(224, 224),

            batch_size=32,

            class_mode="binary",

            shuffle=False
        )

        # Save labels once.
        np.save(
            "pso_cache/val_labels.npy",
            val_gen.classes
        )

        log(
            f"{name}: validation = "
            f"{val_gen.samples:,} images / "
            f"{len(val_gen)} batches"
        )

        log(
            f"{name}: starting validation prediction..."
        )

        val_start = time.time()

        vp = model.predict(
            val_gen,
            verbose=1
        ).flatten()

        val_time = time.time() - val_start

        val_speed = (
            val_gen.samples / val_time
            if val_time > 0
            else 0
        )

        log(
            f"{name}: validation DONE"
        )

        log(
            f"{name}: validation time = "
            f"{format_time(val_time)}"
        )

        log(
            f"{name}: validation speed = "
            f"{val_speed:.1f} images/sec"
        )

        np.save(
            val_output(name),
            vp
        )

        log(
            f"{name}: saved {val_output(name)}"
        )


        # ====================================================
        # TEST
        # ====================================================

        log(f"{name}: preparing test generator...")

        test_gen = datagen.flow_from_directory(

            "datasets_split/test",

            target_size=(224, 224),

            batch_size=32,

            class_mode="binary",

            shuffle=False
        )

        np.save(
            "pso_cache/test_labels.npy",
            test_gen.classes
        )

        log(
            f"{name}: test = "
            f"{test_gen.samples:,} images / "
            f"{len(test_gen)} batches"
        )

        log(
            f"{name}: starting test prediction..."
        )

        test_start = time.time()

        tp = model.predict(
            test_gen,
            verbose=1
        ).flatten()

        test_time = time.time() - test_start

        test_speed = (
            test_gen.samples / test_time
            if test_time > 0
            else 0
        )

        log(
            f"{name}: test DONE"
        )

        log(
            f"{name}: test time = "
            f"{format_time(test_time)}"
        )

        log(
            f"{name}: test speed = "
            f"{test_speed:.1f} images/sec"
        )

        np.save(
            test_output(name),
            tp
        )

        log(
            f"{name}: saved {test_output(name)}"
        )


        # ====================================================
        # VERIFY
        # ====================================================

        if model_is_complete(name):

            model_time = time.time() - model_start

            model_times.append(model_time)

            log("")
            log(
                f"SUCCESS {name}"
            )

            log(
                f"Total model time: "
                f"{format_time(model_time)}"
            )

            log(
                f"Validation predictions: "
                f"{vp.shape}"
            )

            log(
                f"Test predictions: "
                f"{tp.shape}"
            )

        else:

            raise RuntimeError(
                f"{name}: prediction files were not "
                f"created correctly."
            )


        # ====================================================
        # ESTIMATE REMAINING TIME
        # ====================================================

        completed_this_run = len(model_times)

        models_left = (
            len(remaining) - completed_this_run
        )

        if completed_this_run > 0:

            avg_model_time = (
                sum(model_times)
                / completed_this_run
            )

            estimated_remaining = (
                avg_model_time * models_left
            )

            log(
                f"Estimated remaining time: "
                f"{format_time(estimated_remaining)}"
            )


        # ====================================================
        # CLEANUP
        # ====================================================

        del model
        del val_gen
        del test_gen
        del vp
        del tp

        tf.keras.backend.clear_session()

        log(
            f"{name}: memory/session cleared"
        )


    except Exception as e:

        log("")
        log(
            f"ERROR while processing {name}"
        )

        log(
            f"{type(e).__name__}: {e}"
        )

        log(
            traceback.format_exc()
        )

        log(
            f"Skipping {name} and continuing..."
        )

        tf.keras.backend.clear_session()

        continue


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = time.time() - overall_start

log("")
log("=" * 70)
log("RUN FINISHED")
log("=" * 70)

log(
    f"Total runtime: {format_time(total_time)}"
)

log("")

log("FINAL STATUS:")

for name in models:

    if model_is_complete(name):

        log(
            f"  ✓ {name}: COMPLETE"
        )

    else:

        log(
            f"  ✗ {name}: INCOMPLETE"
        )


log("")
log("=" * 70)
log("DONE")
log("=" * 70)

LOG.close()