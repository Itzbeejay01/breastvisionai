"""Validate the model artifacts required by the prediction API.

This is intentionally separate from Django startup so Docker builds and
container logs identify broken/missing artifacts before Gunicorn serves traffic.
"""

import argparse
import json
from pathlib import Path

import joblib
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "EfficientNet": "efficientnet_final_model.keras",
    "DenseNet": "densenet_final_model.keras",
    "ResNet": "resnet_final_model.keras",
    "VGG16": "vgg16_final_model.keras",
    "Xception": "xception_final_model.keras",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load every Keras artifact, not only the PSO-selected models.",
    )
    args = parser.parse_args()

    selection = json.loads((ROOT / "PSO_Result/pso_selected_models.json").read_text())
    names = list(MODEL_FILES) if args.all else selection["selected_models"]

    print(f"Validating {len(names)} Keras model(s)...", flush=True)
    for name in names:
        path = MODEL_DIR / MODEL_FILES[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} model: {path}")
        model = tf.keras.models.load_model(path, compile=False)
        print(
            f"  OK {name}: input={model.input_shape}, output={model.output_shape}",
            flush=True,
        )

    meta_path = ROOT / "Stacking_Result/pso_stacked_gb_model.joblib"
    meta = joblib.load(meta_path)
    if getattr(meta, "n_features_in_", None) != len(selection["selected_models"]) + 1:
        raise ValueError(
            "Meta-learner feature count does not match the selected model ensemble"
        )
    print(
        f"  OK GradientBoosting meta-learner: features={meta.n_features_in_}",
        flush=True,
    )
    print("Model artifact validation passed.", flush=True)


if __name__ == "__main__":
    main()
