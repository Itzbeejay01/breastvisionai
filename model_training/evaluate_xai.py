"""
Quantitative XAI evaluation for BreastVisionAI.

This script does not retrain any model. It evaluates the deployed PSO-selected
CNN explanations using:
1. Deletion faithfulness: mask the most salient pixels and measure target-score drop.
2. Explanation stability: small brightness perturbation vs. Grad-CAM correlation.
3. Inter-model agreement: pairwise correlation among the three Grad-CAM maps.

Example:
    python model_training/evaluate_xai.py \
        --dataset datasets_split/test \
        --limit 20 \
        --output XAI_Evaluation
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import tensorflow as tf

from prediction.services.gradcam import generate_gradcam_artifacts
from prediction.services.model_registry import PSOModelRegistry
from prediction.services.prediction import predict_single
from prediction.services.preprocessing import (
    load_image_as_array,
    preprocess_for_model,
)


def safe_correlation(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def target_score(model, batch, target_class):
    if len(model.inputs) == 1:
        input_name = model.inputs[0].name.split(":", 1)[0]
        prediction = model.predict({input_name: batch}, verbose=0)
    else:
        prediction = model.predict(batch, verbose=0)

    prediction = np.asarray(prediction)
    if prediction.shape[-1] == 1:
        malignant = float(prediction.reshape(-1)[0])
        return malignant if target_class == "malignant" else 1.0 - malignant

    target_index = 1 if target_class == "malignant" else 0
    return float(prediction[0, target_index])


def delete_salient_region(original, heatmap, fraction=0.15):
    """Replace the top-fraction salient pixels with the image-channel mean."""
    resized = tf.image.resize(
        np.expand_dims(heatmap, axis=-1),
        original.shape[:2],
        method="bilinear",
    ).numpy().squeeze()

    threshold = np.quantile(resized, 1.0 - fraction)
    mask = resized >= threshold

    modified = original.astype(np.float32).copy()
    channel_mean = np.mean(modified, axis=(0, 1))
    modified[mask] = channel_mean
    return np.clip(modified, 0, 255)


def discover_images(dataset_dir, limit):
    extensions = {".png", ".jpg", ".jpeg"}
    paths = []
    for root, _, filenames in os.walk(dataset_dir):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() in extensions:
                paths.append(os.path.join(root, filename))
                if limit and len(paths) >= limit:
                    return paths
    return paths


def evaluate_image(image_path, registry):
    selected_models = registry.get_selected_model_names()
    models = registry.get_models()

    ensemble = predict_single(
        image_path,
        image_type="raw",
        generate_heatmap=False,
    )
    target_class = ensemble["prediction"]
    original = load_image_as_array(image_path)

    heatmaps = {}
    deletion_drops = {}
    stability_scores = {}

    for model_name in selected_models:
        batch = preprocess_for_model(original.copy(), model_name)
        artifacts = generate_gradcam_artifacts(
            batch,
            original,
            model_name,
            target_class=target_class,
        )
        heatmap = artifacts["heatmap"]
        heatmaps[model_name] = heatmap

        model = models[model_name]
        baseline_score = target_score(model, batch, target_class)

        deleted = delete_salient_region(original, heatmap)
        deleted_batch = preprocess_for_model(deleted, model_name)
        deleted_score = target_score(model, deleted_batch, target_class)
        deletion_drops[model_name] = float(baseline_score - deleted_score)

        perturbed = np.clip(original.astype(np.float32) * 1.02, 0, 255)
        perturbed_batch = preprocess_for_model(perturbed, model_name)
        perturbed_artifacts = generate_gradcam_artifacts(
            perturbed_batch,
            perturbed,
            model_name,
            target_class=target_class,
        )
        stability_scores[model_name] = safe_correlation(
            heatmap,
            perturbed_artifacts["heatmap"],
        )

    pairwise = {}
    pairwise_values = []
    for left, right in combinations(selected_models, 2):
        value = safe_correlation(heatmaps[left], heatmaps[right])
        pairwise[f"{left}__{right}"] = value
        pairwise_values.append(value)

    return {
        "image": image_path,
        "target_class": target_class,
        "final_malignancy_probability": ensemble["prediction_probability"],
        "deletion_score_drop": deletion_drops,
        "stability_correlation": stability_scores,
        "pairwise_gradcam_correlation": pairwise,
        "mean_inter_model_correlation": float(np.mean(pairwise_values))
        if pairwise_values
        else 0.0,
    }


def summarize(results):
    if not results:
        return {}

    model_names = list(results[0]["deletion_score_drop"].keys())
    summary = {
        "images_evaluated": len(results),
        "per_model": {},
        "mean_inter_model_correlation": float(
            np.mean([r["mean_inter_model_correlation"] for r in results])
        ),
    }

    for model_name in model_names:
        summary["per_model"][model_name] = {
            "mean_deletion_score_drop": float(
                np.mean(
                    [
                        r["deletion_score_drop"][model_name]
                        for r in results
                    ]
                )
            ),
            "mean_stability_correlation": float(
                np.mean(
                    [
                        r["stability_correlation"][model_name]
                        for r in results
                    ]
                )
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets_split/test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="XAI_Evaluation")
    args = parser.parse_args()

    image_paths = discover_images(args.dataset, args.limit)
    if not image_paths:
        raise FileNotFoundError(
            f"No PNG/JPEG images found under {args.dataset!r}."
        )

    os.makedirs(args.output, exist_ok=True)
    registry = PSOModelRegistry.initialize()

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path}")
        results.append(evaluate_image(image_path, registry))

    summary = summarize(results)
    payload = {
        "methodology": {
            "deletion_fraction": 0.15,
            "stability_perturbation": "brightness x1.02",
            "agreement_metric": "Pearson correlation of normalized Grad-CAM maps",
            "target": "final ensemble-predicted class",
        },
        "summary": summary,
        "results": results,
    }

    json_path = os.path.join(args.output, "xai_metrics.json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    text_path = os.path.join(args.output, "xai_summary.txt")
    with open(text_path, "w") as handle:
        handle.write("BreastVisionAI XAI Evaluation\n")
        handle.write("=" * 60 + "\n")
        handle.write(json.dumps(summary, indent=2))
        handle.write("\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
