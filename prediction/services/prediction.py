"""
Prediction pipeline using PSO-selected models with late fusion + GB meta-learner.

Pipeline:
    Image (raw or pre-processed)
        ↓
    3 PSO-selected CNNs
        → per-model probability
        ↓
    Late fusion (equal-weight average)
        → fused probability
        ↓
    GradientBoostingClassifier meta-learner
        → final probability + prediction
"""

import os
import numpy as np
import tensorflow as tf

from prediction.services.model_registry import PSOModelRegistry
from prediction.services.preprocessing import preprocess_image_for_all_models
from prediction.services.gradcam import generate_gradcam_for_image

CLASS_NAMES = ["benign", "malignant"]


def predict_single(image_source, image_type="raw"):
    """
    Run the full PSO ensemble prediction on a single image.

    Args:
        image_source: file path or Django UploadedFile
        image_type:   "raw" (apply model-specific preprocessing) or
                      "processed" (skip preprocessing)

    Returns:
        dict with keys:
            prediction, confidence, fused_probability,
            prediction_probability, model_breakdown,
            pso_weights, ensemble_method, heatmap_base64
    """
    registry = PSOModelRegistry.initialize()
    models = registry.get_models()
    meta_learner = registry.get_meta_learner()
    fusion_config = registry.get_fusion_config()
    selected_models = registry.get_selected_model_names()

    # --- Step 1: Preprocess image for all 3 models ---
    preprocessed = preprocess_image_for_all_models(image_source, image_type)

    # --- Step 2: Run each PSO-selected model ---
    model_probs = {}
    model_breakdown = {}

    for model_name in selected_models:
        batched, original_array = preprocessed[model_name]
        model = models[model_name]

        prob = float(model.predict(batched, verbose=0).flatten()[0])
        model_probs[model_name] = prob

    # --- Step 3: Late fusion (equal-weight average) ---
    fusion_weights = registry.get_fusion_weights()
    probs_array = np.array([model_probs[m] for m in selected_models])
    weights_array = np.array([fusion_weights[m] for m in selected_models])

    fused_probability = float(np.average(probs_array, weights=weights_array))

    # --- Step 4: GB meta-learner ---
    # Features: [p1, p2, p3, fused_probability]
    gb_features = np.array([[model_probs[m] for m in selected_models] + [fused_probability]])
    gb_proba = float(meta_learner.predict_proba(gb_features)[0, 1])

    # --- Step 5: Final prediction ---
    prediction_idx = int(gb_proba >= 0.5)
    prediction = CLASS_NAMES[prediction_idx]
    confidence = gb_proba if prediction == "malignant" else (1.0 - gb_proba)
    confidence = float(confidence)

    # --- Step 6: Model breakdown with weights ---
    for model_name in selected_models:
        model_breakdown[model_name] = {
            "probability": model_probs[model_name],
            "weight": fusion_weights[model_name],
        }

    # --- Step 7: PSO weights ---
    pso_weights = registry.get_pso_weights()
    selected_pso_weights = {m: pso_weights.get(m, fusion_weights[m]) for m in selected_models}

    # --- Step 8: Grad-CAM heatmap (using first model) ---
    # Use EfficientNet for heatmap generation (it's the highest-weighted model)
    heatmap_b64 = None
    primary_model = selected_models[0]
    primary_batch, _ = preprocessed[primary_model]
    try:
        heatmap_b64 = generate_gradcam_for_image(
            primary_batch,
            original_array,
            primary_model,
        )
    except Exception as e:
        heatmap_b64 = None

    return {
        "prediction": prediction,
        "prediction_probability": gb_proba,
        "confidence": confidence,
        "fused_probability": fused_probability,
        "model_breakdown": model_breakdown,
        "selected_models": selected_models,
        "pso_weights": selected_pso_weights,
        "fusion_weights": fusion_weights,
        "ensemble_method": fusion_config.get("fusion_strategy", "decision_level_late_fusion"),
        "meta_learner": "GradientBoostingClassifier",
        "heatmap_base64": heatmap_b64,
        "image_type": image_type,
    }


def predict_batch(image_sources, image_type="raw"):
    """
    Run prediction on multiple images.

    Args:
        image_sources: list of (file_path or UploadedFile)
        image_type:    "raw" or "processed"

    Returns:
        dict: {results: [...], total, summary}
    """
    results = []
    for i, src in enumerate(image_sources):
        result = predict_single(src, image_type=image_type)
        result["index"] = i + 1
        results.append(result)

    predictions = [r["prediction"] for r in results]
    malignant_count = sum(1 for p in predictions if p == "malignant")
    benign_count = sum(1 for p in predictions if p == "benign")

    summary = {
        "total": len(results),
        "malignant_count": malignant_count,
        "benign_count": benign_count,
        "avg_confidence": float(np.mean([r["confidence"] for r in results])),
    }

    return {
        "results": results,
        "total": len(results),
        "summary": summary,
    }
