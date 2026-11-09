"""
Explainability services for BreastVisionAI.

This layer deliberately separates:
- CNN visual evidence: class-targeted Grad-CAM for each PSO-selected CNN.
- Ensemble visual evidence: fusion-weighted consensus Grad-CAM.
- Meta-learner decision evidence: SHAP values for the GradientBoosting model.
- Model agreement: per-model votes, support count, and probability spread.

The prediction model itself is never retrained or modified here.
"""

import os
import logging

import numpy as np
import shap
import tensorflow as tf
from django.conf import settings

from prediction.services.gradcam import (
    generate_gradcam_artifacts,
    overlay_heatmap_on_image,
)
from prediction.services.model_registry import PSOModelRegistry
from prediction.services.class_labels import malignant_column, prediction_from_malignancy_probability
from prediction.services.preprocessing import preprocess_image_for_all_models

logger = logging.getLogger(__name__)
ADDITIVITY_TOLERANCE = 1e-4
EXPLANATION_VERSION = "xai-v2"


def _background_features(feature_names=None, max_samples=200):
    """Load a deterministic, finite, schema-aligned validation background."""
    registry = PSOModelRegistry.initialize()
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "pso_cache",
    )
    val_path = os.path.join(cache_dir, "val_probs.npy")
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            "SHAP background data is unavailable: pso_cache/val_probs.npy is missing."
        )

    val_probs_all = np.asarray(np.load(val_path))
    selected_indices = registry.get_selected_indices()
    selected_names = registry.get_selected_model_names()
    selection = registry.get_selection()
    if val_probs_all.ndim != 2:
        raise ValueError(f"Cached validation probabilities must be 2D, got {val_probs_all.shape}")
    if len(selected_indices) != len(selected_names) or any(
        int(index) < 0 or int(index) >= val_probs_all.shape[0] for index in selected_indices
    ):
        raise ValueError("Selected model indices do not match cached probability rows")
    if [selection["model_order"][int(index)] for index in selected_indices] != list(selected_names):
        raise ValueError("Cached probability model ordering does not match selected model names")
    selected = val_probs_all[selected_indices]

    fusion_weights = registry.get_fusion_weights()
    weights = np.asarray([fusion_weights[name] for name in selected_names], dtype=float)

    fused = np.average(selected, axis=0, weights=weights)
    values = {name: selected[i] for i, name in enumerate(selected_names)}
    values["Late Fusion"] = fused
    feature_names = feature_names or registry.get_meta_feature_names()
    background = np.column_stack([values[name] for name in feature_names])

    if background.ndim != 2 or background.shape[1] != len(feature_names):
        raise ValueError("SHAP background dimensions do not match feature schema")
    if not np.isfinite(background).all():
        raise ValueError("SHAP background contains NaN or infinite values")

    if len(background) > max_samples:
        indices = np.linspace(0, len(background) - 1, max_samples, dtype=int)
        background = background[indices]

    return background.astype(float)


def _extract_tree_shap(explanation, feature_count, class_index):
    """Normalize SHAP values and reject unsupported or mismatched shapes."""
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        # Binary classifiers can expose [sample, feature, class].
        if values.shape[-1] <= class_index:
            raise ValueError("SHAP output has no column for the malignant class")
        values = values[0, :, class_index]
    elif values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(f"Expected one explained sample, got shape {values.shape}")
        values = values[0]
    else:
        raise ValueError(f"Unsupported SHAP values shape: {values.shape}")

    if len(values) != feature_count:
        raise ValueError(f"SHAP feature count {len(values)} does not match schema {feature_count}")

    if base_values.ndim >= 2:
        if base_values.shape[-1] <= class_index:
            raise ValueError("SHAP base value has no malignant class column")
        base_value = float(base_values.reshape(base_values.shape[0], -1)[0, class_index])
    elif base_values.ndim == 1:
        if len(base_values) > 1:
            if len(base_values) <= class_index:
                raise ValueError("SHAP base values have no malignant class column")
            base_value = float(base_values[class_index])
        else:
            base_value = float(base_values[0])
    else:
        base_value = float(base_values)

    return values.astype(float), base_value


def explain_meta_learner(model_probabilities, fused_probability, final_probability):
    """
    Explain the GradientBoosting meta-learner using Tree SHAP.

    Positive SHAP values move the model toward malignancy; negative values move
    it toward benign. Probability output is preferred; raw score is used only
    as a compatibility fallback.
    """
    registry = PSOModelRegistry.initialize()
    meta_learner = registry.get_meta_learner()

    feature_names = registry.get_meta_feature_names()
    feature_values_map = {**model_probabilities, "Late Fusion": fused_probability}
    feature_values = np.asarray([[feature_values_map[name] for name in feature_names]], dtype=float)
    expected_count = int(meta_learner.n_features_in_)
    if feature_values.shape != (1, expected_count) or len(feature_names) != expected_count:
        raise ValueError("Runtime SHAP feature count/order does not match meta-learner schema")
    background = _background_features(feature_names)
    if background.shape[1] != expected_count:
        raise ValueError("SHAP background width does not match meta-learner")
    class_index = malignant_column(meta_learner)

    if "Late Fusion" in feature_names:
        # The baseline contains an exact deterministic duplicate. Interventional
        # probability SHAP would independently vary that duplicate, producing
        # impossible inference inputs. Tree-path-dependent SHAP uses training
        # path counts instead and is therefore reported in raw score space.
        feature_perturbation = "tree_path_dependent"
        output_space = "raw_decision_function"
        explainer = shap.TreeExplainer(
            meta_learner,
            model_output="raw",
            feature_perturbation=feature_perturbation,
            feature_names=feature_names,
        )
        explanation = explainer(feature_values, check_additivity=False)
        shap_values, base_value = _extract_tree_shap(explanation, expected_count, class_index)
        if not hasattr(meta_learner, "decision_function"):
            raise ValueError("Raw TreeSHAP output cannot be validated: decision_function unavailable")
        comparison_value = float(np.asarray(meta_learner.decision_function(feature_values)).reshape(-1)[0])
        reconstructed = float(base_value + np.sum(shap_values))
        error = abs(reconstructed - comparison_value)
        additivity_valid = error <= ADDITIVITY_TOLERANCE
        reconstructed_probability = None
        comparison_name = "decision_function"
    else:
        feature_perturbation = "interventional"
        output_space = "probability"
        try:
            explainer = shap.TreeExplainer(
                meta_learner,
                data=background,
                model_output="probability",
                feature_perturbation=feature_perturbation,
                feature_names=feature_names,
            )
            explanation = explainer(feature_values, check_additivity=False)
            shap_values, base_value = _extract_tree_shap(explanation, expected_count, class_index)
            reconstructed = float(base_value + np.sum(shap_values))
            error = abs(reconstructed - float(final_probability))
            additivity_valid = error <= ADDITIVITY_TOLERANCE
            reconstructed_probability = reconstructed
            comparison_name = "predict_proba malignant output"
        except Exception as exc:
            logger.warning("TreeSHAP probability output failed; trying raw output: %s", exc, exc_info=True)
            output_space = "raw_decision_function"
            explainer = shap.TreeExplainer(
                meta_learner,
                data=background,
                model_output="raw",
                feature_perturbation=feature_perturbation,
                feature_names=feature_names,
            )
            explanation = explainer(feature_values, check_additivity=False)
            shap_values, base_value = _extract_tree_shap(explanation, expected_count, class_index)
            if not hasattr(meta_learner, "decision_function"):
                raise ValueError("Raw TreeSHAP output cannot be validated: decision_function unavailable")
            comparison_value = float(np.asarray(meta_learner.decision_function(feature_values)).reshape(-1)[0])
            reconstructed = float(base_value + np.sum(shap_values))
            error = abs(reconstructed - comparison_value)
            additivity_valid = error <= ADDITIVITY_TOLERANCE
            reconstructed_probability = None
            comparison_name = "decision_function"

    contributions = []
    for name, value, shap_value in zip(feature_names, feature_values[0], shap_values):
        contributions.append(
            {
                "feature": name,
                "value": float(value),
                "shap_value": float(shap_value),
                "direction": "toward_malignant" if shap_value >= 0 else "toward_benign",
            }
        )

    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    return {
        "method": "TreeSHAP",
        "output_space": output_space,
        "feature_perturbation": feature_perturbation,
        "target_class": "malignant",
        "base_value": base_value,
        "final_malignancy_probability": float(final_probability),
        "reconstructed_probability": reconstructed_probability,
        "reconstructed_output": reconstructed,
        "additivity_error": float(error),
        "additivity_valid": bool(additivity_valid),
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "additivity_comparison": comparison_name,
        "features": contributions,
        "feature_order": feature_names,
        "interpretation": (
            "Positive SHAP values push the Gradient Boosting decision toward "
            "malignancy; negative values push it toward benign."
        ),
    }


def calculate_model_agreement(model_probabilities, final_prediction):
    """Summarize agreement among the PSO-selected CNNs."""
    votes = []
    probabilities = []

    for model_name, probability in model_probabilities.items():
        label = prediction_from_malignancy_probability(probability)
        probabilities.append(float(probability))
        votes.append(
            {
                "model": model_name,
                "malignancy_probability": float(probability),
                "prediction": label,
                "supports_final_prediction": label == final_prediction,
            }
        )

    support_count = sum(1 for vote in votes if vote["supports_final_prediction"])
    total = len(votes)
    spread = (max(probabilities) - min(probabilities)) if probabilities else 0.0

    return {
        "votes": votes,
        "support_count": support_count,
        "total_models": total,
        "agreement_ratio": float(support_count / total) if total else 0.0,
        "unanimous": bool(total and support_count == total),
        "probability_spread": float(spread),
        "individual_malignancy_probabilities": {
            vote["model"]: vote["malignancy_probability"] for vote in votes
        },
        "individual_predictions": {vote["model"]: vote["prediction"] for vote in votes},
        "final_prediction": final_prediction,
        "supports_final_prediction": support_count,
    }


def _save_overlay(prediction_id, filename, image_array):
    """Persist an explanation image under MEDIA_ROOT and return its media URL."""
    relative_dir = os.path.join("explanations", f"prediction_{prediction_id}")
    absolute_dir = os.path.join(settings.MEDIA_ROOT, relative_dir)
    os.makedirs(absolute_dir, exist_ok=True)

    absolute_path = os.path.join(absolute_dir, filename)
    tf.keras.utils.array_to_img(image_array).save(absolute_path, format="PNG")

    return f"/api/history/{prediction_id}/explanation-image/{filename}/"


def build_visual_explanation(prediction):
    """Generate three class-targeted Grad-CAMs plus a fusion-weighted consensus."""
    if not prediction.uploaded_image or not prediction.uploaded_image.image:
        raise ValueError("Prediction has no uploaded image to explain.")

    registry = PSOModelRegistry.initialize()
    selected_models = registry.get_selected_model_names()
    fusion_weights = registry.get_fusion_weights()

    preprocessed = preprocess_image_for_all_models(
        prediction.uploaded_image.image.path,
        prediction.image_type,
    )

    heatmaps = {}
    model_results = {}
    original_image = None

    for model_name in selected_models:
        batch, original = preprocessed[model_name]
        original_image = original
        artifacts = generate_gradcam_artifacts(
            batch,
            original,
            model_name,
            target_class=prediction.prediction_result,
        )
        heatmap = np.asarray(artifacts["heatmap"], dtype=np.float32)
        if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
            raise ValueError(f"Invalid Grad-CAM heatmap for {model_name}: {heatmap.shape}")
        heatmaps[model_name] = heatmap

        filename = f"{model_name.lower()}_gradcam.png"
        model_results[model_name] = {
            "image_url": _save_overlay(prediction.id, filename, artifacts["overlay"]),
            "last_conv_layer": artifacts["last_conv_layer"],
            "target_class": prediction.prediction_result,
        }

    # CNN backbones can expose Grad-CAM maps at different resolutions.
    # Resize every normalized map to the common displayed image resolution
    # before combining them into an ensemble consensus.
    consensus = np.zeros(original_image.shape[:2], dtype=np.float32)
    total_weight = 0.0
    for model_name in selected_models:
        weight = float(fusion_weights.get(model_name, 0.0))
        resized_heatmap = tf.image.resize(
            np.expand_dims(heatmaps[model_name], axis=-1),
            original_image.shape[:2],
            method="bilinear",
        ).numpy().squeeze()
        if resized_heatmap.shape != original_image.shape[:2]:
            raise ValueError(f"Resized Grad-CAM dimensions do not match image for {model_name}")
        consensus += resized_heatmap.astype(np.float32) * weight
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Fusion weights must have a positive sum for Grad-CAM consensus")
    consensus /= total_weight
    consensus = np.maximum(consensus, 0.0)
    max_value = float(np.max(consensus))
    if max_value > 0:
        consensus /= max_value

    consensus_overlay = overlay_heatmap_on_image(original_image, consensus)
    consensus_url = _save_overlay(
        prediction.id,
        "ensemble_consensus_gradcam.png",
        consensus_overlay,
    )

    return {
        "method": "class_targeted_gradcam",
        "target_class": prediction.prediction_result,
        "models": model_results,
        "consensus": {
            "image_url": consensus_url,
            "method": "fusion_weighted_average",
            "fusion_weights": {
                name: float(fusion_weights[name]) for name in selected_models
            },
        },
        "note": (
            "Grad-CAM highlights model-influential regions and is not a lesion "
            "segmentation or a clinical boundary."
        ),
    }


def build_explanation_for_prediction(prediction):
    """Build the complete cached XAI payload for a saved prediction."""
    model_probabilities = {
        name: float(info["probability"])
        for name, info in prediction.model_breakdown.items()
    }

    return {
        "version": EXPLANATION_VERSION,
        "target_class": prediction.prediction_result,
        "visual_evidence": build_visual_explanation(prediction),
        "model_agreement": calculate_model_agreement(
            model_probabilities,
            prediction.prediction_result,
        ),
        "meta_learner_explanation": explain_meta_learner(
            model_probabilities,
            prediction.fused_probability,
            prediction.prediction_probability,
        ),
        "limitations": [
            "The explanation describes model behaviour, not a pathological diagnosis.",
            "Grad-CAM is a coarse attribution map and should not be interpreted as segmentation.",
            "SHAP explains the Gradient Boosting decision using its trained input features.",
        ],
    }
