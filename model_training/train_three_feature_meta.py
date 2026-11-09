"""Train and report an experimental three-CNN-feature meta-learner.

The four-feature GradientBoosting artifact remains the deployed baseline. This
script trains a separate candidate from validation predictions only, reports
validation fit metrics, and then reports one untouched test evaluation. Test
results must not be used to tune or select the candidate.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.services.class_labels import malignant_column


CACHE = ROOT / "pso_cache"
RESULTS = ROOT / "Stacking_Result"
SELECTION = json.loads((ROOT / "PSO_Result/pso_selected_models.json").read_text())
MODELS = SELECTION["selected_models"]
BASELINE_PATH = RESULTS / "pso_stacked_gb_model.joblib"
CANDIDATE_PATH = RESULTS / "pso_stacked_gb_three_feature_candidate.joblib"
SCHEMA_PATH = RESULTS / "three_feature_candidate_schema.json"
METRICS_PATH = RESULTS / "three_feature_candidate_evaluation.json"


def malignancy_probability(model, features):
    return model.predict_proba(features)[:, malignant_column(model)]


def metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probability)),
    }


def main():
    labels_val = np.load(CACHE / "val_labels.npy")
    labels_test = np.load(CACHE / "test_labels.npy")
    all_val = np.load(CACHE / "val_probs.npy")
    all_test = np.load(CACHE / "test_probs.npy")
    indices = np.asarray(SELECTION["selected_indices"], dtype=int)
    x_val = all_val[indices].T
    x_test = all_test[indices].T
    if x_val.shape != (len(labels_val), len(MODELS)):
        raise ValueError(f"Unexpected validation shape: {x_val.shape}")
    if x_test.shape != (len(labels_test), len(MODELS)):
        raise ValueError(f"Unexpected test shape: {x_test.shape}")
    if not np.isfinite(x_val).all() or not np.isfinite(x_test).all():
        raise ValueError("Cached CNN probability features contain non-finite values")

    candidate = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    candidate.fit(x_val, labels_val)
    joblib.dump(candidate, CANDIDATE_PATH)
    SCHEMA_PATH.write_text(json.dumps({
        "version": "candidate-three-cnn-probabilities-v1",
        "features": MODELS,
        "training_split": "validation",
        "deployed": False,
    }, indent=2) + "\n")

    baseline = joblib.load(BASELINE_PATH)
    if int(baseline.n_features_in_) != 4:
        raise ValueError("Expected the current deployed baseline to use four features")
    x_val_four = np.column_stack([x_val, x_val.mean(axis=1)])
    x_test_four = np.column_stack([x_test, x_test.mean(axis=1)])

    report = {
        "selection_policy": "No test-set tuning or selection; both artifacts are reported side-by-side.",
        "label_mapping": {
            "documented_by_training_sources": {"0": "benign", "1": "malignant"},
            "semantic_verification": "not independently verified: datasets_split/ is absent",
        },
        "feature_schemas": {
            "baseline": [*MODELS, "Late Fusion"],
            "candidate": MODELS,
        },
        "validation": {
            "baseline": metrics(labels_val, malignancy_probability(baseline, x_val_four)),
            "candidate": metrics(labels_val, malignancy_probability(candidate, x_val)),
            "interpretation": "Training split for the candidate; not an independent evaluation.",
        },
        "test": {
            "baseline": metrics(labels_test, malignancy_probability(baseline, x_test_four)),
            "candidate": metrics(labels_test, malignancy_probability(candidate, x_test)),
            "interpretation": "Final held-out report only; do not use these results to tune or select.",
        },
        "candidate_artifact": CANDIDATE_PATH.name,
        "candidate_deployed": False,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
