"""
stacking_ensemble.py

Final stacking stage.

Pipeline:

    PSO-selected 3 CNNs
            ↓
    Validation/Test predictions
            ↓
    Decision-Level (Late) Fusion
            ↓
    Fused probability + individual probabilities
            ↓
    Gradient Boosting
            ↓
    Final prediction

Gradient Boosting is the ONLY meta-learner.

Fusion strategy:
    Equal-weight decision-level fusion.

    Fused Probability =
        (CNN1 + CNN2 + CNN3) / 3

IMPORTANT:
    - PSO selection is performed separately.
    - This script uses the models selected by PSO.
    - The test set is used ONLY for final evaluation.
    - Gradient Boosting is trained using validation predictions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

CACHE_DIR = "pso_cache"

PSO_RESULT_DIR = "PSO_Result"

OUTPUT_DIR = "Stacking_Result"

SELECTION_FILE = os.path.join(
    PSO_RESULT_DIR,
    "pso_selected_models.json"
)

# Create output directory if it does not exist
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD PSO SELECTION
# ============================================================

print("=" * 70)
print("GRADIENT BOOSTING STACKING ENSEMBLE")
print("=" * 70)

if not os.path.exists(SELECTION_FILE):

    raise FileNotFoundError(
        f"\nPSO selection file not found:\n"
        f"{SELECTION_FILE}\n\n"
        f"Run the PSO selection stage first."
    )


with open(
    SELECTION_FILE,
    "r"
) as f:

    selection = json.load(f)


model_order = selection[
    "model_order"
]

selected_models = selection[
    "selected_models"
]

selected_indices = selection[
    "selected_indices"
]


print(
    "\nPSO-selected models:"
)

for model in selected_models:
    print(
        f"  - {model}"
    )


# ============================================================
# CHECK SELECTED MODEL COUNT
# ============================================================

if len(selected_models) != 3:

    raise ValueError(
        f"Expected exactly 3 PSO-selected models, "
        f"but received {len(selected_models)}."
    )


# ============================================================
# LOAD CACHED PREDICTIONS
# ============================================================

val_probs_all = np.load(
    os.path.join(
        CACHE_DIR,
        "val_probs.npy"
    )
)

test_probs_all = np.load(
    os.path.join(
        CACHE_DIR,
        "test_probs.npy"
    )
)

y_val = np.load(
    os.path.join(
        CACHE_DIR,
        "val_labels.npy"
    )
)

y_test = np.load(
    os.path.join(
        CACHE_DIR,
        "test_labels.npy"
    )
)


# ============================================================
# SELECT PSO-CHOSEN CNN PREDICTIONS
# ============================================================

selected_indices = np.array(
    selected_indices,
    dtype=int
)

selected_val_probs = val_probs_all[
    selected_indices
]

selected_test_probs = test_probs_all[
    selected_indices
]


print(
    "\nSelected prediction shapes:"
)

print(
    f"  Validation: {selected_val_probs.shape}"
)

print(
    f"  Test:       {selected_test_probs.shape}"
)


# ============================================================
# DECISION-LEVEL / LATE FUSION
# ============================================================

print("\n" + "=" * 70)
print("DECISION-LEVEL (LATE) FUSION")
print("=" * 70)

"""
Equal-weight late fusion.

For each sample:

    Fused probability =
        (CNN1 + CNN2 + CNN3) / 3
"""

fusion_weights = np.ones(
    len(selected_models)
) / len(selected_models)


print(
    "\nFusion weights:"
)

for model, weight in zip(
    selected_models,
    fusion_weights
):

    print(
        f"  {model:<20s}: {weight:.4f}"
    )


# ------------------------------------------------------------
# Validation fusion
# ------------------------------------------------------------

val_fused_probability = np.average(
    selected_val_probs,
    axis=0,
    weights=fusion_weights
)


# ------------------------------------------------------------
# Test fusion
# ------------------------------------------------------------

test_fused_probability = np.average(
    selected_test_probs,
    axis=0,
    weights=fusion_weights
)


print(
    "\nFused probability shapes:"
)

print(
    f"  Validation: {val_fused_probability.shape}"
)

print(
    f"  Test:       {test_fused_probability.shape}"
)


# ============================================================
# BUILD META-LEARNER FEATURES
# ============================================================

"""
Gradient Boosting receives:

    CNN 1 probability
    CNN 2 probability
    CNN 3 probability
    Fused probability

Therefore:

    X = [P1, P2, P3, Pfused]
"""

X_val = np.column_stack(
    [
        selected_val_probs.T,
        val_fused_probability
    ]
)

X_test = np.column_stack(
    [
        selected_test_probs.T,
        test_fused_probability
    ]
)


print(
    "\nMeta-learner feature shapes:"
)

print(
    f"  Validation: {X_val.shape}"
)

print(
    f"  Test:       {X_test.shape}"
)


# ============================================================
# GRADIENT BOOSTING META-LEARNER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING GRADIENT BOOSTING META-LEARNER")
print("=" * 70)

meta_learner = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)


print(
    "\nTraining Gradient Boosting..."
)

meta_learner.fit(
    X_val,
    y_val
)


# ============================================================
# TEST PREDICTION
# ============================================================

y_pred = meta_learner.predict(
    X_test
)

y_prob = meta_learner.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

else:

    specificity = 0.0


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy:    {accuracy:.4f}"
)

print(
    f"Precision:   {precision:.4f}"
)

print(
    f"Recall:      {recall:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"F1-score:    {f1:.4f}"
)

print(
    f"AUC-ROC:     {auc:.4f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "stacking_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "PSO-OPTIMIZED GRADIENT BOOSTING STACKING\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Fusion strategy:\n"
    )

    f.write(
        "  Decision-level (Late) Fusion\n"
    )

    f.write(
        "  Equal-weight probability averaging\n\n"
    )

    f.write(
        "Base models selected by PSO:\n"
    )

    for model in selected_models:

        f.write(
            f"  - {model}\n"
        )

    f.write(
        "\nFusion weights:\n"
    )

    for model, weight in zip(
        selected_models,
        fusion_weights
    ):

        f.write(
            f"  {model}: {weight:.6f}\n"
        )

    f.write(
        "\nMeta-learner:\n"
    )

    f.write(
        "  GradientBoostingClassifier\n"
    )

    f.write(
        "\nGradient Boosting parameters:\n"
    )

    f.write(
        "  n_estimators = 100\n"
    )

    f.write(
        "  max_depth = 3\n"
    )

    f.write(
        "  learning_rate = 0.1\n"
    )

    f.write(
        "\nTest-set performance:\n"
    )

    f.write(
        f"  Accuracy:    {accuracy:.6f}\n"
    )

    f.write(
        f"  Precision:   {precision:.6f}\n"
    )

    f.write(
        f"  Recall:      {recall:.6f}\n"
    )

    f.write(
        f"  Specificity: {specificity:.6f}\n"
    )

    f.write(
        f"  F1-score:    {f1:.6f}\n"
    )

    f.write(
        f"  AUC-ROC:     {auc:.6f}\n"
    )

    f.write(
        "\nConfusion matrix:\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# SAVE FUSION INFORMATION
# ============================================================

fusion_info = {

    "fusion_strategy": (
        "decision_level_late_fusion"
    ),

    "fusion_method": (
        "equal_weight_probability_average"
    ),

    "selected_models": selected_models,

    "selected_indices": [
        int(i)
        for i in selected_indices
    ],

    "fusion_weights": {
        model: float(weight)
        for model, weight in zip(
            selected_models,
            fusion_weights
        )
    }
}


fusion_info_path = os.path.join(
    OUTPUT_DIR,
    "fusion_configuration.json"
)

with open(
    fusion_info_path,
    "w"
) as f:

    json.dump(
        fusion_info,
        f,
        indent=4
    )


# ============================================================
# SAVE MODEL IMPORTANCE
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "gb_model_importance.txt"
)

with open(
    importance_path,
    "w"
) as f:

    f.write(
        "Gradient Boosting Feature Importance\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    importances = (
        meta_learner.feature_importances_
    )

    feature_names = (
        selected_models
        + ["Late_Fusion"]
    )

    for feature, importance in zip(
        feature_names,
        importances
    ):

        f.write(
            f"{feature:<20s} "
            f"{importance:.6f}\n"
        )


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"PSO + Late Fusion + GB (AUC={auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve — PSO-Selected Late-Fusion Stacking Ensemble"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "pso_late_fusion_gb_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix — PSO + Late Fusion + Gradient Boosting"
)

plt.colorbar()

class_names = [
    "Negative",
    "Positive"
]

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names
)

plt.yticks(
    tick_marks,
    class_names
)

# Add values inside the matrix
threshold = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=14
        )

plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:"
    f"\n  {confusion_matrix_path}"
)

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    feature_names,
    meta_learner.feature_importances_
)

plt.xlabel(
    "Meta-Learner Feature"
)

plt.ylabel(
    "Feature Importance"
)

plt.title(
    "Gradient Boosting Feature Importance"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

importance_plot_path = os.path.join(
    OUTPUT_DIR,
    "gb_model_importance.png"
)

plt.savefig(
    importance_plot_path,
    dpi=150
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.npy"
    ),
    y_pred
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_probabilities.npy"
    ),
    y_prob
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_fused_probabilities.npy"
    ),
    test_fused_probability
)


# ============================================================
# SAVE VALIDATION FUSION
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "validation_fused_probabilities.npy"
    ),
    val_fused_probability
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STACKING COMPLETE")
print("=" * 70)

print(
    "\nSelected models:"
)

print(
    ", ".join(selected_models)
)

print(
    "\nFusion strategy:"
)

print(
    "Decision-level (Late) Fusion"
)

print(
    "Equal-weight probability averaging"
)

print(
    "\nFinal Accuracy:"
)

print(
    f"{accuracy:.4f}"
)

print(
    "Final F1:"
)

print(
    f"{f1:.4f}"
)

print(
    "Final AUC:"
)

print(
    f"{auc:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    f"  {OUTPUT_DIR}/"
)