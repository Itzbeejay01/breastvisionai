import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Model-specific preprocessing ---
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.densenet import preprocess_input as den_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.xception import preprocess_input as xcep_pre

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# =========================================================
# PATHS
# =========================================================
base_dir = 'datasets_split'
test_dir = os.path.join(base_dir, 'test')

output_dir = 'Ensemble_Result'
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# PARAMETERS
# =========================================================
img_size = (224, 224)
batch_size = 32

# =========================================================
# MODELS + PREPROCESSING
# =========================================================
models_info = {
    "EfficientNet": {
        "path": "models/efficientnet_final_model.keras",
        "preprocess": eff_pre
    },
    "DenseNet": {
        "path": "models/densenet_final_model.keras",
        "preprocess": den_pre
    },
    "ResNet": {
        "path": "models/resnet_final_model.keras",
        "preprocess": res_pre
    },
    "VGG16": {
        "path": "models/vgg16_final_model.keras",
        "preprocess": vgg_pre
    },
    "Xception": {
        "path": "models/xception_final_model.keras",
        "preprocess": xcep_pre
    }
}

# =========================================================
# GET PREDICTIONS FROM ALL MODELS
# =========================================================
all_probs = []
y_true = None

for name, info in models_info.items():
    print(f"\n🔹 Processing {name}...")

    model = tf.keras.models.load_model(info["path"])

    datagen = ImageDataGenerator(preprocessing_function=info["preprocess"])

    generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    if y_true is None:
        y_true = generator.classes

    probs = model.predict(generator).flatten()
    all_probs.append(probs)

all_probs = np.array(all_probs)  # shape (5, N)

# =========================================================
# INDIVIDUAL MODEL ROC (FOR PLOTTING)
# =========================================================
model_fprs = {}
model_tprs = {}
model_aucs = {}

for name, probs in zip(models_info.keys(), all_probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    model_fprs[name] = fpr
    model_tprs[name] = tpr
    model_aucs[name] = auc

# =========================================================
# ENSEMBLES
# =========================================================

# --- HARD VOTING (≥3 malignant) ---
hard_preds_matrix = (all_probs > 0.5).astype(int)
hard_vote = (np.sum(hard_preds_matrix, axis=0) >= 3).astype(int)

# --- SOFT VOTING ---
soft_probs = np.mean(all_probs, axis=0)
soft_preds = (soft_probs > 0.5).astype(int)

# --- WEIGHTED AVERAGING ---
weights = np.array([0.25, 0.20, 0.20, 0.15, 0.20])
weighted_probs = np.average(all_probs, axis=0, weights=weights)
weighted_preds = (weighted_probs > 0.5).astype(int)

# =========================================================
# METRICS FUNCTION
# =========================================================
def compute_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return accuracy, precision, recall, specificity, f1, auc

hard_metrics = compute_metrics(y_true, hard_vote, hard_vote)
soft_metrics = compute_metrics(y_true, soft_preds, soft_probs)
weighted_metrics = compute_metrics(y_true, weighted_preds, weighted_probs)

# =========================================================
# PRINT + SAVE METRICS
# =========================================================
def print_metrics(name, m):
    print(f"\n{name}:")
    print(f"Accuracy   : {m[0]:.4f}")
    print(f"Precision  : {m[1]:.4f}")
    print(f"Recall     : {m[2]:.4f}")
    print(f"Specificity: {m[3]:.4f}")
    print(f"F1-Score   : {m[4]:.4f}")
    print(f"AUC-ROC    : {m[5]:.4f}")

print_metrics("Hard Voting", hard_metrics)
print_metrics("Soft Voting", soft_metrics)
print_metrics("Weighted Averaging", weighted_metrics)

with open(os.path.join(output_dir, "ensemble_metrics.txt"), "w") as f:
    for name, m in zip(
        ["Hard Voting", "Soft Voting", "Weighted Averaging"],
        [hard_metrics, soft_metrics, weighted_metrics]
    ):
        f.write(f"\n{name}:\n")
        f.write(f"Accuracy   : {m[0]:.4f}\n")
        f.write(f"Precision  : {m[1]:.4f}\n")
        f.write(f"Recall     : {m[2]:.4f}\n")
        f.write(f"Specificity: {m[3]:.4f}\n")
        f.write(f"F1-Score   : {m[4]:.4f}\n")
        f.write(f"AUC-ROC    : {m[5]:.4f}\n")

# =========================================================
# PLOTTING FUNCTION
# =========================================================
def plot_ensemble(title, ensemble_probs, name, save_path):
    plt.figure()

    # Individual models
    for model_name in model_fprs:
        plt.plot(
            model_fprs[model_name],
            model_tprs[model_name],
            label=f'{model_name} (AUC={model_aucs[model_name]:.4f})'
        )

    # Ensemble
    fpr_e, tpr_e, _ = roc_curve(y_true, ensemble_probs)
    auc_e = roc_auc_score(y_true, ensemble_probs)

    plt.plot(
        fpr_e, tpr_e,
        linewidth=3,
        linestyle='--',
        label=f'{name} (AUC={auc_e:.4f})'
    )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

# =========================================================
# GENERATE 3 PLOTS
# =========================================================
plot_ensemble(
    "Hard Voting vs Individual Models",
    hard_vote,
    "Hard Voting",
    os.path.join(output_dir, "hard_voting_roc.png")
)

plot_ensemble(
    "Soft Voting vs Individual Models",
    soft_probs,
    "Soft Voting",
    os.path.join(output_dir, "soft_voting_roc.png")
)

plot_ensemble(
    "Weighted Averaging vs Individual Models",
    weighted_probs,
    "Weighted Averaging",
    os.path.join(output_dir, "weighted_voting_roc.png")
)

print(f"\n✅ All results saved in: {output_dir}")