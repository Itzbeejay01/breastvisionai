import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# --- Paths ---
base_dir = 'datasets_split'
test_dir = os.path.join(base_dir, 'test')

model_path = 'results/resnet_final_model.keras'
output_dir = 'ResNet_Result'

os.makedirs(output_dir, exist_ok=True)

# --- Parameters ---
img_size = (224, 224)
batch_size = 32

# --- Load Model ---
model = tf.keras.models.load_model(model_path)
print("ResNet50 model loaded successfully.")

# --- Test Data ---
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# --- Predictions ---
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
specificity = recall_score(y_true, y_pred, pos_label=0)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

# --- Save Metrics ---
metrics_text = f"""
ResNet50 Test Set Performance Metrics:

Accuracy   : {accuracy:.4f}
Precision  : {precision:.4f}
Recall     : {recall:.4f}
Specificity: {specificity:.4f}
F1-Score   : {f1:.4f}
AUC-ROC    : {auc:.4f}
"""

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(metrics_text)

print(metrics_text)

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'ResNet50 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Benign', 'Malignant']
)

disp.plot()
plt.title("Confusion Matrix")

plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print(f"\nAll ResNet results saved in: {output_dir}")