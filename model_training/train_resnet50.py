import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# --- Directory Setup ---
base_dir = 'datasets_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# --- Results Folder ---
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# --- Parameters ---
img_size = (224, 224)   # ResNet expects 224x224
batch_size = 32
epochs = 30

# --- Data Augmentation ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# --- Generators ---
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print("Class distribution in training set:", np.bincount(train_generator.classes))
print("Class distribution in validation set:", np.bincount(validation_generator.classes))

# --- Compute class weights ---
classes = np.unique(train_generator.classes)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_generator.classes
)
class_weights = dict(zip(classes, class_weights_array))
print("Class weights:", class_weights)

# --- Base model ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # Freeze base layers

# --- Full model ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- Compile model ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Checkpoint + EarlyStopping ---
checkpoint_path = os.path.join(results_dir, 'resnet_best_model.weights.h5')
epoch_record_path = os.path.join(results_dir, 'resnet_last_epoch.txt')

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(epoch_record_path, 'w') as f:
            f.write(str(epoch))
        print(f"📝 Last epoch {epoch} saved to {epoch_record_path}")

# --- Load last checkpoint if exists ---
initial_epoch = 0
if os.path.exists(checkpoint_path) and os.path.exists(epoch_record_path):
    model.load_weights(checkpoint_path)
    with open(epoch_record_path, 'r') as f:
        initial_epoch = int(f.read().strip()) + 1
    print(f"✅ Resuming training from epoch {initial_epoch}")
else:
    print("⚠️ No checkpoint found. Starting training from scratch...")

# --- Train model ---
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    initial_epoch=initial_epoch,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, EpochSaver()]
)

# --- Load best weights and save full model ---
model.load_weights(checkpoint_path)
final_model_path = os.path.join(results_dir, 'resnet_final_model.keras')
model.save(final_model_path)
print(f"✅ Final model saved to {final_model_path}")

# --- Evaluate on test set ---
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
specificity = recall_score(y_true, y_pred, pos_label=0)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print("\nTest Set Performance Metrics:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score   : {f1:.4f}")
print(f"AUC-ROC    : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ResNet50 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.grid(True)

roc_plot_path = os.path.join(results_dir, 'roc_curve.png')
plt.savefig(roc_plot_path)
plt.show()
print(f"✅ ROC curve saved to {roc_plot_path}")

# --- Feature Extraction ---
feature_extractor = models.Model(inputs=base_model.input, outputs=layers.GlobalAveragePooling2D()(base_model.output))
features = feature_extractor.predict(test_generator)
features_path = os.path.join(results_dir, 'test_features.npy')
np.save(features_path, features)
print(f"✅ Extracted features saved to {features_path}")
