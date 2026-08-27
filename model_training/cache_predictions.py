"""
Cache predictions ONE MODEL AT A TIME to avoid OOM crashes.
Run this script to cache each model individually.
If a model is already cached, it's skipped.
"""
import os, sys, time, numpy as np, tensorflow as tf, warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.densenet import preprocess_input as den_pre
from tensorflow.keras.applications.resnet import preprocess_input as res_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.xception import preprocess_input as xcep_pre

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

base_dir = 'datasets_split'
cache_dir = 'pso_cache'
os.makedirs(cache_dir, exist_ok=True)

models_info = {
    "EfficientNet": {"path": "models/efficientnet_final_model.keras", "preprocess": eff_pre},
    "DenseNet":    {"path": "models/densenet_final_model.keras",    "preprocess": den_pre},
    "ResNet":      {"path": "models/resnet_final_model.keras",      "preprocess": res_pre},
    "VGG16":       {"path": "models/vgg16_final_model.keras",       "preprocess": vgg_pre},
    "Xception":    {"path": "models/xception_final_model.keras",    "preprocess": xcep_pre},
}
model_order = list(models_info.keys())
img_size = (224, 224)
batch_size = 32

print("=" * 60)
print("CACHING PREDICTIONS (one model at a time)")
print("=" * 60)

for idx, name in enumerate(model_order):
    val_cache = os.path.join(cache_dir, f"{name}_val.npy")
    test_cache = os.path.join(cache_dir, f"{name}_test.npy")

    if os.path.exists(val_cache) and os.path.exists(test_cache):
        print(f"\n  [{idx+1}/5] {name}: Already cached, skipping.")
        continue

    print(f"\n  [{idx+1}/5] {name}")
    info = models_info[name]
    t0 = time.time()

    # Load model
    print(f"    Loading...")
    model = tf.keras.models.load_model(info["path"])
    print(f"    Loaded in {time.time()-t0:.0f}s")

    datagen = ImageDataGenerator(preprocessing_function=info["preprocess"])

    # --- Validation ---
    val_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'val'), target_size=img_size,
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    N_val = len(val_gen.classes) if sum(1 for _ in val_gen.classes) is not None else val_gen.samples
    # Need to recreate generator since we enumerated classes
    val_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'val'), target_size=img_size,
        batch_size=batch_size, class_mode='binary', shuffle=False
    )

    # Save labels on first model
    label_cache = os.path.join(cache_dir, "val_labels.npy")
    if not os.path.exists(label_cache):
        np.save(label_cache, val_gen.classes)

    val_batches = len(val_gen)
    print(f"    Val: {val_gen.samples} images, {val_batches} batches", end="")
    sys.stdout.flush()

    t1 = time.time()
    val_probs = model.predict(val_gen, verbose=0).flatten()
    dt = time.time() - t1
    print(f" -> done in {dt:.0f}s ({val_gen.samples/dt:.0f} img/s)")

    np.save(val_cache, val_probs)

    # --- Test ---
    test_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'test'), target_size=img_size,
        batch_size=batch_size, class_mode='binary', shuffle=False
    )

    label_cache = os.path.join(cache_dir, "test_labels.npy")
    if not os.path.exists(label_cache):
        np.save(label_cache, test_gen.classes)

    test_batches = len(test_gen)
    print(f"    Test: {test_gen.samples} images, {test_batches} batches", end="")
    sys.stdout.flush()

    t1 = time.time()
    test_probs = model.predict(test_gen, verbose=0).flatten()
    dt = time.time() - t1
    print(f" -> done in {dt:.0f}s ({test_gen.samples/dt:.0f} img/s)")

    np.save(test_cache, test_probs)

    # Cleanup
    del model, val_gen, test_gen, val_probs, test_probs
    import gc; gc.collect()

    print(f"    Total: {time.time()-t0:.0f}s | Cached to {cache_dir}/")

# --- Combine individual caches into single arrays ---
print("\n" + "=" * 60)
print("Combining cached predictions...")
print("=" * 60)

val_list = []
test_list = []
for name in model_order:
    v = np.load(os.path.join(cache_dir, f"{name}_val.npy"))
    t = np.load(os.path.join(cache_dir, f"{name}_test.npy"))
    val_list.append(v)
    test_list.append(t)
    print(f"  {name}: val={v.shape}, test={t.shape}")

val_all = np.array(val_list)
test_all = np.array(test_list)

np.save(os.path.join(cache_dir, "val_probs.npy"), val_all)
np.save(os.path.join(cache_dir, "test_probs.npy"), test_all)

print(f"\n  Combined: val_probs={val_all.shape}, test_probs={test_all.shape}")
print(f"  All cached in {cache_dir}/")
print(f"\n  Ready to run: python model_training/pso_model_selection.py")
