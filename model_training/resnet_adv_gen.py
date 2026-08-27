#!/usr/bin/env python3
import os, json, random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

BASE_SPLIT_DIR = Path("datasets_split")
MODEL_PATH = Path("models/resnet50_final_model.keras")  # Update your ResNet50 model path
RESULTS_DIR = Path("results/adversarial/resnet50")
MANIFEST_DIR = RESULTS_DIR / "manifests"
IMG_SIZE = (224, 224)
SAMPLE_PCT = 0.35
SPLITS = ["train", "val", "test"]
EPSILON = 8.0/255.0
ALPHA = 2.0/255.0
ITERS = 40
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

def gather_images(split):
    split_dir = BASE_SPLIT_DIR / split
    if not split_dir.exists(): return [], {}
    labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    mapping = {name: idx for idx, name in enumerate(labels)}
    items = []
    for lbl in labels:
        for p in (split_dir / lbl).iterdir():
            if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg"):
                items.append((f"{lbl}/{p.name}", str(p), mapping[lbl], lbl))
    return items, mapping

def load_preprocess(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img); arr = np.expand_dims(arr, 0)
    return resnet_preprocess(arr.astype(np.float32))

def model_predict(model, x):
    p = model.predict(x, verbose=0); prob = float(np.squeeze(p))
    return int(prob>0.5), prob

def pgd_linf(model, x_pre, y_true, eps=EPSILON, alpha=ALPHA, iters=ITERS):
    x = tf.convert_to_tensor(x_pre); x_adv = tf.identity(x)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            preds = model(x_adv, training=False)
            loss = loss_fn(tf.convert_to_tensor([[y_true]], dtype=tf.float32), preds)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 255)  # ResNet50 expects [0,255] range
    return x_adv.numpy()

def resnet_pre_to_pixels(x_adv_pre):
    x = np.squeeze(x_adv_pre).copy()
    # ResNet50 preprocess: (x - mean)/std → may vary, so clip back to [0,255]
    pixels = np.clip(x, 0, 255).astype('uint8')
    return pixels

def load_manifest(split):
    p = MANIFEST_DIR / f"{split}_manifest.json"
    return json.loads(p.read_text()) if p.exists() else None

def save_manifest(split, data):
    (MANIFEST_DIR / f"{split}_manifest.json").write_text(json.dumps(data, indent=2))

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")
model = load_model(str(MODEL_PATH))

final_report = {"model":"resnet50","config":{"sample_pct":SAMPLE_PCT,"splits":SPLITS,"eps":EPSILON,"alpha":ALPHA,"iters":ITERS},"results":{}}

for split in SPLITS:
    items, mapping = gather_images(split)
    if not items:
        print(f"skip {split} (no images)")
        continue

    manifest = load_manifest(split)
    if manifest is None:
        total = len(items)
        sample_n = max(1, int(total * SAMPLE_PCT))
        indices = random.sample(range(total), sample_n)
        sampled = [items[i] for i in indices]
        manifest = {"sampled": [it[0] for it in sampled], "processed": []}
        save_manifest(split, manifest)
    else:
        sampled = []
        for rel in manifest["sampled"]:
            for it in items:
                if it[0] == rel:
                    sampled.append(it); break

    sample_n = len(manifest["sampled"])
    # build per-class lists
    by_class = {}
    for it in sampled:
        rel, full, lbl_idx, lbl_name = it
        by_class.setdefault(lbl_name, []).append(it)

    out_dir = RESULTS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    attempted = 0; successful = 0; adv_records = []

    class_counts = {k: len(v) for k,v in by_class.items()}
    print(f"\nSplit '{split}': sampled total {sample_n} (by class: {class_counts})")

    for cls_name, lst in by_class.items():
        cls_total = len(lst)
        processed_set = set(manifest.get("processed", []))
        to_proc = [it for it in lst if it[0] not in processed_set]
        pbar = tqdm(total=cls_total, desc=f"{split} {cls_name} 0/{cls_total}")
        already_cls = len([r for r in manifest.get("processed", []) if r.startswith(cls_name + "/")])
        if already_cls>0:
            pbar.update(already_cls)
        for relpath, fullpath, true_label, lblname in to_proc:
            pbar.set_description(f"{split} {cls_name} {len([x for x in manifest.get('processed',[]) if x.startswith(cls_name+'/')])}/{cls_total}")
            x_pre = load_preprocess(fullpath)
            pred, prob = model_predict(model, x_pre)
            if pred != true_label:
                manifest["processed"].append(relpath); save_manifest(split, manifest)
                pbar.update(1); continue
            attempted += 1
            x_adv_pre = pgd_linf(model, x_pre, true_label)
            adv_label, adv_prob = model_predict(model, x_adv_pre)
            success = (adv_label != true_label)
            if success: successful += 1
            pixels = resnet_pre_to_pixels(x_adv_pre)
            save_name = Path(relpath).stem + "_resnet_adv.png"
            save_path = out_dir / save_name
            Image.fromarray(pixels).save(save_path)
            adv_records.append({"relpath":relpath,"saved_at":str(save_path),"true_label":int(true_label),"orig_pred":int(pred),"adv_pred":int(adv_label),"adv_prob":float(adv_prob)})
            manifest["processed"].append(relpath); save_manifest(split, manifest)
            pbar.update(1)
        pbar.close()

    ASR = (successful/attempted) if attempted>0 else None
    res = {"mapping":mapping,"sampled_count":sample_n,"class_counts":class_counts,"attempted":attempted,"successful":successful,"ASR":ASR,"adv_examples":adv_records,"adv_folder":str(out_dir)}
    final_report["results"][split] = res
    with open(RESULTS_DIR / f"resnet50_pgd_{split}_report.json","w") as f:
        json.dump(res,f,indent=2)

with open(RESULTS_DIR / "resnet50_pgd_full_report.json","w") as f:
    json.dump(final_report,f,indent=2)

print("done. reports and adversarials:", RESULTS_DIR)
