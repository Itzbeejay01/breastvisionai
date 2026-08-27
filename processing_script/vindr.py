import os
import shutil
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
csv_path = "datasets/metadata.csv"     # path to your CSV
dataset_root = "datasets/vindr"        # folder containing the 5000 study_id folders
output_root = "datasets/vindr_processed_dataset"  # new organized dataset

# === STEP 1. Load CSV ===
df = pd.read_csv(csv_path)

# === STEP 2. Map BI-RADS → benign / malignant ===
def birads_to_label(birads_str):
    num = int(birads_str.split()[-1])
    if num in [1, 2]:
        return "benign"
    elif num in [3, 4, 5]:
        return "malignant"
    else:
        raise ValueError(f"Unexpected BI-RADS: {birads_str}")

df["label"] = df["breast_birads"].apply(birads_to_label)

# === STEP 3. Build full image path ===
df["image_path"] = df.apply(
    lambda row: os.path.join(dataset_root, row["study_id"], f"{row['image_id']}.png"),
    axis=1
)

# === STEP 4. Create output folders ===
for split in ["training", "test"]:
    for label in ["benign", "malignant"]:
        os.makedirs(os.path.join(output_root, split, label), exist_ok=True)

# === STEP 5. Copy images to their new place (skip if exists) ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    src = row["image_path"]
    dst = os.path.join(output_root, row["split"], row["label"], f"{row['image_id']}.png")
    
    if not os.path.exists(src):
        print(f"⚠️ Missing file: {src}")
        continue
    
    # skip if already copied
    if os.path.exists(dst):
        continue
    
    shutil.copy(src, dst)




# for spliting 
import os
import random
import shutil
from sklearn.model_selection import train_test_split

# === CONFIG ===
dataset_root = "datasets/vindr_processed_dataset"
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "val")
val_ratio = 0.15   # 15% of training goes to validation

# === CREATE VAL FOLDERS ===
for label in ["benign", "malignant"]:
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# === SPLIT PER CLASS ===
for label in ["benign", "malignant"]:
    label_train_dir = os.path.join(train_dir, label)
    label_val_dir   = os.path.join(val_dir, label)

    images = os.listdir(label_train_dir)
    
    # Select val set
    _, val_images = train_test_split(
        images, test_size=val_ratio, random_state=42
    )

    # Move selected images
    for img in val_images:
        src = os.path.join(label_train_dir, img)
        dst = os.path.join(label_val_dir, img)
        shutil.move(src, dst)

    print(f"✅ {label}: moved {len(val_images)} images to validation")
