import os
import shutil
import pandas as pd

# ==== CONFIGURATION ====
csv_path = "datasets/INBreast/INbreast.csv"   # your CSV file
image_dir = "datasets/INBreast/png"           # folder where PNGs are stored
output_dir = "datasets/INBreast"              # output directory for organized dataset

# Train/Validation split ratio
train_ratio = 0.8

# ==== LOAD METADATA ====
df = pd.read_csv(csv_path, sep=';')  # Use semicolon separator

# Clean BI-RADS labels (convert 4a/4b/4c → 4)
df["Bi-Rads"] = df["Bi-Rads"].astype(str).str[0]   # take only the first char
df["Bi-Rads"] = df["Bi-Rads"].astype(int)

# Define benign/malignant mapping
# BI-RADS 1,2 = benign; BI-RADS 3,4,5,6 = malignant
df["label"] = df["Bi-Rads"].apply(lambda x: "benign" if x in [1,2] else "malignant")

# ==== MATCH IMAGES ====
# Images are like: 24055203_606e9b184978a350_MG_L_CC_ANON.png
# CSV has "File Name" column (e.g., 24055203)
id_to_label = dict(zip(df["File Name"].astype(str), df["label"]))

# Collect matched files
matched = []
for fname in os.listdir(image_dir):
    if fname.lower().endswith(".png"):
        # Extract the ID (take everything before first "_")
        img_id = fname.split("_")[0]
        if img_id in id_to_label:
            matched.append((fname, id_to_label[img_id]))

print(f"✅ Matched {len(matched)} images out of {len(os.listdir(image_dir))}")

# ==== TRAIN/TEST SPLIT ====
from sklearn.model_selection import train_test_split

files, labels = zip(*matched)
train_files, val_files, train_labels, val_labels = train_test_split(
    files, labels, stratify=labels, test_size=(1-train_ratio), random_state=42
)

# ==== FUNCTION TO COPY ====
def copy_files(file_list, label_list, split):
    for fname, label in zip(file_list, label_list):
        src = os.path.join(image_dir, fname)
        dst_dir = os.path.join(output_dir, split, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, fname))

# Copy train & validation
copy_files(train_files, train_labels, "train")
copy_files(val_files, val_labels, "val")

print("✅ Dataset organized successfully!")
print(f"Train: {len(train_files)} images | Validation: {len(val_files)} images")
