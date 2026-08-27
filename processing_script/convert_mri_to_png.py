import os
from pathlib import Path
import cv2

# Debug mode
DEBUG = True

# Paths
input_dir = Path("datasets/MRI")  # Original MRI folder
output_dir = Path("datasets/MRI/png")  # Folder for resized PNGs
output_dir.mkdir(parents=True, exist_ok=True)

# Categories and splits
categories_mapping = {"Healthy": "benign", "Sick": "malignant"}
splits = ["train", "val"]

for split in splits:
    for original_cat, new_cat in categories_mapping.items():
        src_folder = input_dir / split / original_cat
        if not src_folder.exists():
            print(f"❌ Source folder does not exist: {src_folder}")
            continue

        dest_folder = output_dir / split / new_cat
        dest_folder.mkdir(parents=True, exist_ok=True)

        # List all JPG files
        img_files = list(src_folder.glob("*.jpg"))
        if DEBUG:
            print(f"🔹 Found {len(img_files)} JPG files in {src_folder}")

        if not img_files:
            print(f"⚠️ No JPG files found in {src_folder}")
            continue

        for img_file in img_files:
            # Load grayscale
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Failed to load image: {img_file}")
                continue

            # Resize to 224x224
            img_resized = cv2.resize(img, (224, 224))

            # Save as PNG
            save_path = dest_folder / f"{img_file.stem}.png"
            cv2.imwrite(str(save_path), img_resized)

            if DEBUG:
                print(f"✅ Converted & resized: {img_file.name} → {save_path.name}")

print("✅ MRI conversion script finished!")
