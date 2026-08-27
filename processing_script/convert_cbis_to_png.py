import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
processed_dir = Path("datasets/CBIS-DDSM/processed")   # processed dataset from cbis.py
png_dir = Path("datasets/CBIS-DDSM/png")   # target folder for resized PNGs

# Desired image size
IMG_SIZE = (224, 224)

# Ensure PNG directory exists
png_dir.mkdir(parents=True, exist_ok=True)

# Check if processed directory exists
if not processed_dir.exists():
    print(f"❌ Processed directory not found: {processed_dir}")
    print("Please run the CBIS processing script first (cbis.py)")
    exit(1)

total_processed = 0
total_failed = 0

# Process each split and category
for split in ["train", "test"]:
    split_dir = processed_dir / split
    if not split_dir.exists():
        print(f"⚠️ Split directory not found: {split_dir}")
        continue
        
    for category in ["benign", "malignant"]:
        source_path = split_dir / category
        target_path = png_dir / split / category
        
        if not source_path.exists():
            print(f"⚠️ Category directory not found: {source_path}")
            continue
            
        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Processing {split}/{category}...")
        
        count = 0
        failed = 0
        
        # Get all image files
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg")) + list(source_path.glob("*.png"))
        
        if not image_files:
            print(f"  ⚠️ No image files found in {source_path}")
            continue
        
        # Progress bar for this category
        with tqdm(total=len(image_files), desc=f"{split}/{category}", unit="img") as pbar:
            for img_file in image_files:
                try:
                    # Read image with OpenCV (much faster than PIL)
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        raise ValueError(f"Could not read image: {img_file}")
                    
                    # Resize to 224x224
                    img_resized = cv2.resize(img, IMG_SIZE)
                    
                    # Normalize to [0,255] uint8 (ensure consistent format)
                    img_resized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

                    # Save as PNG with same name
                    dst = target_path / f"{img_file.stem}.png"
                    cv2.imwrite(str(dst), img_resized)

                    count += 1
                    pbar.set_postfix({"success": count, "failed": failed})

                except Exception as e:
                    failed += 1
                    pbar.set_postfix({"success": count, "failed": failed})
                    print(f"\n⚠️ Failed to process {img_file.name}: {e}")
                
                pbar.update(1)

        total_processed += count
        total_failed += failed
        print(f"✅ {split}/{category} → {count} images resized and saved. Failed: {failed}")

print(f"\n=== SUMMARY ===")
print(f"Total images processed: {total_processed}")
print(f"Total failed: {total_failed}")
print(f"PNG files saved to: {png_dir}")
