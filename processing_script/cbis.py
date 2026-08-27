import os
import shutil
import glob
from pathlib import Path
import pandas as pd

# === CONFIG ===
DEBUG = True
BASE_DIR = Path("datasets/CBIS-DDSM")
CSV_DIR = BASE_DIR / "csv"
JPEG_DIR = BASE_DIR / "jpeg"
OUTPUT_DIR = BASE_DIR / "processed"
SPLITS = ["train", "test"]
DATASETS = {
    "mass": ["mass_case_description_train_set.csv", "mass_case_description_test_set.csv"],
    "calc": ["calc_case_description_train_set.csv", "calc_case_description_test_set.csv"]
}

# Map pathology to folder names (including BENIGN_WITHOUT_CALLBACK)
PATHOLOGY_MAP = {
    "benign": "benign",
    "malignant": "malignant",
    "benign_without_callback": "benign"  # Map this to benign folder
}

# === STATS ===
stats = { "total": 0, "copied": 0, "missing": 0, "uid_folders_found": 0, "uid_folders_missing": 0 }

# === HELPER FUNCTION ===
def extract_uids_from_path(image_path):
    """
    Extract UIDs from CSV image path like:
    Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/000000.dcm
    Returns list of UIDs found in the path
    """
    uids = []
    parts = image_path.split('/')
    for part in parts:
        if part.startswith('1.3.6.1.4.1.9590.100.1.2.'):
            uids.append(part)
    return uids

def find_jpg_files_in_uid_folder(uid):
    """
    Find all .jpg files in a UID folder
    """
    uid_path = JPEG_DIR / uid
    if not uid_path.exists():
        stats["uid_folders_missing"] += 1
        if DEBUG:
            print(f"❌ UID folder not found: {uid}")
        return []
    
    stats["uid_folders_found"] += 1
    jpg_files = list(uid_path.glob("*.jpg"))
    if DEBUG and jpg_files:
        print(f"📁 Found {len(jpg_files)} .jpg files in UID: {uid}")
    return jpg_files

# === PROCESS EACH DATASET ===
for dataset_type, csv_files in DATASETS.items():
    for csv_file in csv_files:
        csv_path = CSV_DIR / csv_file
        if not csv_path.exists():
            print(f"❌ CSV not found: {csv_path}")
            continue

        print(f"📁 Processing {csv_file}...")
        df = pd.read_csv(csv_path)
        
        # Column for image file path
        col = "image file path"
        if col not in df.columns:
            print(f"❌ Column '{col}' not found in CSV: {csv_file}")
            continue

        # Determine split from filename
        split = "test" if "test" in csv_file else "train"

        for idx, row in df.iterrows():
            stats["total"] += 1
            pathology = str(row.get("pathology", "")).lower()
            
            # Map pathology to folder name
            if pathology in PATHOLOGY_MAP:
                folder_name = PATHOLOGY_MAP[pathology]
            else:
                if DEBUG:
                    print(f"⚠️ Unknown pathology: {pathology}")
                stats["missing"] += 1
                continue

            dest_folder = OUTPUT_DIR / split / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)

            image_path = row[col]
            uids = extract_uids_from_path(image_path)
            
            if not uids:
                if DEBUG:
                    print(f"❌ No UIDs found in path: {image_path}")
                stats["missing"] += 1
                continue

            if DEBUG:
                print(f"🔍 Found UIDs: {uids}")

            # Copy all .jpg files from all UID folders
            copied_any = False
            for uid in uids:
                jpg_files = find_jpg_files_in_uid_folder(uid)
                for jpg_file in jpg_files:
                    # Create unique destination filename
                    dest_filename = f"{uid}_{jpg_file.name}"
                    dest_path = dest_folder / dest_filename
                    
                    try:
                        shutil.copy(jpg_file, dest_path)
                        copied_any = True
                        if DEBUG:
                            print(f"✅ Copied: {jpg_file.name} → {dest_filename}")
                    except Exception as e:
                        if DEBUG:
                            print(f"❌ Failed to copy {jpg_file}: {e}")

            if copied_any:
                stats["copied"] += 1
            else:
                if DEBUG:
                    print(f"❌ No .jpg files found for UIDs: {uids}")
                stats["missing"] += 1

# === SUMMARY ===
print("\n=== SUMMARY ===")
print(f"Total images processed: {stats['total']}")
print(f"Successfully copied: {stats['copied']}")
print(f"Missing images: {stats['missing']}")
print(f"UID folders found: {stats['uid_folders_found']}")
print(f"UID folders missing: {stats['uid_folders_missing']}")
