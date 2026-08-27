import os
import sys
from pathlib import Path
import pandas as pd
import pydicom
import cv2
import glob

# Resolve project root (one level up from this script's directory)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# Paths (resolved against repo root)
csv_path = ROOT_DIR / "datasets/INBreast/INbreast.csv"  # CSV with filename/id
dicom_dir = ROOT_DIR / "datasets/INBreast/raw"          # DICOM folder
png_dir = ROOT_DIR / "datasets/INBreast/png"            # PNG output folder

# Validate inputs
if not csv_path.exists():
    print(f"❌ CSV not found at: {csv_path}. Please place INbreast.csv there.")
    sys.exit(1)
if not dicom_dir.exists():
    print(f"❌ DICOM directory not found at: {dicom_dir}. Please create it and add DICOM files.")
    sys.exit(1)

# Ensure PNG folder exists
png_dir.mkdir(parents=True, exist_ok=True)

# Load CSV with semicolon separator
df = pd.read_csv(csv_path, sep=';')
print(f"ℹ️ CSV columns: {list(df.columns)}")

# Look for "File Name" column specifically
if "File Name" in df.columns:
    id_column = "File Name"
    print(f"ℹ️ Using 'File Name' column for DICOM matching")
else:
    # Fallback to first column if "File Name" not found
    id_column = df.columns[0]
    print(f"ℹ️ 'File Name' column not found. Using first column: '{id_column}'")

# Convert DICOM to PNG, preserving original DICOM filenames when possible
converted_count = 0
missing_count = 0
for _, row in df.iterrows():
    csv_id = str(row[id_column]).strip()
    if not csv_id or csv_id == "nan":
        continue

    # Try several matching strategies for the DICOM file ID
    patterns = [
        f"{csv_id}_*.dcm",     # prefix with underscore
        f"*{csv_id}*.dcm",     # anywhere in name
        f"{csv_id}.dcm"        # exact match
    ]

    dicom_files = []
    for pat in patterns:
        dicom_files = glob.glob(str(dicom_dir / pat))
        if dicom_files:
            break

    # If csv_id looks like a filename with .dcm, try direct path
    if not dicom_files and csv_id.lower().endswith(".dcm"):
        direct = dicom_dir / csv_id
        if direct.exists():
            dicom_files = [str(direct)]

    if not dicom_files:
        print(f"❌ No DICOM found for CSV ID: {csv_id}")
        missing_count += 1
        continue

    dicom_path = dicom_files[0]  # take first match
    dicom_file_name = os.path.basename(dicom_path)

    # Read DICOM
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array

    # Resize to 224x224
    img_resized = cv2.resize(img, (224, 224))

    # Normalize to [0,255] uint8
    img_resized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Save PNG with original DICOM filename
    png_file_name = dicom_file_name.replace(".dcm", ".png")
    png_path = png_dir / png_file_name
    cv2.imwrite(str(png_path), img_resized)

    converted_count += 1
    print(f"✅ Converted: {dicom_file_name} → {png_file_name}")

print(f"✅ Conversion complete. Converted: {converted_count}, Missing: {missing_count}")
