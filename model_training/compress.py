import os
import zipfile
from tqdm import tqdm

# Path to dataset split
dataset_dir = "datasets_split"
output_zip = "datasets_split.zip"

# Collect all files
all_files = []
for root, _, files in os.walk(dataset_dir):
    for file in files:
        filepath = os.path.join(root, file)
        arcname = os.path.relpath(filepath, start=dataset_dir)  # relative path inside zip
        all_files.append((filepath, arcname))

print(f"Found {len(all_files)} files to compress.")

# Create zip with progress bar
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filepath, arcname in tqdm(all_files, desc="Compressing", unit="files"):
        zipf.write(filepath, arcname)

print(f"\n✅ Dataset compressed successfully into: {output_zip}")
