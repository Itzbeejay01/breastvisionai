import os
from pathlib import Path
import zipfile
from tqdm import tqdm  # pip install tqdm

# Folder to compress
datasets_folder = Path("datasets")
output_zip = Path("datasets_compressed.zip")

# Gather all files
all_files = [f for f in datasets_folder.rglob("*") if f.is_file()]

# Create zip with progress
with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for file in tqdm(all_files, desc="Compressing datasets", unit="file"):
        # preserve folder structure
        zipf.write(file, arcname=file.relative_to(datasets_folder.parent))

print(f"✅ Dataset compressed successfully: {output_zip.resolve()}")
