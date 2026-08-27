import os
import shutil
import random
from pathlib import Path

# --------- CONFIGURE ----------
SOURCE_DIR = "datasets"              # folder that currently has benign/ and malignant/
OUTPUT_DIR = "datasets_split"        # new folder to create train/ val/ test/
SPLITS = {'train': 0.75, 'val': 0.15, 'test': 0.10}  # adjust test split to 10% if you like
SEED = 42
# -----------------------------------

random.seed(SEED)

def split_class(class_name: str):
    src_class_dir = Path(SOURCE_DIR) / class_name
    images = [f for f in src_class_dir.iterdir() if f.is_file()]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLITS['train'])
    n_val   = int(n_total * SPLITS['val'])

    splits_idx = {
        'train': images[:n_train],
        'val'  : images[n_train:n_train + n_val],
        'test' : images[n_train + n_val:]
    }

    for split, files in splits_idx.items():
        dest_dir = Path(OUTPUT_DIR) / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), dest_dir)   # ✅ MOVE instead of copy

    print(f"{class_name}: {n_total} -> "
          f"train {len(splits_idx['train'])}, "
          f"val {len(splits_idx['val'])}, "
          f"test {len(splits_idx['test'])}")

def main():
    classes = [d.name for d in Path(SOURCE_DIR).iterdir() if d.is_dir()]
    for c in classes:
        split_class(c)
    print("\n✅ Splitting complete. New structure:")
    print(OUTPUT_DIR + "/train|val|test/<class_name>")

if __name__ == "__main__":
    main()
