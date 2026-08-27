import os
import random
import shutil

# Paths
base_dir = "datasets/MRI"   # change to your dataset root
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Create test/benign and test/malignant
for cls in ["benign", "malignant"]:
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Split ratio
test_split = 0.0571  # 20% goes to testidation

for cls in ["benign", "malignant"]:
    cls_train_dir = os.path.join(train_dir, cls)
    cls_test_dir = os.path.join(test_dir, cls)

    images = os.listdir(cls_train_dir)
    random.shuffle(images)

    test_size = int(len(images) * test_split)
    test_images = images[:test_size]

    print(f"Moving {len(test_images)} {cls} images to testidation folder...")

    for img in test_images:
        src = os.path.join(cls_train_dir, img)
        dst = os.path.join(cls_test_dir, img)
        shutil.move(src, dst)

print("✅ Dataset split complete!")
