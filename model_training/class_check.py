import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Paths ---
base_dir = "datasets_split"   # change if needed
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

# --- Data generators ---
datagen = ImageDataGenerator(rescale=1./255)

# --- Function to check classes ---
def check_classes(directory, subset_name):
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',   # make sure binary for 2 classes
        shuffle=False
    )
    print(f"\n{subset_name} set info:")
    print("Class indices:", generator.class_indices)
    counts = {cls: 0 for cls in generator.class_indices.keys()}
    for label in generator.classes:
        # get class name from index
        for k,v in generator.class_indices.items():
            if v == label:
                counts[k] += 1
                break
    print("Images per class:", counts)
    print("Total images:", len(generator.classes))

# --- Run checks ---
check_classes(train_dir, "Training")
check_classes(val_dir, "Validation")
check_classes(test_dir, "Test")
