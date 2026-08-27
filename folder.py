import os

# Base datasets folder
datasets = ['INBreast', 'CBIS-DDSM', 'MRI']
base_path = 'datasets'  # adjust if your folder path is different

# Subfolders to create for each dataset
subfolders = [
    'raw', 'png', 'preprocessed',
    os.path.join('train', 'benign'),
    os.path.join('train', 'malignant'),
    os.path.join('val', 'benign'),
    os.path.join('val', 'malignant'),
    os.path.join('test', 'benign'),
    os.path.join('test', 'malignant')
]

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created main folder: {dataset_path}")
    
    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created: {folder_path}")

print("✅ All subfolders have been created successfully!")
