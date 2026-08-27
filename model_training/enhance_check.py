import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

# --- Path to your dataset ---
base_dir = "datasets_split"
splits = ["train", "val", "test"]
classes = ["benign", "malignant"]

def analyze_image(img_path):
    # Load image in grayscale for analysis
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Create a slightly degraded version (blurred) for comparison
    degraded = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate metrics
    psnr_val = psnr(img, degraded, data_range=255)
    ssim_val = ssim(img, degraded, data_range=255)
    mse_val = mse(img, degraded)

    return psnr_val, ssim_val, mse_val

# --- Check first 5 images in each split/class ---
for split in splits:
    for cls in classes:
        folder = os.path.join(base_dir, split, cls)
        images = sorted(os.listdir(folder))[:5]  # take first 5
        print(f"\nChecking {split}/{cls}:")
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            result = analyze_image(img_path)
            if result:
                psnr_val, ssim_val, mse_val = result
                print(f"{img_name} -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.2f}")
            else:
                print(f"{img_name} -> Could not read image")
