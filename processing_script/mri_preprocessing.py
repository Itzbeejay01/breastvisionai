import os
import cv2
import numpy as np
import pandas as pd
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import filters, morphology, exposure, restoration
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
BASE_DIR = Path("datasets")
MRI_PNG_DIR = BASE_DIR / "MRI/png"
PROCESSED_DIR = BASE_DIR / "processed_mri"

# Classes for MRI dataset
CLASSES = ['benign', 'malignant']

# Test mode - set to False for processing all images
TEST_MODE = False
N_TEST_IMAGES = 5

# Create processed directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
for split in ['train', 'val']:
    for cls in CLASSES:
        os.makedirs(PROCESSED_DIR / split / cls, exist_ok=True)


def extract_brain_region(img):
    """Extract brain region using Otsu thresholding and morphological operations."""
    # Apply Otsu thresholding to separate brain from background
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    
    # Remove small objects
    binary = morphology.remove_small_objects(binary, min_size=100)
    
    # Fill holes using OpenCV morphological operations
    binary_uint8 = binary.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Create mask
    mask = filled
    
    # Apply mask to original image
    return cv2.bitwise_and(img, img, mask=mask)


def denoise_mri_image(img, method='nlm'):
    """Apply specialized denoising for MRI images."""
    if method == 'nlm':
        # Non-local means denoising - very effective for MRI
        return cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)
    elif method == 'bilateral':
        # Bilateral filter - preserves edges
        return cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        # Gaussian blur
        return cv2.GaussianBlur(img, (5, 5), 1.0)
    else:
        return img


def enhance_mri_contrast(img, method='clahe'):
    """Enhance contrast specifically for MRI images."""
    if method == 'clahe':
        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    elif method == 'histogram_equalization':
        # Global histogram equalization
        return cv2.equalizeHist(img)
    elif method == 'adaptive_histogram':
        # Adaptive histogram equalization
        return exposure.equalize_adapthist(img, clip_limit=0.02)
    else:
        return img


def remove_artifacts(img):
    """Remove common MRI artifacts using morphological operations."""
    # Create a structural element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening operation to remove small artifacts
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Closing operation to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def normalize_mri_intensity(img):
    """Normalize MRI intensity values to [0, 255] range."""
    # Clip outliers (common in MRI)
    p2, p98 = np.percentile(img, (2, 98))
    img_clipped = np.clip(img, p2, p98)
    
    # Normalize to [0, 255]
    img_normalized = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_normalized.astype(np.uint8)


def preprocess_mri_image(img: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced preprocessing pipeline specifically designed for MRI images
    """
    original = img.copy()
    
    # 1. Brain region extraction (similar to breast extraction but for brain)
    brain = extract_brain_region(img)
    
    # 2. Denoise using Non-local Means (very effective for MRI)
    denoised = denoise_mri_image(brain, method='nlm')
    
    # 3. Remove artifacts
    cleaned = remove_artifacts(denoised)
    
    # 4. Enhance contrast using CLAHE
    enhanced = enhance_mri_contrast(cleaned, method='clahe')
    
    # 5. Normalize intensity values
    normalized = normalize_mri_intensity(enhanced)
    
    # 6. Apply bilateral filter for final smoothing
    final = cv2.bilateralFilter(normalized, d=9, sigmaColor=50, sigmaSpace=50)
    
    # 7. Resize to 224x224
    target_size = (224, 224)
    resized = cv2.resize(final, target_size, interpolation=cv2.INTER_CUBIC)
    original_resized = cv2.resize(original, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Calculate metrics
    mse = np.mean((original_resized - resized) ** 2)
    psnr_val = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
    ssim_val = ssim(original_resized, resized)
    
    metrics = {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mse': mse
    }
    
    return resized, metrics


def process_mri_dataset():
    """Process MRI dataset"""
    print("📁 Processing MRI dataset...")
    metrics_list = []
    
    for split in ['train', 'val']:
        split_dir = MRI_PNG_DIR / split
        if not split_dir.exists():
            print(f"⚠️ MRI {split} directory not found: {split_dir}")
            continue
            
        for cls in CLASSES:
            class_dir = split_dir / cls
            processed_class_dir = PROCESSED_DIR / split / cls
            
            if not class_dir.exists():
                print(f"⚠️ MRI {split}/{cls} directory not found: {class_dir}")
                continue
            
            # Get all PNG files
            image_files = list(class_dir.glob("*.png"))
            
            if not image_files:
                print(f"⚠️ No images found in {class_dir}")
                continue
            
            # Limit to first N images in test mode
            if TEST_MODE:
                image_files = image_files[:N_TEST_IMAGES]
                print(f"🧪 TEST MODE: Processing first {len(image_files)} images")
            
            print(f"📁 Processing MRI {split}/{cls} ({len(image_files)} images)...")
            
            with tqdm(total=len(image_files), desc=f"MRI-{split}-{cls}", unit="img") as pbar:
                for img_file in image_files:
                    try:
                        # Read image
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            logging.warning(f"Could not read {img_file}")
                            continue
                        
                        # Preprocess
                        processed_img, metrics = preprocess_mri_image(img)
                        
                        # Save processed image
                        out_path = processed_class_dir / img_file.name
                        cv2.imwrite(str(out_path), processed_img)
                        
                        # Save metrics
                        metrics_list.append({
                            'dataset': 'MRI',
                            'split': split,
                            'class': cls,
                            'filename': f'{split}/{cls}/{img_file.name}',
                            'psnr': metrics['psnr'],
                            'ssim': metrics['ssim'],
                            'mse': metrics['mse']
                        })
                        
                        pbar.set_postfix({
                            'psnr': f"{metrics['psnr']:.2f}",
                            'ssim': f"{metrics['ssim']:.4f}"
                        })
                        
                    except Exception as e:
                        logging.error(f"Failed to process {img_file}: {e}")
                    
                    pbar.update(1)
    
    return metrics_list


def save_separate_metrics(all_metrics):
    """Save separate metrics files for each dataset and split"""
    if not all_metrics:
        return
    
    df = pd.DataFrame(all_metrics)
    
    # Add timestamp to the data
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save comprehensive CSV with all data
    comprehensive_filename = results_dir / f"mri_preprocessing_metrics_{timestamp}.csv"
    df.to_csv(comprehensive_filename, index=False)
    print(f"✅ Saved comprehensive metrics: {comprehensive_filename} ({len(df)} total images)")
    
    # Save separate files for each dataset and split
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        for split in dataset_df['split'].unique():
            split_df = dataset_df[dataset_df['split'] == split]
            
            # Create filename with timestamp
            filename = results_dir / f"{dataset.lower()}_{split}_metrics_{timestamp}.csv"
            
            # Save to CSV
            split_df.to_csv(filename, index=False)
            print(f"✅ Saved {filename} ({len(split_df)} images)")
            
            # Print summary for this split
            print(f"📊 {dataset} {split}:")
            print(f"  Images: {len(split_df)}")
            print(f"  Avg PSNR: {split_df['psnr'].mean():.2f}")
            print(f"  Avg SSIM: {split_df['ssim'].mean():.4f}")
            print(f"  Avg MSE: {split_df['mse'].mean():.2f}")
            print()
    
    # Print overall summary
    print(f"📊 OVERALL SUMMARY:")
    print(f"  Total Images: {len(df)}")
    print(f"  Overall Avg PSNR: {df['psnr'].mean():.2f}")
    print(f"  Overall Avg SSIM: {df['ssim'].mean():.4f}")
    print(f"  Overall Avg MSE: {df['mse'].mean():.2f}")
    print()
    
    # Save summary statistics
    summary_stats = {
        'total_images': len(df),
        'overall_avg_psnr': df['psnr'].mean(),
        'overall_avg_ssim': df['ssim'].mean(),
        'overall_avg_mse': df['mse'].mean(),
        'timestamp': timestamp,
        'pipeline_version': 'enhanced_mri_with_brain_extraction'
    }
    
    # Add per-dataset and per-split statistics
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        summary_stats[f'{dataset.lower()}_avg_psnr'] = dataset_df['psnr'].mean()
        summary_stats[f'{dataset.lower()}_avg_ssim'] = dataset_df['ssim'].mean()
        summary_stats[f'{dataset.lower()}_avg_mse'] = dataset_df['mse'].mean()
        
        for split in dataset_df['split'].unique():
            split_df = dataset_df[dataset_df['split'] == split]
            summary_stats[f'{dataset.lower()}_{split}_avg_psnr'] = split_df['psnr'].mean()
            summary_stats[f'{dataset.lower()}_{split}_avg_ssim'] = split_df['ssim'].mean()
            summary_stats[f'{dataset.lower()}_{split}_avg_mse'] = split_df['mse'].mean()
    
    # Save summary as JSON
    import json
    summary_filename = results_dir / f"mri_preprocessing_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Saved summary statistics: {summary_filename}")


def main():
    """Main processing function"""
    if TEST_MODE:
        print("🧪 TEST MODE: Processing first 5 images from each category")
    else:
        print("🚀 Starting full MRI preprocessing for all images...")
    
    all_metrics = []
    
    # Process MRI dataset
    mri_metrics = process_mri_dataset()
    all_metrics.extend(mri_metrics)
    
    # Save separate metrics files
    if all_metrics:
        save_separate_metrics(all_metrics)
        print(f"📊 Total images processed: {len(all_metrics)}")
    else:
        print("❌ No images were processed!")


if __name__ == "__main__":
    main()
