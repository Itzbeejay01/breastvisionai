import os
import cv2
import numpy as np
import pandas as pd
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
BASE_DIR = Path("datasets")
VINDR_DIR = BASE_DIR / "vindr_dataset"
PROCESSED_DIR = BASE_DIR / "vindr_processed_dataset"
# Classes for both datasets
CLASSES = ['benign', 'malignant']

# Test mode - set to True for first 5 images only
TEST_MODE = False
N_TEST_IMAGES = 5

# Create processed directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
for split in ['train', 'val', 'test']:
    for cls in CLASSES:
        os.makedirs(PROCESSED_DIR / "vindr" / split / cls, exist_ok=True)


def extract_breast_region(img):
    """Mask out background by simple threshold and keep largest component."""
    # Assume img is grayscale uint8
    # Rough threshold to separate breast from black background
    _, thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)  
    # Keep largest connected component (the breast)
    labels = cv2.connectedComponentsWithStats(thresh)[1]
    if labels.max() == 0:
        return img  # fallback
    # Determine largest CC (excluding background label 0)
    largest_label = 1 + np.argmax(cv2.connectedComponentsWithStats(thresh)[2][1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8) * 255
    # Optional: remove pectoral muscle by cutting top-right corner (for MLO views)
    # mask = remove_pectoral(mask)  # implement Hough or other if needed
    return cv2.bitwise_and(img, img, mask=mask)

def denoise_image(img, method='median'):
    """Apply denoising filter: 'median', 'gaussian', 'bilateral', or 'nlm'."""
    if method == 'median':
        return cv2.medianBlur(img, ksize=3)
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, (5,5), sigmaX=1.0)
    elif method == 'bilateral':
        # d=9, sigmaColor=75, sigmaSpace=75 as common defaults
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'nlm':
        # h=10 is a starting point; convert to 32F for fastNlMeans
        img32 = img.astype(np.uint8)
        return cv2.fastNlMeansDenoising(img32, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        return img

def enhance_contrast(img, clipLimit=2.0, tileGridSize=(8,8)):
    """Apply CLAHE for local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

def sharpen_image(img):
    """Apply unsharp masking (USM) sharpening."""
    # Gaussian blur then combine
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    # Weighted sum: original + (original - blurred)
    sharpen = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return np.clip(sharpen, 0, 255).astype(np.uint8)

def normalize_intensity(img):
    """Normalize image to zero mean and unit variance (float32 output)."""
    img_float = img.astype(np.float32)
    mean, std = img_float.mean(), img_float.std()
    if std > 0:
        img_norm = (img_float - mean) / std
    else:
        img_norm = img_float - mean
    return img_norm

def preprocess_mammography_image(img: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced preprocessing pipeline for mammography images with breast region extraction
    """
    original = img.copy()
    
    # 1. Breast region extraction
    breast = extract_breast_region(img)
    
    # 2. First-pass denoising (median + NLM)
    med = denoise_image(breast, method='median')
    nlm = denoise_image(med, method='nlm')
    
    # 3. Contrast enhancement (CLAHE)
    clahe_img = enhance_contrast(nlm, clipLimit=2.0, tileGridSize=(8,8))
    
    # 4. Sharpen
    sharp = sharpen_image(clahe_img)
    
    # 5. Normalize intensities
    normalized = normalize_intensity(sharp)
    
    # 6. Convert back to uint8 for saving and metrics calculation
    processed = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    
    # 7. Resize to 224x224
    target_size = (224, 224)
    resized = cv2.resize(processed, target_size, interpolation=cv2.INTER_CUBIC)
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


def process_vindr_dataset():
    """Process VINDR dataset"""
    print("📁 Processing VINDR dataset...")
    metrics_list = []
    
    for split in ['train', 'val', 'test']:
        split_dir = VINDR_DIR / split
        if not split_dir.exists():
            print(f"⚠️ VINDR {split} directory not found: {split_dir}")
            continue
            
        for cls in CLASSES:
            class_dir = split_dir / cls
            processed_class_dir = PROCESSED_DIR / "vindr" / split / cls
            
            if not class_dir.exists():
                print(f"⚠️ VINDR {split}/{cls} directory not found: {class_dir}")
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
            
            print(f"📁 Processing VINDR {split}/{cls} ({len(image_files)} images)...")
            
            with tqdm(total=len(image_files), desc=f"VINDR-{split}-{cls}", unit="img") as pbar:
                for img_file in image_files:
                    try:
                        # Read image
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            logging.warning(f"Could not read {img_file}")
                            continue
                        
                        # Preprocess
                        processed_img, metrics = preprocess_mammography_image(img)
                        
                        # Save processed image
                        out_path = processed_class_dir / img_file.name
                        cv2.imwrite(str(out_path), processed_img)
                        
                        # Save metrics
                        metrics_list.append({
                            'dataset': 'vindr',
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
    comprehensive_filename = results_dir / f"vindr_preprocessing_metrics_{timestamp}.csv"
    df.to_csv(comprehensive_filename, index=False)
    print(f"✅ Saved comprehensive metrics: {comprehensive_filename} ({len(df)} total images)")
    
    # Save separate files for each dataset and split
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        for split in dataset_df['split'].unique():
            split_df = dataset_df[dataset_df['split'] == split]
            
            # Create filename with timestamp
            filename = results_dir / f"{dataset.lower().replace('-', '_')}_{split}_metrics_{timestamp}.csv"
            
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
        'pipeline_version': 'enhanced_with_breast_extraction'
    }
    
    # Add per-dataset and per-split statistics
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        summary_stats[f'{dataset.lower().replace("-", "_")}_avg_psnr'] = dataset_df['psnr'].mean()
        summary_stats[f'{dataset.lower().replace("-", "_")}_avg_ssim'] = dataset_df['ssim'].mean()
        summary_stats[f'{dataset.lower().replace("-", "_")}_avg_mse'] = dataset_df['mse'].mean()
        
        for split in dataset_df['split'].unique():
            split_df = dataset_df[dataset_df['split'] == split]
            summary_stats[f'{dataset.lower().replace("-", "_")}_{split}_avg_psnr'] = split_df['psnr'].mean()
            summary_stats[f'{dataset.lower().replace("-", "_")}_{split}_avg_ssim'] = split_df['ssim'].mean()
            summary_stats[f'{dataset.lower().replace("-", "_")}_{split}_avg_mse'] = split_df['mse'].mean()
    
    # Save summary as JSON
    import json
    summary_filename = results_dir / f"preprocessing_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Saved summary statistics: {summary_filename}")


def main():
    """Main processing function"""
    if TEST_MODE:
        print("🧪 TEST MODE: Processing first 5 images from each category")
        print("🚀 Starting full vindr preprocessing...")
    else:
        print("🚀 Starting full vindr preprocessing...")
    
    all_metrics = []
    
    # Process VINDR
    vindr_metrics = process_vindr_dataset()
    all_metrics.extend(vindr_metrics)
    

    
    # Save separate metrics files
    if all_metrics:
        save_separate_metrics(all_metrics)
        print(f"📊 Total images processed: {len(all_metrics)}")
    else:
        print("❌ No images were processed!")


if __name__ == "__main__":
    main()
