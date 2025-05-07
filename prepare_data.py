import os
import shutil
import numpy as np
import skimage.io as sio
from sklearn.model_selection import train_test_split
import argparse

def prepare_data(sample_dir, output_dir, test_size=0.2, random_state=42):
    """
    Prepare data for training and validation.
    
    Args:
        sample_dir: Directory containing sample images
        output_dir: Directory to save prepared data
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create output directories
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_mask_dir = os.path.join(output_dir, "train", "masks")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_mask_dir = os.path.join(output_dir, "val", "masks")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    # Find image and mask files
    image_path = os.path.join(sample_dir, "image.tif")
    mask_path = os.path.join(sample_dir, "class2.tif")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Image or mask not found in {sample_dir}")
        return
    
    # Load image and mask
    image = sio.imread(image_path)
    mask = sio.imread(mask_path)
    
    # Create patches from the image and mask
    patch_size = 256
    stride = 128
    
    patches_img = []
    patches_mask = []
    patch_indices = []
    
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Only keep patches with some mask content
            if np.sum(mask_patch) > 0:
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)
                patch_indices.append((y, x))
    
    # Split patches into train and validation sets
    indices = np.arange(len(patches_img))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # Save train patches
    for i, idx in enumerate(train_indices):
        img_patch = patches_img[idx]
        mask_patch = patches_mask[idx]
        y, x = patch_indices[idx]
        
        sio.imsave(os.path.join(train_img_dir, f"patch_{i}_y{y}_x{x}.tif"), img_patch)
        sio.imsave(os.path.join(train_mask_dir, f"patch_{i}_y{y}_x{x}.tif"), mask_patch)
    
    # Save validation patches
    for i, idx in enumerate(val_indices):
        img_patch = patches_img[idx]
        mask_patch = patches_mask[idx]
        y, x = patch_indices[idx]
        
        sio.imsave(os.path.join(val_img_dir, f"patch_{i}_y{y}_x{x}.tif"), img_patch)
        sio.imsave(os.path.join(val_mask_dir, f"patch_{i}_y{y}_x{x}.tif"), mask_patch)
    
    print(f"Created {len(train_indices)} training patches and {len(val_indices)} validation patches")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--sample_dir", type=str, default="./sample-image", help="Directory containing sample images")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save prepared data")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_data(args.sample_dir, args.output_dir, args.test_size, args.random_state)
