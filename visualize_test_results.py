import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools import mask as mask_utils
import argparse
from tqdm import tqdm
from PIL import Image
import skimage.io as sio

def load_test_results(json_file):
    """
    Load test results from JSON file
    """
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Group predictions by image_id
    predictions_by_image = {}
    for pred in results:
        image_id = pred['image_id']
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(pred)
    
    return predictions_by_image

def rle_to_mask(rle, height, width):
    """
    Convert RLE to binary mask
    """
    if isinstance(rle['counts'], list):
        # Convert list to string for mask_utils
        rle = mask_utils.frPyObjects(rle, height, width)
    
    mask = mask_utils.decode(rle)
    return mask

def visualize_test_predictions(test_dir, results_file, output_dir, num_images=None, min_score=0.5):
    """
    Visualize test predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    with open('test_image_name_to_ids.json', 'r') as f:
        test_image_name_to_ids = json.load(f)

    mapping = {}
    for item in test_image_name_to_ids:
        mapping[item['file_name']] = item['id']
    
    # Load test results
    predictions_by_image = load_test_results(results_file)
    
    # Get test image folders
    test_folders = sorted([f for f in os.listdir(test_dir)])
    
    # Define colors for visualization
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
    ]
    
    # Limit number of images if specified
    if num_images is not None:
        test_folders = test_folders[:num_images]
    
    # Process each test image
    for i, img_path in enumerate(tqdm(test_folders, desc="Visualizing test predictions")):
        # Load image
        image = np.array(sio.imread(os.path.join(test_dir, img_path)))
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # RGBA image
            # Alpha blending
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3]
            background = np.ones_like(rgb) * 255
            image = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Normalize image for visualization
        img_viz = image.astype(float) / 255.0
        
        # Get image ID (assuming folder name is the image ID)
        image_id = mapping[img_path]
        
        # Get predictions for this image
        predictions = predictions_by_image.get(image_id, [])
        
        # Filter predictions by score
        predictions = [p for p in predictions if p['score'] > min_score]
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        ax[0].imshow(img_viz)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        # Plot image with predictions
        ax[1].imshow(img_viz)
        
        # Process each prediction
        for pred in predictions:
            # Get segmentation mask
            segm = pred['segmentation']
            category_id = pred['category_id']
            score = pred['score']
            
            # Convert RLE to mask
            mask = rle_to_mask(segm, image.shape[0], image.shape[1])
            
            # Get color for this class (1-indexed)
            color = colors[category_id - 1]
            
            # Create colored mask
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
            for c in range(3):
                colored_mask[..., c] = np.where(mask == 1, color[c], 0)
            
            # Overlay mask on image
            ax[1].imshow(colored_mask, alpha=0.5)
            
            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                ax[1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1)
        
        # Add title with prediction count
        ax[1].set_title(f'Predictions (n={len(predictions)})')
        ax[1].axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=colors[i], 
                                      markersize=10, label=f'Class {i+1}') 
                          for i in range(4)]
        ax[1].legend(handles=legend_elements, loc='upper right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{image_id}.png'), dpi=200)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize test predictions")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--results_file", type=str, required=True, help="Path to test results JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to visualize (default: all)")
    parser.add_argument("--min_score", type=float, default=0.5, help="Minimum score threshold for predictions")
    
    args = parser.parse_args()
    
    visualize_test_predictions(
        args.test_dir,
        args.results_file,
        args.output_dir,
        num_images=args.num_images,
        min_score=args.min_score
    )

if __name__ == "__main__":
    main()
