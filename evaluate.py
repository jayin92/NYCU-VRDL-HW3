import os
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import skimage.io as sio
import cv2
from PIL import Image
from pycocotools import mask as mask_utils

from model import get_maskrcnn_model, get_maskrcnn_model_fpnv2, get_maskrcnn_model_convnext, get_maskrcnn_model_swin
from dataset import MaskRCNNTestDataset, get_transform, collate_fn
from utils import encode_mask

def mask_to_rle(mask):
    """
    Convert a binary mask to RLE format for COCO
    """
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test dataset
    test_ds = MaskRCNNTestDataset(
        image_dir=args.image_dir,
        transform=get_transform(train=False)
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Initialize model
    if args.backbone == "fpn":
        model = get_maskrcnn_model(num_classes=5, pretrained=False)  # 5 classes: background + 4 classes
    elif args.backbone == "fpnv2":
        model = get_maskrcnn_model_fpnv2(num_classes=5, pretrained=False)  # 5 classes: background + 4 classes
    elif args.backbone == "convnext":
        print("Using ConvNeXt backbone")
        model = get_maskrcnn_model_convnext(num_classes=5, variant=args.variant, pretrained=False)  # 5 classes: background + 4 classes
    elif args.backbone == "swin":
        model = get_maskrcnn_model_swin(num_classes=5, variant=args.variant, pretrained=False)  # 5 classes: background + 4 classes
    else:
        raise ValueError(f"Invalid backbone type: {args.backbone}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate predictions
    results = []
    
    with torch.no_grad():
        for images, image_ids, image_names in tqdm(test_loader, desc="Generating predictions"):
            # Move image to device
            images = list(image.to(device) for image in images)
            
            # Get predictions
            outputs = model(images)
            
            # Process each image
            for i, (output, image_id, image_name) in enumerate(zip(outputs, image_ids, image_names)):
                # Get original image dimensions
                orig_img = sio.imread(os.path.join(args.image_dir, image_name))
                orig_h, orig_w = orig_img.shape[:2]
                
                # Handle different image formats
                if len(orig_img.shape) == 2:  # Grayscale image
                    orig_img = np.stack([orig_img, orig_img, orig_img], axis=2)
                elif orig_img.shape[2] == 4:  # RGBA image
                    # Convert RGBA to RGB
                    alpha = orig_img[:, :, 3:4] / 255.0
                    rgb = orig_img[:, :, :3]
                    background = np.ones_like(rgb) * 255
                    orig_img = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
                
                # Get predicted masks with score > threshold
                scores = output['scores'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                boxes = output['boxes'].cpu().numpy()
                
                # Filter by score threshold
                keep = scores > args.threshold
                scores = scores[keep]
                masks = masks[keep]
                boxes = boxes[keep]
                class_ids = output['labels'].cpu().numpy()[keep]
                
                # Save visualization if requested
                if args.save_images:
                    # Create a visualization of the predictions
                    vis_img = orig_img.copy()
                    
                    # Define fixed colors for each class
                    class_colors = {
                        1: [255, 0, 0],    # Red for class 1
                        2: [0, 255, 0],    # Green for class 2
                        3: [0, 0, 255],    # Blue for class 3
                        4: [255, 255, 0],  # Yellow for class 4
                    }
                    
                    # Draw each mask with a class-specific color
                    for j, (mask, class_id) in enumerate(zip(masks, class_ids)):
                        # Get binary mask - no need to resize as it's already the correct shape
                        binary_mask = mask[0] > 0.5
                        
                        # Get color for this class
                        color = class_colors.get(class_id, [128, 128, 128])  # Default to gray if class not found
                        color = np.array(color, dtype=np.uint8)
                        
                        # Apply mask properly with broadcasting
                        for c in range(3):  # For each color channel (RGB only)
                            vis_img[:,:,c][binary_mask] = vis_img[:,:,c][binary_mask] * 0.5 + color[c] * 0.5
                        
                        # Draw bounding box (no scaling needed)
                        box = boxes[j]
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color.tolist(), 2)
                    
                    # Save visualization
                    cv2.imwrite(
                        os.path.join(args.output_dir, "visualizations", f"pred_{image_name}.png"),
                        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    )
                
                # Process each mask
                for j, (score, mask, box, class_id) in enumerate(zip(scores, masks, boxes, class_ids)):
                    # Get binary mask - no need to resize as it's already the correct shape
                    binary_mask = mask[0] > 0.5
                    
                    # Skip masks that are too small
                    if np.sum(binary_mask) < args.min_area:
                        continue
                    
                    # Convert mask to RLE format
                    rle = mask_to_rle(binary_mask)
                    
                    # Get box coordinates (no need to scale since test images are not resized)
                    x1, y1, x2, y2 = box
                    
                    # Add to results
                    results.append({
                        "image_id": image_id,
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # COCO format: [x, y, width, height]
                        "score": float(score),
                        "category_id": int(class_id),
                        "segmentation": rle
                    })
    
    # Save results to JSON file
    with open(os.path.join(args.output_dir, "test-results.json"), "w") as f:
        json.dump(results, f)
    
    print(f"Saved results to {os.path.join(args.output_dir, 'test-results.json')}")
    print(f"Generated {len(results)} instance predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mask R-CNN model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for predictions")
    parser.add_argument("--backbone", type=str, default="fpn", choices=["fpn", "fpnv2", "convnext", "swin"], help="Backbone type")
    parser.add_argument("--variant", type=str, default="base", choices=["tiny", "small", "base", "large"], help="Variant for ConvNeXt or Swin Transformer")
    parser.add_argument("--threshold", type=float, default=0.0, help="Score threshold for predictions")
    parser.add_argument("--min_area", type=int, default=0, help="Minimum area for a region to be considered")
    parser.add_argument("--save_images", action="store_true", help="Save prediction images")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    main(args)
