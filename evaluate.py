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

from model import get_maskrcnn_model, get_maskrcnn_model_fpnv2
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
    else:  # fpnv2
        model = get_maskrcnn_model_fpnv2(num_classes=5, pretrained=False)  # 5 classes: background + 4 classes
    
    # Load model weights with handling for different number of classes
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting to load model with different class count...")
        
        # Try to load with 2 classes instead
        if args.backbone == "fpn":
            model = get_maskrcnn_model(num_classes=2, pretrained=False)
        else:  # fpnv2
            model = get_maskrcnn_model_fpnv2(num_classes=2, pretrained=False)
            
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Successfully loaded model with 2 classes.")
        
        # Now recreate the model with 5 classes for future training
        if args.backbone == "fpn":
            new_model = get_maskrcnn_model(num_classes=5, pretrained=False)
        else:  # fpnv2
            new_model = get_maskrcnn_model_fpnv2(num_classes=5, pretrained=False)
            
        # Copy weights that match
        pretrained_dict = model.state_dict()
        model_dict = new_model.state_dict()
        
        # Filter out layers with different shapes
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        
        # Update the model with the pretrained weights
        model_dict.update(pretrained_dict)
        new_model.load_state_dict(model_dict)
        
        # Use the new model
        model = new_model
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
                
                # Get predicted masks with score > threshold
                scores = output['scores'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                boxes = output['boxes'].cpu().numpy()
                
                # Filter by score threshold
                keep = scores > args.threshold
                scores = scores[keep]
                masks = masks[keep]
                boxes = boxes[keep]
                
                # Save visualization if requested
                if args.save_images:
                    # Create a visualization of the predictions
                    vis_img = orig_img.copy()
                    
                    # Draw each mask with a different color
                    for j, mask in enumerate(masks):
                        mask = cv2.resize(mask[0], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        mask = mask > 0.5
                        
                        # Create random color
                        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                        
                        # Apply mask
                        vis_img[mask] = vis_img[mask] * 0.5 + color * 0.5
                        
                        # Draw bounding box
                        box = boxes[j]
                        x1, y1, x2, y2 = box
                        # Scale box to original image size
                        x1 = int(x1 * orig_w / 512)
                        y1 = int(y1 * orig_h / 512)
                        x2 = int(x2 * orig_w / 512)
                        y2 = int(y2 * orig_h / 512)
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color.tolist(), 2)
                    
                    # Save visualization
                    cv2.imwrite(
                        os.path.join(args.output_dir, f"pred_{image_name}.png"),
                        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    )
                
                # Process each mask
                for j, (score, mask, box) in enumerate(zip(scores, masks, boxes)):
                    # Resize mask to original image size
                    mask = cv2.resize(mask[0], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    mask = mask > 0.5
                    
                    # Skip masks that are too small
                    if np.sum(mask) < args.min_area:
                        continue
                    
                    # Convert mask to RLE format
                    rle = mask_to_rle(mask)
                    
                    # Convert box to COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    # Scale box to original image size
                    x1 = float(x1 * orig_w / 512)
                    y1 = float(y1 * orig_h / 512)
                    x2 = float(x2 * orig_w / 512)
                    y2 = float(y2 * orig_h / 512)
                    
                    # Add to results
                    results.append({
                        "image_id": image_id,
                        "bbox": [x1, y1, x2, y2],  # COCO format: [x, y, width, height]
                        "score": float(score),
                        "category_id": 1,  # Always 1 for foreground
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
    parser.add_argument("--backbone", type=str, default="fpn", choices=["fpn", "fpnv2"], help="Backbone type")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for predictions")
    parser.add_argument("--min_area", type=int, default=100, help="Minimum area for a region to be considered")
    parser.add_argument("--save_images", action="store_true", help="Save prediction images")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
