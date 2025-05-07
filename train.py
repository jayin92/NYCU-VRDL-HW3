import os
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import argparse
import wandb
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

from model import get_maskrcnn_model, get_maskrcnn_model_fpnv2, get_maskrcnn_model_convnext, get_maskrcnn_model_swin, count_parameters
from dataset import MaskRCNNDataset, get_transform, collate_fn
from utils import encode_mask, decode_maskobj

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None, grad_accum_steps=1):
    model.train()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_mask = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    
    # Reset gradients at the beginning
    optimizer.zero_grad()
    
    loop = tqdm(data_loader)
    
    for i, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        # Normalize the loss to account for gradient accumulation
        losses = losses / grad_accum_steps
        
        if scaler is not None:
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                scaler.scale(losses).backward()
        else:
            # Standard full precision training
            losses.backward()
        
        # Only step and update gradients after accumulating for grad_accum_steps
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Track losses (use the non-normalized loss for logging)
        batch_loss = losses.item() * grad_accum_steps
        running_loss += batch_loss
        running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
        running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
        running_loss_mask += loss_dict['loss_mask'].item() if 'loss_mask' in loss_dict else 0
        running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
        running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
        
        # Update tqdm loop
        loop.set_postfix(loss=losses.item())
        
        if i % print_freq == 0:
            print(f"Epoch: {epoch}, Batch: {i}/{len(data_loader)}, Loss: {losses.item():.4f}")
    
    # Calculate average losses
    epoch_loss = running_loss / len(data_loader)
    epoch_loss_classifier = running_loss_classifier / len(data_loader)
    epoch_loss_box_reg = running_loss_box_reg / len(data_loader)
    epoch_loss_mask = running_loss_mask / len(data_loader)
    epoch_loss_objectness = running_loss_objectness / len(data_loader)
    epoch_loss_rpn_box_reg = running_loss_rpn_box_reg / len(data_loader)
    
    return {
        'loss': epoch_loss,
        'loss_classifier': epoch_loss_classifier,
        'loss_box_reg': epoch_loss_box_reg,
        'loss_mask': epoch_loss_mask,
        'loss_objectness': epoch_loss_objectness,
        'loss_rpn_box_reg': epoch_loss_rpn_box_reg
    }

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    
    # Initialize lists to store predictions and ground truth in COCO format
    coco_gt = {'images': [], 'annotations': [], 'categories': []}
    coco_dt = []
    
    # Setup categories for COCO format
    for i in range(1, 5):  # Classes 1-4
        coco_gt['categories'].append({
            'id': i,
            'name': f'class{i}',
            'supercategory': 'cell'
        })
    
    ann_id = 0
    img_id = 0
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        
        # Get predictions
        outputs = model(images)
        
        for i, (image, target, output) in enumerate(zip(images, targets, outputs)):
            # Image info for COCO format
            h, w = image.shape[-2:]
            coco_gt['images'].append({
                'id': img_id,
                'width': w,
                'height': h,
                'file_name': f'img_{img_id}.jpg'
            })
            
            # Convert ground truth to COCO format
            target_masks = target['masks'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()
            target_boxes = target['boxes'].cpu().numpy()
            
            for mask, label, box in zip(target_masks, target_labels, target_boxes):
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf-8')
                area = mask_util.area(rle).item()
                x1, y1, x2, y2 = box
                
                coco_gt['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': int(label),
                    'segmentation': rle,
                    'area': area,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'iscrowd': 0
                })
                
                ann_id += 1
            
            # Convert predictions to COCO format
            pred_masks = output['masks'].cpu()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            
            for mask, label, score, box in zip(pred_masks, pred_labels, pred_scores, pred_boxes):
                mask = (mask > 0.5).squeeze().numpy().astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                area = mask_util.area(rle).item()
                x1, y1, x2, y2 = box
                
                coco_dt.append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'segmentation': rle,
                    'score': float(score),
                    'area': area,
                    'bbox': [x1, y1, x2 - x1, y2 - y1]
                })
            
            img_id += 1
    
    # Calculate mAP using pycocotools
    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get mAP values
    ap50 = coco_eval.stats[1]  # AP at IoU=0.50
    ap75 = coco_eval.stats[2]  # AP at IoU=0.75
    map = coco_eval.stats[0]   # mAP@[0.5:0.95]
    
    return {
        'mAP': map,
        'AP50': ap50,
        'AP75': ap75
    }

def visualize_predictions(model, data_loader, device, output_dir, num_images=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Define fixed colors for each class
    class_colors = {
        0: [0.5, 0.5, 0.5],  # Gray for background
        1: [1.0, 0.0, 0.0],  # Red for class 1
        2: [0.0, 1.0, 0.0],  # Green for class 2
        3: [0.0, 0.0, 1.0],  # Blue for class 3
        4: [1.0, 1.0, 0.0],  # Yellow for class 4
    }
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_images:
                break
            
            images = list(image.to(device) for image in images)
            
            # Get predictions
            outputs = model(images)
            
            for j, (image, output, target) in enumerate(zip(images, outputs, targets)):
                # Convert image to numpy
                img = image.cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                
                # Get predicted masks and labels with score > 0.5
                keep = output['scores'] > 0.5
                pred_masks = output['masks'][keep]
                pred_masks = (pred_masks > 0.5).squeeze(1).cpu().numpy()
                pred_labels = output['labels'][keep].cpu().numpy()
                
                # Get target masks and labels
                target_masks = target['masks'].cpu().numpy()
                target_labels = target['labels'].cpu().numpy()
                
                # Create figure with 2 rows and 2 columns
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot original image
                axs[0, 0].imshow(img)
                axs[0, 0].set_title('Original Image')
                axs[0, 0].axis('off')
                
                # Plot ground truth masks - FIX: Create separate image for each class to avoid overlap
                gt_img = img.copy()
                axs[0, 1].imshow(gt_img)
                
                # Create a separate mask image for each class
                for class_id in range(1, 5):
                    class_mask = np.zeros_like(gt_img)
                    for mask, label in zip(target_masks, target_labels):
                        if label == class_id:
                            color = class_colors[label.item()]
                            for c in range(3):
                                class_mask[..., c] = np.where(mask == 1, color[c], class_mask[..., c])
                    
                    # Only overlay if there are actual masks for this class
                    if np.any(class_mask > 0):
                        axs[0, 1].imshow(class_mask, alpha=0.5)
                
                # Add legend for ground truth
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=class_colors[i], 
                                            markersize=10, label=f'Class {i}') 
                                for i in range(1, 5)]
                axs[0, 1].legend(handles=legend_elements, loc='upper right')
                axs[0, 1].set_title('Ground Truth Masks')
                axs[0, 1].axis('off')
                
                # Plot predicted masks - using the same fix as above
                pred_img = img.copy()
                axs[1, 0].imshow(pred_img)
                
                # Create a separate mask image for each class
                for class_id in range(1, 5):
                    class_mask = np.zeros_like(pred_img)
                    for mask, label in zip(pred_masks, pred_labels):
                        if label == class_id:
                            color = class_colors[label.item()]
                            for c in range(3):
                                class_mask[..., c] = np.where(mask == 1, color[c], class_mask[..., c])
                    
                    # Only overlay if there are actual masks for this class
                    if np.any(class_mask > 0):
                        axs[1, 0].imshow(class_mask, alpha=0.5)
                
                # Add legend for predictions
                axs[1, 0].legend(handles=legend_elements, loc='upper right')
                axs[1, 0].set_title('Predicted Masks')
                axs[1, 0].axis('off')
                
                # Plot combined view (ground truth and predictions)
                axs[1, 1].imshow(img)
                
                # Add ground truth masks with solid borders
                for mask, label in zip(target_masks, target_labels):
                    # Get contours of the mask
                    if np.any(mask):
                        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                                     cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                        # Draw contours with class color
                        color = class_colors[label.item()]
                        for contour in contours:
                            contour = contour.reshape(-1, 2)
                            axs[1, 1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=2, linestyle='-')
                
                # Add prediction masks with dashed borders
                for mask, label in zip(pred_masks, pred_labels):
                    # Get contours of the mask
                    if np.any(mask):
                        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                                     cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)
                        # Draw contours with class color
                        color = class_colors[label.item()]
                        for contour in contours:
                            contour = contour.reshape(-1, 2)
                            axs[1, 1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=2, linestyle='--')
                
                # Add legend for combined view
                gt_elements = [plt.Line2D([0], [0], color=class_colors[i], linewidth=2,
                                        linestyle='-', label=f'GT Class {i}') 
                              for i in range(1, 5)]
                pred_elements = [plt.Line2D([0], [0], color=class_colors[i], linewidth=2,
                                          linestyle='--', label=f'Pred Class {i}') 
                                for i in range(1, 5)]
                axs[1, 1].legend(handles=gt_elements + pred_elements, loc='upper right', fontsize='small')
                axs[1, 1].set_title('Ground Truth (solid) vs Predictions (dashed)')
                axs[1, 1].axis('off')
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'pred_{i}_{j}.png'), dpi=200)
                plt.close()

def main(args):
    # Initialize wandb if enabled
    if args.use_wandb:
        run_name = f"maskrcnn_{args.backbone}_class{args.class_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "backbone": args.backbone,
                "class_id": args.class_id,
                "trainable_backbone_layers": args.trainable_backbone_layers,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "mixed_precision": args.mixed_precision,
                "grad_accum_steps": args.grad_accum_steps,
            }
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create full dataset
    full_dataset = MaskRCNNDataset(
        data_dir=args.data_dir,
        transform=None,
        num_classes=4  # Use all 4 classes
    )
    
    # Split dataset into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.05)  # 10% for validation
    train_size = dataset_size - val_size
    
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transformations
    train_ds.dataset.transform = get_transform(train=True)
    val_ds.dataset.transform = get_transform(train=False)
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Initialize model
    num_classes = 5
    if args.backbone == 'resnet':
        model = get_maskrcnn_model(
            num_classes=num_classes,
            trainable_backbone_layers=args.trainable_backbone_layers
        )
    elif args.backbone == 'fpnv2':
        model = get_maskrcnn_model_fpnv2(
            num_classes=num_classes,
            trainable_backbone_layers=args.trainable_backbone_layers
        )
    elif args.backbone == 'convnext':
        model = get_maskrcnn_model_convnext(
            num_classes=num_classes,
            variant=args.variant,
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers
        )
    elif args.backbone == 'swin':
        model = get_maskrcnn_model_swin(
            num_classes=num_classes,
            variant=args.variant,
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers
        )
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
        
    # Print model parameter count
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, only_trainable=True):,}")
    print(f"Percentage trainable: {count_parameters(model, only_trainable=True)/count_parameters(model)*100:.2f}%")
    
    print(f"Model has {count_parameters(model)/1e6:.2f}M parameters")
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=0.0005
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=0.0005
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler - Cosine Annealing for smoother decay
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,  # Maximum number of iterations
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Initialize mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Training loop
    best_map = 0.0
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, grad_accum_steps=args.grad_accum_steps
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Validation mAP: {val_metrics['mAP']:.4f}, AP50: {val_metrics['AP50']:.4f}, AP75: {val_metrics['AP75']:.4f}")
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save best model based on mAP
        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved best model with mAP: {best_map:.4f}")
            
            if args.use_wandb:
                # Save model to wandb
                artifact = wandb.Artifact(f"best_model_{wandb.run.id}", type="model")
                artifact.add_file(os.path.join(args.output_dir, "best_model.pth"))
                wandb.log_artifact(artifact)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_map': best_map,
        }, os.path.join(args.output_dir, "checkpoint.pth"))
        
        # Visualize predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(
                model, 
                val_loader, 
                device, 
                os.path.join(args.output_dir, f"predictions_epoch_{epoch+1}"),
                num_images=5
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data folders")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for models and predictions")
    parser.add_argument("--backbone", type=str, default="fpn", choices=["fpn", "fpnv2", "convnext", "swin"], help="Backbone type")
    parser.add_argument("--variant", type=str, default="base", choices=["tiny", "small", "base", "large"], help="Variant for ConvNeXt or Swin Transformer") 
    parser.add_argument("--class_id", type=int, default=2, choices=[1, 2], help="Class ID to use for training (1 or 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--trainable_backbone_layers", type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help="Number of trainable backbone layers")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="Optimizer to use (sgd, adam, or adamw)")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training to reduce memory usage")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="nycu-vrdl-hw3", help="Wandb project name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()