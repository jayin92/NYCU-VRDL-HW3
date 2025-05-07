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

from model import get_maskrcnn_model, get_maskrcnn_model_fpnv2
from dataset import MaskRCNNDataset, get_transform, collate_fn
from utils import encode_mask, decode_maskobj

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_mask = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    
    loop = tqdm(data_loader)
    
    for i, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_classifier'].item()
        running_loss_box_reg += loss_dict['loss_box_reg'].item()
        running_loss_mask += loss_dict['loss_mask'].item()
        running_loss_objectness += loss_dict['loss_objectness'].item()
        running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
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
    
    # Placeholder for metrics
    # In a real evaluation, you would calculate mAP, IoU, etc.
    # For simplicity, we'll just count correct detections
    
    total_masks = 0
    correct_masks = 0
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Get predictions
        outputs = model(images)
        
        # Compare predictions with targets
        for i, (output, target) in enumerate(zip(outputs, targets)):
            # Get predicted masks with score > 0.5
            pred_masks = output['masks'][output['scores'] > 0.5]
            pred_masks = (pred_masks > 0.5).squeeze(1)
            
            # Get target masks
            target_masks = target['masks']
            
            # Count masks
            total_masks += len(target_masks)
            
            # For each target mask, find the best matching predicted mask
            for target_mask in target_masks:
                if len(pred_masks) == 0:
                    continue
                
                # Calculate IoU for each predicted mask
                ious = []
                for pred_mask in pred_masks:
                    intersection = (pred_mask & target_mask).sum().float()
                    union = (pred_mask | target_mask).sum().float()
                    iou = intersection / (union + 1e-8)
                    ious.append(iou)
                
                # If any predicted mask has IoU > 0.5, count as correct
                if max(ious) > 0.5:
                    correct_masks += 1
    
    # Calculate accuracy
    mask_accuracy = correct_masks / total_masks if total_masks > 0 else 0
    
    return {'mask_accuracy': mask_accuracy}

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
                
                # Plot ground truth masks for all classes
                axs[0, 1].imshow(img)
                for mask, label in zip(target_masks, target_labels):
                    # Get color for this class
                    color = class_colors[label.item()]
                    # Create colored mask
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                    for c in range(3):
                        colored_mask[..., c] = np.where(mask == 1, color[c], 0)
                    # Overlay mask on image
                    axs[0, 1].imshow(colored_mask, alpha=0.5)
                
                # Add legend for ground truth
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=class_colors[i], 
                                            markersize=10, label=f'Class {i}') 
                                for i in range(1, 5)]
                axs[0, 1].legend(handles=legend_elements, loc='upper right')
                axs[0, 1].set_title('Ground Truth Masks')
                axs[0, 1].axis('off')
                
                # Plot predicted masks for all classes
                axs[1, 0].imshow(img)
                for mask, label in zip(pred_masks, pred_labels):
                    # Get color for this class
                    color = class_colors[label.item()]
                    # Create colored mask
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
                    for c in range(3):
                        colored_mask[..., c] = np.where(mask == 1, color[c], 0)
                    # Overlay mask on image
                    axs[1, 0].imshow(colored_mask, alpha=0.5)
                
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
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
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
    val_size = int(dataset_size * 0.1)  # 10% for validation
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
    if args.backbone == "fpn":
        model = get_maskrcnn_model(num_classes=5, pretrained=True, trainable_backbone_layers=args.trainable_backbone_layers)  # 5 classes: background + 4 classes
    else:  # fpnv2
        model = get_maskrcnn_model_fpnv2(num_classes=5, pretrained=True, trainable_backbone_layers=args.trainable_backbone_layers)  # 5 classes: background + 4 classes
    
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
    else:  # adam
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=0.0005
        )
    
    # Learning rate scheduler - Cosine Annealing for smoother decay
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,  # Maximum number of iterations
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {val_metrics['mask_accuracy']:.4f}")
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_metrics['mask_accuracy'] > best_accuracy:
            best_accuracy = val_metrics['mask_accuracy']
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")
            
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
            'best_accuracy': best_accuracy,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data folders")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for models and predictions")
    parser.add_argument("--backbone", type=str, default="fpn", choices=["fpn", "fpnv2"], help="Backbone type")
    parser.add_argument("--class_id", type=int, default=2, choices=[1, 2], help="Class ID to use for training (1 or 2)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--trainable_backbone_layers", type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help="Number of trainable backbone layers")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer to use (sgd or adam)")
    
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
