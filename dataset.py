import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import skimage.io as sio
import torchvision.transforms as T
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from skimage.measure import label as skimage_label, regionprops

class MaskRCNNDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_classes=4):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes  # Number of classes (1-4)
        self.folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index):
        folder_name = self.folders[index]
        folder_path = os.path.join(self.data_dir, folder_name)
        
        # Read image
        img_path = os.path.join(folder_path, "image.tif")
        image = np.array(sio.imread(img_path))
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # RGBA image
            # Use alpha blending with white background
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3]
            background = np.ones_like(rgb) * 255  # White background
            image = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Initialize empty masks list and class labels
        masks = []
        labels = []
        
        # Check each class mask
        for class_id in range(1, self.num_classes + 1):
            mask_path = os.path.join(folder_path, f"class{class_id}.tif")
            if os.path.exists(mask_path):
                mask = np.array(sio.imread(mask_path))
                # Convert mask to binary (0 or 1)
                mask = (mask > 0).astype(np.uint8)
                
                # If mask has any foreground pixels, add it
                if np.any(mask):
                    masks.append(mask)
                    labels.append(class_id)  # Use class_id as the label
        
        # If no masks were found, create a dummy mask for background
        if len(masks) == 0:
            masks = [np.zeros(image.shape[:2], dtype=np.uint8)]
            labels = [0]  # Background class
        
        # Apply transformations if provided
        if self.transform is not None:
            # Apply transformation to image first
            transformed = self.transform(image=image)
            image_transformed = transformed["image"]
            
            # Apply same transformation to each mask separately
            transformed_masks = []
            for mask in masks:
                # Make sure image is numpy array for albumentations
                if isinstance(image, torch.Tensor):
                    image_np = image.numpy().transpose(1, 2, 0)
                else:
                    image_np = image
                    
                transformed = self.transform(image=image_np, mask=mask)
                transformed_masks.append(transformed["mask"])
            
            image = image_transformed
            masks = transformed_masks
        else:
            # Convert to tensor manually if no transform
            image = T.ToTensor()(image)
            masks = [torch.from_numpy(mask) for mask in masks]
        
        # Find instances (connected components) in the mask
        # Process each mask to get instances and bounding boxes
        boxes = []
        final_masks = []
        final_labels = []
        
        for i, (mask, class_label) in enumerate(zip(masks, labels)):
            # Find connected components in the mask
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
                
            labeled_mask = skimage_label(mask_np)
            props = regionprops(labeled_mask)
            
            for prop in props:
                y1, x1, y2, x2 = prop.bbox
                # Convert to the format expected by Mask R-CNN [x1, y1, x2, y2]
                boxes.append([x1, y1, x2, y2])
                
                # Create a binary mask for this instance
                instance_mask = (labeled_mask == prop.label).astype(np.uint8)
                final_masks.append(instance_mask)
                
                # Use the class label for this instance
                final_labels.append(class_label)
        
        # If no instances were found, create a dummy box and mask
        if len(boxes) == 0:
            boxes.append([0, 0, 1, 1])  # Dummy box
            final_masks.append(np.zeros(image.shape[:2], dtype=np.uint8))
            final_labels.append(0)  # Background
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Convert list of masks to a single numpy array first to avoid warning
        final_masks = np.array(final_masks)
        final_masks = torch.as_tensor(final_masks, dtype=torch.uint8)
        final_labels = torch.as_tensor(final_labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': final_labels,
            'masks': final_masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        return image, target

class MaskRCNNTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        
        with open('test_image_name_to_ids.json', 'r') as f:
            self.image_name_to_id = json.load(f)
        self.mapping = {item['file_name']: item['id'] for item in self.image_name_to_id}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Extract image_id from filename (without extension)
        image_id = self.mapping[img_name]
        
        # Read image
        image = np.array(sio.imread(img_path))
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:  # RGBA image
            # Use alpha blending with white background
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3]
            background = np.ones_like(rgb) * 255  # White background
            image = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Apply transformations if provided
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        return image, image_id, img_name

def get_transform(train=True):
    transforms = []
    # Convert to tensor and normalize
    transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms.append(ToTensorV2())
    
    if train:
        # Add training transforms
        transforms = [
            A.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    
    return A.Compose(transforms)

def collate_fn(batch):
    """
    Custom collate function for Mask R-CNN data loader
    """
    return tuple(zip(*batch))
