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
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Initialize lists for boxes, masks, and labels
        boxes = []
        all_masks = []
        labels = []
        
        # Process each class
        for class_id in range(1, self.num_classes + 1):
            mask_path = os.path.join(folder_path, f"class{class_id}.tif")
            if os.path.exists(mask_path):
                # Load the mask for this class
                class_mask = np.array(sio.imread(mask_path))
                
                # Skip if mask is empty
                if not np.any(class_mask):
                    continue
                
                # Find unique instance IDs (excluding 0, which is background)
                instance_ids = np.unique(class_mask)
                instance_ids = instance_ids[instance_ids > 0]
                
                # Process each instance in this class
                for instance_id in instance_ids:
                    # Create binary mask for this instance
                    instance_mask = (class_mask == instance_id).astype(np.uint8)
                    
                    # Calculate bounding box for this instance
                    y_indices, x_indices = np.where(instance_mask > 0)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue  # Skip if no pixels in the mask
                    
                    # Get bbox coordinates [x1, y1, x2, y2]
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices) + 1, np.max(y_indices) + 1
                    
                    # Add this instance's data
                    boxes.append([x1, y1, x2, y2])
                    all_masks.append(instance_mask)
                    labels.append(class_id)
        
        # If no instances were found, create a dummy box and mask
        if len(boxes) == 0:
            boxes.append([0, 0, 1, 1])  # Dummy box
            all_masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
            labels.append(0)  # Background class
        
        # Apply transformations if provided
        if self.transform is not None:
            # Prepare the format for Albumentations
            # The format is [x1, y1, x2, y2, class_id]
            transformed_boxes = []
            for box, label in zip(boxes, labels):
                transformed_boxes.append(box + [label])
            
            # Apply transformations to image, masks, and bboxes
            transformed = self.transform(
                image=image,
                masks=all_masks,
                bboxes=transformed_boxes,
                category_ids=labels
            )
            
            # Extract transformed data
            image = transformed['image']
            all_masks = transformed['masks']
            transformed_boxes = transformed['bboxes']
            
            # Extract boxes and labels from transformed_boxes
            if transformed_boxes:
                boxes = [box[:4] for box in transformed_boxes]
                labels = [int(box[4]) for box in transformed_boxes]
            else:
                # If all boxes were removed during transformation (e.g., crop cut them out)
                boxes = [[0, 0, 1, 1]]
                all_masks = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)]
                labels = [0]  # Background class
        else:
            # Convert to tensor manually if no transform
            image = T.ToTensor()(image)
            all_masks = [torch.from_numpy(mask) for mask in all_masks]
        
        # Convert everything to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.array(all_masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
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
        
        # Extract image_id from mapping
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
    if train:
        # Training transformations with bounding box support
        return A.Compose([
            # A.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(p=1.0),
                A.RandomGamma(p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0)
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=2, p=1.0)
            ], p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
            ToTensorV2(),
            # A.RandomRotate90(p=0.5),
            # A.OneOf([
            #     A.CLAHE(p=1.0),
            #     A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            #     A.MedianBlur(blur_limit=5, p=1.0)
            # ], p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.OneOf([
            #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            #     A.GridDistortion(p=1.0),
            #     A.OpticalDistortion(distort_limit=2, p=1.0)
            # ], p=0.3),
            # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
            # ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # [x1, y1, x2, y2]
            label_fields=['category_ids'],
        ))
    else:
        # Validation/test transformations (no augmentations)
        return A.Compose([
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2()
        ])

def collate_fn(batch):
    """
    Custom collate function for Mask R-CNN data loader
    """
    return tuple(zip(*batch))