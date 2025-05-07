import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_model(num_classes=2, pretrained=False, trainable_backbone_layers=3):
    """
    Get a Mask R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights on COCO
        
    Returns:
        model: Mask R-CNN model
    """
    # Load an instance segmentation model pre-trained on COCO
    if pretrained:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def get_maskrcnn_model_fpnv2(num_classes=2, pretrained=False, trainable_backbone_layers=3):
    """
    Get a Mask R-CNN model with a ResNet-50-FPN-v2 backbone (better performance).
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights on COCO
        
    Returns:
        model: Mask R-CNN model
    """
    # Load an instance segmentation model pre-trained on COCO
    if pretrained:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    else:
        weights = None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model
