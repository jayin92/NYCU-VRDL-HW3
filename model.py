import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN, LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

def count_parameters(model, only_trainable=False):
    """Count the number of parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        only_trainable (bool): If True, only count trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_maskrcnn_model(num_classes=2, pretrained=False):
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
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    
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

def get_maskrcnn_model_fpnv2(num_classes=2, pretrained=False):
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
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    
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

def get_convnext_fpn_backbone(variant='base', pretrained=False, trainable_backbone_layers=3):
    """
    Create a ConvNeXt + FPN backbone for Mask R-CNN.
    
    Args:
        variant (str): Which ConvNeXt variant to use ('tiny', 'small', 'base', 'large')
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        backbone: ConvNeXt + FPN backbone
    """
    # Select the appropriate ConvNeXt variant
    if variant == 'tiny':
        if pretrained:
            weights = torchvision.models.convnext.ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None
        convnext = torchvision.models.convnext_tiny(weights=weights)
        in_channels_list = [96, 192, 384, 768]  # ConvNeXt Tiny channels
    elif variant == 'small':
        if pretrained:
            weights = torchvision.models.convnext.ConvNeXt_Small_Weights.DEFAULT
        else:
            weights = None
        convnext = torchvision.models.convnext_small(weights=weights)
        in_channels_list = [96, 192, 384, 768]  # ConvNeXt Small channels
    elif variant == 'base':
        if pretrained:
            weights = torchvision.models.convnext.ConvNeXt_Base_Weights.DEFAULT
        else:
            weights = None
        convnext = torchvision.models.convnext_base(weights=weights)
        in_channels_list = [128, 256, 512, 1024]  # ConvNeXt Base channels
    elif variant == 'large':
        if pretrained:
            weights = torchvision.models.convnext.ConvNeXt_Large_Weights.DEFAULT
        else:
            weights = None
        convnext = torchvision.models.convnext_large(weights=weights)
        in_channels_list = [192, 384, 768, 1536]  # ConvNeXt Large channels
    else:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
    
    # Use the features part of the model directly
    backbone = convnext.features
    
    # Freeze layers based on trainable_backbone_layers parameter
    # ConvNeXt has 4 stages, so we'll freeze from the beginning
    num_stages = 4
    stages_to_train = min(trainable_backbone_layers, num_stages)
    
    # First freeze all parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Then unfreeze the last 'trainable_backbone_layers' stages
    if stages_to_train > 0:
        # Stages in ConvNeXt are at indices 1, 3, 5, 7
        stage_indices = [1, 3, 5, 7]
        for i in range(stages_to_train):
            # Get the index of the stage to unfreeze (starting from the end)
            stage_idx = stage_indices[-(i+1)]
            for param in backbone[stage_idx].parameters():
                param.requires_grad = True
    
    # Define which layers to use for FPN
    # For ConvNeXt, we want to use the outputs of each stage
    # The stages are at indices 1, 3, 5, and 7 in the features module
    return_layers = {
        '1': '0',   # First stage output (after stem)
        '3': '1',   # Second stage output
        '5': '2',   # Third stage output
        '7': '3',   # Fourth stage output
    }
    
    # Standard FPN output channels
    out_channels = 256
    
    # Create FPN from ConvNeXt backbone
    fpn_backbone = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    
    return fpn_backbone

def get_swin_fpn_backbone(variant='base', pretrained=False):
    """
    Create a Swin Transformer + FPN backbone for Mask R-CNN.
    
    Args:
        variant (str): Which Swin variant to use ('tiny', 'small', 'base', 'large')
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        backbone: Swin + FPN backbone
    """
    # Select the appropriate Swin Transformer variant
    if variant == 'tiny':
        if pretrained:
            weights = torchvision.models.swin_transformer.Swin_T_Weights.DEFAULT
        else:
            weights = None
        swin = torchvision.models.swin_t(weights=weights)
        in_channels_list = [96, 192, 384, 768]  # Swin-T channels
    elif variant == 'small':
        if pretrained:
            weights = torchvision.models.swin_transformer.Swin_S_Weights.DEFAULT
        else:
            weights = None
        swin = torchvision.models.swin_s(weights=weights)
        in_channels_list = [96, 192, 384, 768]  # Swin-S channels
    elif variant == 'base':
        if pretrained:
            weights = torchvision.models.swin_transformer.Swin_B_Weights.DEFAULT
        else:
            weights = None
        swin = torchvision.models.swin_b(weights=weights)
        in_channels_list = [128, 256, 512, 1024]  # Swin-B channels
    elif variant == 'large':
        # Note: Swin-L has ~197M parameters, very close to the 200M limit
        if pretrained:
            weights = torchvision.models.swin_transformer.Swin_V2_L_Weights.DEFAULT  # Using V2 for latest version
        else:
            weights = None
        swin = torchvision.models.swin_v2_l(weights=weights)
        in_channels_list = [192, 384, 768, 1536]  # Swin-L channels (approximate)
    else:
        raise ValueError(f"Unsupported Swin variant: {variant}")
    
    # Use the features part of the model directly
    backbone = swin.features
    
    # Define which layers to use for FPN
    # For Swin, we want to use the outputs of each stage
    # The stages are at indices 0, 1, 2, 3, 4 in the features module
    return_layers = {
        '0': '0',   # Patch embedding
        '1': '1',   # First stage
        '2': '2',   # Second stage
        '3': '3',   # Third stage
        '4': '4',   # Fourth stage
    }
    
    # Standard FPN output channels
    out_channels = 256
    
    # Create FPN from Swin backbone
    fpn_backbone = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )
    
    return fpn_backbone

def get_maskrcnn_model_convnext(num_classes=5, variant='base', pretrained=False, trainable_backbone_layers=3):
    """
    Get a Mask R-CNN model with a ConvNeXt + FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        variant (str): Which ConvNeXt variant to use ('tiny', 'small', 'base', 'large')
        pretrained (bool): Whether to use pretrained weights for backbone
        
    Returns:
        model: Mask R-CNN model with ConvNeXt backbone
    """
    # Get the ConvNeXt backbone
    backbone = get_convnext_fpn_backbone(
        variant=variant,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
    # Create the Mask R-CNN model with the backbone
    model = torchvision.models.detection.MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=None,  # Use default
        box_roi_pool=None,  # Use default
    )
    
    # Standard Mask R-CNN settings
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def get_maskrcnn_model_swin(num_classes=2, variant='base', pretrained=False):
    """
    Get a Mask R-CNN model with a Swin Transformer + FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        variant (str): Which Swin variant to use ('tiny', 'small', 'base', 'large')
        pretrained (bool): Whether to use pretrained weights for backbone
        
    Returns:
        model: Mask R-CNN model with Swin Transformer backbone
    """
    # Get the Swin Transformer backbone
    backbone = get_swin_fpn_backbone(
        variant=variant,
        pretrained=pretrained
    )
    
    # Create the Mask R-CNN model with the backbone
    model = torchvision.models.detection.MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=None,  # Use default
        box_roi_pool=None,  # Use default
    )
    
    # Standard Mask R-CNN settings
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model