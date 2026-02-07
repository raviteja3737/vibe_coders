"""
DINOv2-FPN Segmentation Model
Offroad semantic scene segmentation using DINOv2 backbone with FPN decoder.
"""

import torch
import torch.nn as nn


class FPNDecoder(nn.Module):
    """FPN decoder for DINOv2 backbone."""
    
    def __init__(self, in_channels, num_classes, feature_size=128):
        super().__init__()
        
        self.lateral = nn.Conv2d(in_channels, feature_size, 1)
        
        self.fpn = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(feature_size, feature_size // 2, 3, padding=1),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size // 2, feature_size // 4, 3, padding=1),
            nn.BatchNorm2d(feature_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size // 4, num_classes, 1),
        )

    def forward(self, x, target_size):
        x = self.lateral(x)
        x = self.fpn(x)
        x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.head(x)
        return x


class DINOv2Segmentation(nn.Module):
    """DINOv2-FPN segmentation model for offroad scene understanding."""
    
    def __init__(self, num_classes, backbone_name="dinov2_vitb14_reg"):
        super().__init__()
        
        # Load DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=True)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # FPN decoder
        self.decoder = FPNDecoder(self.backbone.embed_dim, num_classes)
        self.patch_size = 14

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Get patch features from DINOv2
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']
        
        # Reshape to spatial format
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
        
        # Decode to segmentation map
        logits = self.decoder(patch_tokens, (H, W))
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Always keep backbone frozen
        return self


def get_model(num_classes):
    """
    Get segmentation model.
    
    Args:
        num_classes: Number of segmentation classes
        
    Returns:
        DINOv2Segmentation model
    """
    return DINOv2Segmentation(num_classes)
