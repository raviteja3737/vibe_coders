"""
Dataset utilities for offroad segmentation.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# Class mapping for mask values
CLASS_MAP = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Rocks
    700: 7,      # Logs (CRITICAL)
    800: 8,      # Ground
    7100: 9,     # Vehicle
    10000: 10,   # Other
}

CLASS_NAMES = [
    "Background",
    "Trees",
    "Lush Bushes", 
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Rocks",
    "Logs",
    "Ground",
    "Vehicle",
]


class SegmentationDataset(Dataset):
    """
    Dataset class for offroad segmentation.
    
    Args:
        img_dir: Path to images directory
        mask_dir: Path to masks directory (optional for test)
        size: Target image size (height, width) or single int for square
    """
    
    def __init__(self, img_dir, mask_dir=None, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Handle size as tuple or int
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
            
        self.images = sorted(os.listdir(img_dir))
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size[1], self.size[0]))
        
        # Normalize
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()
        image = self.normalize(image)

        # Load mask if available
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            raw_mask = cv2.resize(raw_mask, (self.size[1], self.size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Convert to class IDs
            clean_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            for k, v in CLASS_MAP.items():
                clean_mask[raw_mask == k] = v

            mask = torch.tensor(clean_mask).long()
            return image, mask

        return image
