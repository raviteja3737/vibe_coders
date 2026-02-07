"""
Metrics for segmentation evaluation.
"""

import torch
import numpy as np


def iou_score(pred, target, num_classes):
    """
    Compute mean Intersection over Union (mIoU).
    
    Args:
        pred: Model predictions (B, C, H, W)
        target: Ground truth masks (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Mean IoU score
    """
    pred = torch.argmax(pred, dim=1)
    iou = 0.0
    valid_classes = 0

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union > 0:
            iou += intersection / union
            valid_classes += 1

    return iou / max(valid_classes, 1)


def compute_confusion_matrix(pred, target, num_classes):
    """
    Compute confusion matrix for IoU calculation.
    
    Args:
        pred: Predicted class indices (H, W) or (B, H, W)
        target: Ground truth class indices (H, W) or (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target, pred):
        conf_mat[t, p] += 1
        
    return conf_mat


def compute_iou_from_confusion(conf_mat):
    """
    Compute per-class IoU from confusion matrix.
    
    Args:
        conf_mat: Confusion matrix (num_classes, num_classes)
        
    Returns:
        per_class_iou: IoU for each class
        mean_iou: Mean IoU across classes
    """
    intersection = conf_mat.diag()
    union = conf_mat.sum(1) + conf_mat.sum(0) - intersection
    
    # Avoid division by zero
    valid = union > 0
    per_class_iou = torch.zeros(conf_mat.shape[0])
    per_class_iou[valid] = intersection[valid].float() / union[valid].float()
    
    mean_iou = per_class_iou[valid].mean()
    
    return per_class_iou, mean_iou


def pixel_accuracy(pred, target):
    """
    Compute pixel accuracy.
    
    Args:
        pred: Predictions (B, C, H, W) or (B, H, W)
        target: Ground truth (B, H, W)
        
    Returns:
        Pixel accuracy
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).sum().float()
    total = target.numel()
    
    return correct / total
