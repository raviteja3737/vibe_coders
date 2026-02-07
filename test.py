"""
Test/Benchmark Script for Offroad Segmentation
Evaluates trained model on test dataset.

Usage:
    python test.py
"""

import os
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm

from models.segformer import get_model
from utils.dataset import SegmentationDataset, CLASS_NAMES
from utils.metrics import iou_score, compute_confusion_matrix, compute_iou_from_confusion


def main():
    print("🧪 TESTING SCRIPT STARTED 🧪")
    
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = get_model(cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load("segformer.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
    
    # Test dataset
    test_ds = SegmentationDataset(
        img_dir="dataset/test/images",
        mask_dir="dataset/test/masks",
        size=(cfg["image_height"], cfg["image_width"])
    )
    
    print(f"Test samples: {len(test_ds)}")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Evaluate
    total_conf_mat = torch.zeros(cfg["num_classes"], cfg["num_classes"], dtype=torch.long)
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_ds, desc="Testing")):
            img = img.unsqueeze(0).to(device)
            mask = mask.to(device)
            
            pred = model(img)
            pred_mask = torch.argmax(pred, dim=1).squeeze()
            
            # Accumulate confusion matrix
            conf_mat = compute_confusion_matrix(pred_mask, mask, cfg["num_classes"])
            total_conf_mat += conf_mat.cpu()
            
            # Save prediction
            pred_np = pred_mask.cpu().numpy().astype(np.uint8)
            cv2.imwrite(f"outputs/pred_{i:04d}.png", pred_np)
    
    # Compute final metrics
    per_class_iou, mean_iou = compute_iou_from_confusion(total_conf_mat)
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS")
    print("=" * 50)
    
    print("\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, per_class_iou)):
        print(f"  {name:20s}: {iou:.4f}")
    
    print(f"\n🏆 Mean IoU (mIoU): {mean_iou:.4f}")
    print("=" * 50)
    
    # Save results
    with open("outputs/benchmark_results.txt", "w") as f:
        f.write("OFFROAD SEGMENTATION BENCHMARK RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-class IoU:\n")
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            f.write(f"  {name:20s}: {iou:.4f}\n")
        f.write(f"\nMean IoU (mIoU): {mean_iou:.4f}\n")
    
    print(f"\n✅ Results saved to outputs/benchmark_results.txt")
    print(f"✅ Predictions saved to outputs/")


if __name__ == "__main__":
    main()
