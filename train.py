"""
Training Script for Offroad Segmentation
DINOv2 Backbone + FPN Decoder

Usage:
    python train.py
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import SegmentationDataset
from models.segformer import get_model
from utils.metrics import iou_score


def main():
    print("🔥 TRAINING SCRIPT STARTED 🔥")
    
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Dataset
    train_ds = SegmentationDataset(
        img_dir="dataset/train/images",
        mask_dir="dataset/train/masks",
        size=(cfg["image_height"], cfg["image_width"])
    )
    
    val_ds = SegmentationDataset(
        img_dir="dataset/val/images",
        mask_dir="dataset/val/masks",
        size=(cfg["image_height"], cfg["image_width"])
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("❌ Dataset empty! Check dataset paths.")
    
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False)
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False)
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = get_model(cfg["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_iou = 0.0
    
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        
        for imgs, masks in tqdm(train_dl, desc="Training"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dl)
        
        # Validation
        model.eval()
        total_iou = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                total_iou += iou_score(preds, masks, cfg["num_classes"]).item()
        
        avg_iou = total_iou / len(val_dl)
        
        print(f"Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")
        
        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "segformer.pth")
            print(f"✅ New best model saved! IoU: {avg_iou:.4f}")
    
    print(f"\n🏆 Training complete! Best IoU: {best_iou:.4f}")
    print("✅ Model saved as segformer.pth")


if __name__ == "__main__":
    main()
