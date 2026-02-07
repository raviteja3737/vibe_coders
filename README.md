# Offroad Semantic Scene Segmentation using Digital Twins

Off-road semantic scene segmentation using Falcon digital twin data, trained with a DINOv2-FPN model and evaluated on unseen desert environments.

## Description

This project performs pixel-wise semantic segmentation of off-road scenes using:
- **Backbone**: DINOv2 (Vision Transformer with self-supervised learning)
- **Decoder**: Feature Pyramid Network (FPN)
- **Training Data**: Falcon synthetic digital twin dataset

## Problem Statement

Autonomous navigation in off-road environments requires understanding of:
- Traversable terrain (roads, ground)
- Obstacles (rocks, logs, vegetation)
- Environmental features (trees, bushes)

## Dataset

### Data Split
| Split | Images | Purpose |
|-------|--------|---------|
| Train | Training data | Model training |
| Val | Validation data | Hyperparameter tuning |
| Test | Test images | Final evaluation |

### Classes
- Background, Trees, Lush Bushes, Dry Grass, Dry Bushes
- Ground Clutter, Rocks, Logs, Ground, Vehicle

## Model Architecture

- **Backbone**: DINOv2-ViT-B/14 (frozen, pretrained)
- **Decoder**: FPN with progressive upsampling
- **Input Resolution**: 518 × 294
- **Framework**: PyTorch

## Evaluation Metric

- **Mean Intersection-over-Union (mIoU)**: Primary metric for semantic segmentation

## Project Structure

```
offroad_segmentation/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── models/
│   └── segformer.py      # DINOv2-FPN model
│
├── utils/
│   ├── dataset.py        # Dataset loader
│   └── metrics.py        # IoU metrics
│
├── outputs/              # Saved predictions
├── train.py              # Training script
├── test.py               # Evaluation script
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── segformer.pth         # Trained weights
└── README.md
```

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

## Requirements

- Python >= 3.9
- PyTorch
- torchvision
- opencv-python
- pyyaml
- tqdm
- numpy

## License

This project is for research and educational purposes.
