# TheGrid - Hybrid CNN Vision Transformer for Object Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

A practical object detection framework combining Convolutional Neural Networks (CNN) and Vision Transformers with YOLO-style training/inference utilities (training, evaluation, inference, and visualization).

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Overview](#-overview)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#️-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [How to Train](#-how-to-train-this-model)
- [Advanced Usage](#-advanced-usage)
- [Performance Monitoring](#-performance-monitoring)
- [Configuration Guide](#️-configuration-guide)
- [Technical Report](#-technical-report)
- [Contributing](#-contributing)
- [Roadmap](#-roadmap)
- [License](#-license)

## ✨ Key Features

🎯 **Hybrid Architecture**
- CNN backbone for local feature extraction
- Vision Transformer blocks for global context
- Multi-scale feature fusion (FPN-style)
- Separate detection heads (bbox, objectness, classification)

⚡ **Performance Optimized**
- Mixed Precision Training (AMP) for 2x speedup
- Multi-worker data loading
- Efficient batch processing
- CUDA acceleration support
- EMA weights for more stable final mAP (optional)

📊 **Complete Evaluation Suite**
- mAP@0.5 calculation (COCO-style)
- Per-class Average Precision
- Precision-Recall curves
- Confusion matrix visualization
- Training loss tracking and plotting

🔧 **Production Features**
- Checkpoint management with best model tracking
- Resume training from any checkpoint
- Batch inference with JSON/CSV export
- Automated visualization with bounding boxes
- YOLO dataset format support

📁 **Professional Output**
- Model checkpoints with optimizer state
- Training curves and evaluation plots
- Detection visualizations
- Structured result exports (JSON/CSV)

## 📋 Overview

TheGrid implements a hybrid architecture that leverages the strengths of both CNNs and Vision Transformers:
- **Backbone (pretrained ResNet by default)**: Strong multi-scale features from ImageNet pretraining
- **Transformer Layers**: Captures long-range dependencies and global context
- **Feature Fusion**: Intelligently merges multi-scale features for robust detection
- **Multi-Head Detection**: Separate heads for bounding box regression, objectness scoring, and classification

## 🛠️ Tech Stack

### Core Framework
- **Python 3.x**: Primary programming language
- **PyTorch**: Deep learning framework for model development and training
- **torchvision**: Computer vision utilities and pre-trained models
- **CUDA**: GPU acceleration for faster training and inference

### Computer Vision & Data Processing
- **PIL (Pillow)**: Image loading and preprocessing
- **NumPy**: Numerical computations and array operations
- **OpenCV**: Image visualization and transformations
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning utilities (confusion matrix)
- **tqdm**: Progress bars for training and inference

### Model Architecture Components
- **Backbone**: Pretrained ResNet feature extractor (default: `resnet50`)
- **Vision Transformer**: Self-attention mechanism with 8 attention heads
- **Feature Fusion Module**: FPN-style fusion across three scales
- **Detection Heads**:
  - Box Head: Bounding box regression (x, y, w, h)
  - Class Head: 20-class object classification
  - Objectness Head: Object presence scoring

### Training Infrastructure
- **AdamW Optimizer**: Weight decay-based optimization (lr=1e-3, wd=1e-4)
- **Learning Rate Scheduler**: Adaptive learning rate adjustment
- **Mixed Precision (AMP)**: Automatic Mixed Precision for faster training
- **Custom Loss Functions**:
    - Bounding box loss (CIoU-style, via `giou_loss` wrapper)
  - Classification loss (Focal Loss)
  - Objectness loss
- **Data Augmentation**: Training-specific transformations
- **Checkpointing**: Auto-save with resume capability

### Inference & Evaluation
- **NMS (Non-Maximum Suppression)**: Duplicate detection filtering
- **mAP Calculation**: Mean Average Precision at IoU=0.5
- **Prediction Pipeline**: End-to-end inference with confidence thresholding
- **Batch Processing**: Efficient dataset-wide inference
- **Visualization Tools**: Detection result rendering with bounding boxes
- **Export Formats**: JSON, CSV, and image outputs

## 🏗️ Project Structure

```
hybriddetector/
├── backbone/              # Feature extraction modules
│   ├── cnn_backbone.py    # Multi-scale CNN feature extractor
│   ├── transformer.py     # Vision Transformer blocks
│   └── fusion.py          # Feature fusion module
├── heads/                 # Detection heads
│   ├── box_head.py        # Bounding box regression
│   ├── class_head.py      # Object classification
│   └── objectness_head.py # Objectness scoring
├── dataset/               # Data loading and augmentation
│   ├── custom_dataset.py  # Custom dataset implementation
│   └── transforms.py      # Image transformations
├── losses/                # Loss functions
│   ├── bbox_loss.py       # Bounding box loss
│   ├── cls_loss.py        # Classification loss
│   └── obj_loss.py        # Objectness loss
├── trainer/               # Training pipeline
│   ├── train.py           # Training loop
│   ├── scheduler.py       # Learning rate scheduling
│   └── evaluate.py        # Model evaluation
├── inference/             # Inference and visualization
│   ├── predictor.py       # Prediction pipeline
│   └── visualize.py       # Result visualization
├── utils/                 # Utilities
│   ├── config.py          # Configuration parameters
│   ├── nms.py             # Non-maximum suppression
│   └── seed.py            # Random seed setting
└── main.py                # Main training and inference script
```

## 🚀 Key Features

### Architecture Highlights
- **Competitive Backbone**: Pretrained ResNet (default) for much stronger representations
- **Hybrid Attention**: Combines local CNN features with global transformer attention
- **Efficient Fusion**: Adaptive feature fusion across different scales
- **Stronger Heads**: Deeper Conv+BN+SiLU towers for box/obj/cls predictions
- **Better Localization**: CIoU loss for box regression

### Training Configuration
- **Image Size**: 640×640 pixels
- **Batch Size**: 8
- **Epochs**: 5 (default; increase via CLI for longer runs)
- **Number of Classes**: 20
- **Transformer Heads**: 8
- **Learning Rate**: 1e-3 with weight decay 1e-4

### Inference Configuration
- **Confidence Threshold**: 0.3
- **IoU Threshold**: 0.5 (for NMS)
- **Device**: CUDA (GPU) or CPU fallback

## 📊 Model Pipeline

1. **Input**: RGB images (640×640)
2. **Feature Extraction**: CNN backbone produces 3 feature maps at different scales (P3/P4/P5)
3. **Attention Enhancement**: Transformer blocks process each feature map
4. **Feature Fusion**: Multi-scale features are fused into unified representation
5. **Detection (Multi-Scale Heads)**: Heads run on all three levels and predictions are concatenated:
   - Bounding boxes (coordinates)
   - Objectness scores (presence of objects)
   - Class probabilities (20 classes)
6. **Post-processing**: NMS filters overlapping detections
7. **Output**: Final detections with boxes, scores, and class labels

Notes:
- If EMA is enabled in [hybriddetector/utils/config.py](hybriddetector/utils/config.py), the checkpoint saved as `best_model.pth` uses EMA weights for better final stability.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/SAUNAK359/TheGrid.git
cd TheGrid

# Install required dependencies
pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn opencv-python tqdm

# For augmentations (required)
pip install albumentations
```

## 🚀 Quick Start

### Kaggle / YOLO-Style (Clone → Train → Predict)

This repo is structured so you can **clone into Kaggle** and run commands similar to YOLO.

- Install deps: `pip install -r requirements.txt`
- Create your `data.yaml` (start from `data.yaml.example`) and point it to your dataset.
- Use the module CLI:

Train (fast default):

`python -m hybriddetector train --data /kaggle/working/data.yaml --epochs 5 --batch 8 --img 640 --device cuda`

Example: train on the included Aquarium dataset (10 epochs):

`python -m hybriddetector train --data /workspaces/TheGrid/Test_Data/aquarium_pretrain/data.yaml --epochs 10 --batch 8 --img 640 --device cuda`

Train with early stopping (recommended when you have a validation split):

`python -m hybriddetector train --data /kaggle/working/data.yaml --epochs 50 --early-stop --patience 5 --min-epochs 5 --val-every 1`

Resume:

`python -m hybriddetector train --data /kaggle/working/data.yaml --resume checkpoints/latest_checkpoint.pth`

Predict (testing):

`python -m hybriddetector predict --weights checkpoints/best_model.pth --source /kaggle/input/yourdataset/images/val --data /kaggle/working/data.yaml --save-dir results/visualizations`

Notes:
- Inference uses the same normalization as training.
- Visualizations are saved as normal (non-black) RGB images with boxes drawn.
- If you see no boxes, try lowering the threshold: `--conf 0.01`

Model knobs (via `hybriddetector/utils/config.py`):
- `BACKBONE`: default `resnet50` (also supports `resnet18`, `resnet34`, `resnet101`)
- `BACKBONE_PRETRAINED`: load ImageNet weights (recommended)
- `ANCHORS`: 3 anchor priors as normalized `(w, h)`

Optional verification (consistency check):

`python -m hybriddetector predict --weights checkpoints/best_model.pth --source dataset/images/val --verify --verify-iou 0.5 --save-dir results/visualizations`

This writes `verification.csv` in `--save-dir`. A detection is marked `verified=True` if it is stable under a horizontal flip test-time augmentation (same class, IoU >= `--verify-iou`).

Class names:
- If you pass `--data path/to/data.yaml`, class names come from `names:` in that YAML.
- Otherwise the CLI falls back to `Config.CLASS_NAMES` (and pads with `class_N` if needed).

Dataset layout must follow YOLO convention:

`images/train`, `images/val`, `labels/train`, `labels/val` with per-image `.txt` labels.

---

## 🏷️ How to Use YOLO `.txt` Labels (Step-by-Step)

This project expects **YOLO-format label files**: one `.txt` per image, same filename (different extension), stored under `labels/<split>/`.

### Step 1 — Create the YOLO folder structure

```
dataset/
    images/
        train/
            img001.jpg
            img002.jpg
        val/
            img101.jpg
    labels/
        train/
            img001.txt
            img002.txt
        val/
            img101.txt
```

### Step 2 — Create `data.yaml`

Start from `data.yaml.example` and update it:

```yaml
path: .
train: dataset/images/train
val: dataset/images/val

names:
    - person
    - car
    - dog
```

Important:
- Class IDs in `.txt` must match the **index** in `names` (0-based).
    - `person` → `0`, `car` → `1`, `dog` → `2`

### Step 3 — Create one `.txt` per image

For an image `dataset/images/train/img001.jpg`, create:

`dataset/labels/train/img001.txt`

If an image has **no objects**, you can either:
- Create an **empty** `.txt` file, or
- Omit the `.txt` file entirely (this code treats missing labels as “no boxes”).

### Step 4 — Write labels in YOLO format

Each line is:

```
class_id x_center y_center width height
```

All coordinates are **normalized** to `[0, 1]` relative to image width/height:

- `x_center = x_center_pixels / image_width`
- `y_center = y_center_pixels / image_height`
- `width = box_width_pixels / image_width`
- `height = box_height_pixels / image_height`

### Step 5 — Example label file

If `img001.jpg` contains:
- a `person` (class 0) near the center
- a `car` (class 1) on the left

Then `img001.txt` might look like:

```
0 0.50 0.50 0.20 0.40
1 0.25 0.60 0.30 0.25
```

Notes:
- Multiple objects → multiple lines.
- Values must be separated by spaces.
- Use `.` decimals (not commas).

### Step 6 — Common mistakes to avoid

- ❌ Using `x1 y1 x2 y2` (corner format). YOLO requires **center + width/height**.
- ❌ Using pixel coordinates. Values must be **normalized** to `[0,1]`.
- ❌ Wrong class IDs (must match the `names:` order).
- ❌ Mismatched filenames (label must match image stem exactly).

### Step 7 — Train

Once your folders + labels are ready:

`python -m hybriddetector train --data ./data.yaml --epochs 50 --batch 8 --img 640 --device cuda`

```bash
# 1. Prepare your YOLO format dataset (see Dataset Preparation section)
# 2. Update configuration in hybriddetector/utils/config.py
# 3. Start training
python hybriddetector/main.py
```

**That's it!** The framework will automatically:
- ✅ Train with mixed precision for optimal performance
- ✅ Save checkpoints every 5 epochs
- ✅ Generate evaluation metrics (mAP, PR curves, confusion matrix)
- ✅ Create loss curves and visualizations
- ✅ Save predictions in JSON and CSV formats
- ✅ Export detection images with bounding boxes

All results will be organized in the `./results` directory.

## 💻 Usage

```bash
# Quick start - train the model with all features
python hybriddetector/main.py

# The training will:
# ✅ Train with mixed precision (AMP) for faster training
# ✅ Save checkpoints every 5 epochs
# ✅ Run evaluation with mAP, PR curves, and confusion matrix
# ✅ Generate and save loss curves
# ✅ Save visualizations of predictions
# ✅ Export results as JSON and CSV
```

## ⚡ New Features

### 🎯 Training Enhancements
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training on modern GPUs
- **Checkpointing**: Auto-save checkpoints every N epochs with best model tracking
- **Resume Training**: Continue training from saved checkpoints
- **Loss Tracking**: Track total, bbox, classification, and objectness losses separately

### 🚄 Training Speed Optimizations (Biggest Wins First)

| Upgrade | Expected Speed Gain | Risk |
|---|---:|---|
| Mixed Precision (AMP) | 1.6–2.0× | None |
| Freeze CNN early | 1.3–1.5× | Low |
| Transformer only at low-res (40×40) | ~1.4× | Very low |
| Gradient accumulation tuning | ~1.2× (effective) | None |
| Smarter data loading | 1.1–1.3× | None |
| Token pooling inside MHSA | 2–4× (transformer part) | Medium |

These are implemented as simple toggles in the config (no code changes needed).

### 📊 Evaluation & Metrics
- **mAP Calculation**: Mean Average Precision at IoU=0.5
- **Per-Class AP**: Individual Average Precision for each class
- **Precision-Recall Curves**: Visualized PR curves for all classes
- **Confusion Matrix**: Heatmap showing classification performance
- **Loss Curves**: Training loss visualization over epochs

### 🔍 Inference Capabilities
- **Batch Inference**: Process entire datasets efficiently
- **Multiple Export Formats**: JSON and CSV result files
- **Visualization**: Auto-generated detection images with bounding boxes
- **YOLO-style Outputs**: Industry-standard result format

### 📁 Output Structure
```
results/
├── checkpoints/           # Model checkpoints
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   ├── checkpoint_epoch_5.pth
│   └── loss_history.json
├── plots/                 # Training and evaluation plots
│   ├── loss_curves.png
│   ├── pr_curves.png
│   └── confusion_matrix.png
├── visualizations/        # Detection result images
│   ├── demo_result.jpg
│   ├── detection_0001.jpg
│   └── ...
├── predictions.json       # Predictions in JSON format
└── predictions.csv        # Predictions in CSV format
```

## 🎓 How to Train This Model

### Step 1: Prepare Your Dataset (YOLO Format)

**This model uses YOLO dataset format** - the industry standard for object detection.

1. **Organize your data structure**:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img100.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   ├── img002.txt
│   │   └── ...
│   └── val/
│       ├── img100.txt
│       └── ...
└── data.yaml
```

2. **Create annotation files** - One `.txt` file per image with YOLO format:

**Example: `img001.txt`**
```
5 0.512 0.438 0.312 0.219
12 0.687 0.562 0.234 0.187
```

**Format**: `class_id x_center y_center width height`

Where:
- `class_id`: Integer class label (0-19 for 20 classes)
- `x_center`: Center X coordinate (normalized 0-1)
- `y_center`: Center Y coordinate (normalized 0-1)
- `width`: Bounding box width (normalized 0-1)
- `height`: Bounding box height (normalized 0-1)

**Note**: All coordinates are normalized relative to image dimensions.

3. **Create `data.yaml`** configuration file:
```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 20  # number of classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
```

**Converting from other formats**:
- **COCO to YOLO**: Use `coco2yolo.py` converter scripts
- **Pascal VOC to YOLO**: Use `voc2yolo.py` converter scripts
- **LabelImg**: Export directly to YOLO format

### Step 2: Configure Training Parameters

Edit `hybriddetector/utils/config.py` to customize your training:

```python
class Config:
    # Device Configuration
    DEVICE = 'cuda'  # Use 'cuda' for GPU or 'cpu' for CPU training
    
    # Training Hyperparameters
    BATCH_SIZE = 8      # Adjust based on GPU memory (reduce if OOM)
    IMG_SIZE = 640      # Input image size (640x640)
    EPOCHS = 50         # Number of training epochs
    LR = 1e-3           # Initial learning rate
    WEIGHT_DECAY = 1e-4 # L2 regularization strength
    
    # Model Architecture
    NUM_CLASSES = 20           # Number of object classes in your dataset
    BACKBONE_CHANNELS = [64, 256, 512]  # Feature channels at each scale
    TRANSFORMER_HEADS = 8      # Number of attention heads
    
    # Detection Parameters
    CONF_THRESH = 0.3   # Confidence threshold for predictions
    IOU_THRESH = 0.5    # IoU threshold for NMS
    
    # Dataset Paths (YOLO format)
    DATA_DIR = './dataset/images/train'       # Training images
    VAL_DIR = './dataset/images/val'          # Validation images
    LABELS_DIR = './dataset/labels/train'     # Training labels
    VAL_LABELS_DIR = './dataset/labels/val'   # Validation labels
    DATA_YAML = './dataset/data.yaml'         # Dataset configuration
    SAVE_DIR = './checkpoints'                # Model checkpoints
```

### Step 3: Verify Environment

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check installed packages
pip list | grep -E "torch|torchvision|pillow"
```

### Step 4: Start Training

```bash
# Navigate to project root
cd /workspaces/TheGrid

# Start training
python hybriddetector/main.py
```

**Expected Output**:
```
Epoch 1/50
Average Loss: 2.4567
Epoch 2/50
Average Loss: 2.1234
...
```

### Step 5: Monitor Training

The training process will:
1. ✅ Load and preprocess your dataset
2. ✅ Initialize the hybrid CNN-Transformer model
3. ✅ Train for the specified number of epochs
4. ✅ Display average loss for each epoch
5. ✅ Run inference on a demo image
6. ✅ Visualize predictions

### Step 6: Advanced Training Options

#### Modify Data Augmentation

Edit `hybriddetector/dataset/transforms.py` to customize augmentations:
```python
def get_transforms(train=True, img_size=640):
    if train:
        return Compose([
            Resize(img_size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ])
```

        ## Supported Dataset Layouts

        Alternative layout also supported (common on Kaggle):

#### Custom Training Loop

For more control, modify `hybriddetector/trainer/train.py`:
```python
# Add validation during training
# Add model checkpointing
# Add tensorboard logging
# Add early stopping
```

#### Multi-GPU Training

```python
# In main.py, wrap model with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

### Step 7: Save and Load Model

```python
# Save trained model
torch.save(model.state_dict(), 'checkpoints/hybrid_detector.pth')

# Load pre-trained model
model = HybridDetector()
model.load_state_dict(torch.load('checkpoints/hybrid_detector.pth'))
model.eval()
```

### Training Tips

1. **GPU Memory Issues**: If you encounter OOM errors:
   - Reduce `BATCH_SIZE` (try 4 or 2)
   - Reduce `IMG_SIZE` (try 512 or 416)
   - Enable gradient accumulation

2. **Slow Convergence**: If loss decreases slowly:
   - Increase learning rate (try 5e-3)
   - Adjust optimizer parameters
   - Check data augmentation isn't too aggressive

3. **Overfitting**: If validation loss increases:
   - Increase weight decay
   - Add more data augmentation
   - Reduce model complexity

4. **Best Practices**:
   - Start with a small subset of data to verify pipeline
   - Monitor training loss and learning rate
   - Save checkpoints regularly
   - Use tensorboard for visualization
   - Validate on held-out test set

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16 GB
- GPU: 6 GB VRAM (e.g., GTX 1060)
- Storage: 10 GB

**Recommended**:
- CPU: 8+ cores
- RAM: 32 GB
- GPU: 12+ GB VRAM (e.g., RTX 3080, V100)
- Storage: 50 GB SSD

### Training Time Estimates

| Dataset Size | GPU          | Batch Size | Time per Epoch | Total (50 epochs) |
|-------------|--------------|------------|----------------|-------------------|
| 1K images   | RTX 3080     | 8          | ~2 min         | ~2 hours          |
| 5K images   | RTX 3080     | 8          | ~8 min         | ~7 hours          |
| 10K images  | V100         | 16         | ~10 min        | ~8 hours          |
| 50K images  | A100         | 32         | ~30 min        | ~25 hours         |

## 📈 Technical Report

### Model Architecture

**Backbone**: The CNN backbone extracts features at three different scales, capturing both fine-grained details and high-level semantic information. Each feature map is then processed by a dedicated Vision Transformer block that applies self-attention to capture long-range dependencies.

**Feature Fusion**: The fusion module combines features from different scales using learnable weights, creating a rich multi-scale representation that balances spatial resolution and semantic depth.

**Detection Heads**: Three specialized heads work in parallel:
- **Box Head**: Regresses precise bounding box coordinates
- **Objectness Head**: Scores the likelihood of object presence
- **Class Head**: Predicts probability distribution over 20 object classes

### Training Strategy

- **Optimizer**: AdamW with decoupled weight decay for better generalization
- **Learning Rate**: Adaptive scheduling for optimal convergence
- **Data Augmentation**: Training-specific transformations to improve robustness
- **Reproducibility**: Fixed random seed (42) for consistent results

### Innovation Points

1. **Hybrid Architecture**: Combines the best of CNNs (local features) and Transformers (global context)
2. **Multi-Scale Processing**: Separate transformer blocks for each scale preserve scale-specific information
3. **Modular Design**: Easy to swap components and experiment with different configurations
4. **End-to-End Training**: All components trained jointly for optimal performance

### Performance Considerations

- **GPU Acceleration**: CUDA support for fast training and inference
- **Efficient Attention**: Transformer heads tuned for optimal speed/accuracy trade-off
- **Batch Processing**: Configurable batch size for memory optimization
- **Inference Speed**: Optimized prediction pipeline with NMS for real-time capability

## 📝 License

See LICENSE file for details.

## 🚀 Advanced Usage

### Resume Training from Checkpoint

```python
# In config.py, set:
RESUME_TRAINING = True
RESUME_CHECKPOINT = './checkpoints/latest_checkpoint.pth'

# Then run training normally
python hybriddetector/main.py
```

### Run Evaluation Only

```python
from trainer.evaluate import evaluate_model
from utils import config
import torch

# Load model
model = HybridDetector()
model.load_state_dict(torch.load('./checkpoints/best_model.pth')['model_state_dict'])

# Run evaluation
results = evaluate_model(
    model, 
    val_loader, 
    device='cuda',
    num_classes=config.Config.NUM_CLASSES,
    save_dir='./eval_results'
)

print(f"mAP@0.5: {results['mAP']:.4f}")
```

### Batch Inference on Test Set

```python
from inference.predictor import Predictor
from utils import config

# Create predictor
predictor = Predictor(model, conf_thresh=0.3, iou_thresh=0.5, device='cuda')

# Run batch inference
results = predictor.predict_batch_dataset(
    test_dataloader,
    class_names=config.Config.CLASS_NAMES,
    save_json=True,
    save_csv=True,
    output_dir='./test_results'
)
```

### Visualize Results

```python
from inference.visualize import save_detection_image

# Save single detection image
save_detection_image(
    image,
    boxes,
    labels,
    scores,
    class_names=config.Config.CLASS_NAMES,
    save_path='./results/my_detection.jpg'

If you use this Kaggle layout, set `train:` in `data.yaml` to the **images folder** (e.g. `/kaggle/input/train/images`) and the code will infer labels as `/kaggle/input/train/labels`.
)
```

## 📊 Performance Monitoring

The framework automatically generates comprehensive performance metrics:

### Training Metrics
- **Loss Curves**: Total, BBox, Classification, and Objectness losses
- **Checkpoints**: Regular saves with best model tracking
- **Progress Tracking**: Real-time loss updates via tqdm

### Evaluation Metrics
- **mAP@0.5**: Standard COCO-style mean Average Precision
- **Per-Class AP**: Individual performance for each class
- **PR Curves**: Precision-Recall visualization
- **Confusion Matrix**: Classification performance heatmap

### Export Formats
- **JSON**: Structured predictions with metadata
- **CSV**: Tabular format for easy analysis
- **Images**: Visualized detections with bounding boxes

## 🛠️ Configuration Guide

All settings are in `hybriddetector/utils/config.py`:

```python
# Training Settings
BATCH_SIZE = 8          # Reduce if OOM
USE_AMP = True          # Mixed precision (faster)
EPOCHS = 50             # Training duration

# Speed knobs
FREEZE_CNN_EPOCHS = 10          # Freeze CNN for early epochs
TRANSFORMER_LOW_RES_ONLY = True # Transformer only on 40x40 map
TOKEN_POOL_FACTOR = 2           # AvgPool factor inside attention
GRAD_ACCUM_STEPS = 1            # Increase to simulate bigger batches

# DataLoader performance
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Checkpointing
SAVE_EVERY_N_EPOCHS = 5         # Checkpoint frequency
SAVE_BEST = True                # Save best model
RESUME_TRAINING = False         # Resume from checkpoint

# Evaluation
EVAL_EVERY_N_EPOCHS = 5         # Evaluation frequency
SAVE_EVAL_PLOTS = True          # Generate plots

# Paths
SAVE_DIR = './checkpoints'      # Model checkpoints
RESULTS_DIR = './results'       # All results
PLOTS_DIR = './results/plots'   # Metrics plots
VIS_DIR = './results/visualizations'  # Detection images
```

## 🤝 Contributing

Contributions are welcome! This modular architecture makes it easy to experiment with:
- Different backbone architectures
- Alternative fusion strategies
- Novel attention mechanisms
- Custom loss functions
- Enhanced data augmentation techniques
- New evaluation metrics
- Performance optimizations

## 🎯 Roadmap

- [x] Hybrid CNN-Transformer architecture
- [x] Multi-scale feature fusion
- [x] Mixed precision training
- [x] Comprehensive evaluation metrics (mAP, PR curves, confusion matrix)
- [x] Checkpoint management and resume training
- [x] Batch inference with multiple export formats
- [x] Automated visualization
- [ ] Multi-GPU distributed training (DDP)
- [ ] TensorBoard integration
- [ ] ONNX export for deployment
- [ ] Model quantization
- [ ] Real-time video inference
- [ ] Web demo interface

---

**Note**: This is a production-ready object detection framework implementing cutting-edge hybrid CNN-Transformer architectures with complete YOLO-style features including training, evaluation, inference, and visualization capabilities.
