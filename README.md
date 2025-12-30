# TheGrid - Hybrid CNN Vision Transformer for Object Detection

A state-of-the-art object detection framework combining Convolutional Neural Networks (CNN) and Vision Transformers to achieve superior performance in detecting and classifying objects in images.

## 📋 Overview

TheGrid implements a hybrid architecture that leverages the strengths of both CNNs and Vision Transformers:
- **CNN Backbone**: Extracts multi-scale hierarchical features with strong local inductive biases
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
- **OpenCV** (implied): Image visualization and transformations

### Model Architecture Components
- **CNN Backbone**: Multi-scale feature extraction with 3 feature pyramid levels
- **Vision Transformer**: Self-attention mechanism with 8 attention heads
- **Feature Fusion Module**: Multi-scale feature integration (64→256→512 channels)
- **Detection Heads**:
  - Box Head: Bounding box regression (x, y, w, h)
  - Class Head: 20-class object classification
  - Objectness Head: Object presence scoring

### Training Infrastructure
- **AdamW Optimizer**: Weight decay-based optimization (lr=1e-3, wd=1e-4)
- **Learning Rate Scheduler**: Adaptive learning rate adjustment
- **Custom Loss Functions**:
  - Bounding box loss
  - Classification loss
  - Objectness loss
- **Data Augmentation**: Training-specific transformations

### Inference & Evaluation
- **NMS (Non-Maximum Suppression)**: Duplicate detection filtering
- **Prediction Pipeline**: End-to-end inference with confidence thresholding
- **Visualization Tools**: Detection result rendering

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
- **Multi-Scale Feature Extraction**: 3-level feature pyramid (64, 256, 512 channels)
- **Hybrid Attention**: Combines local CNN features with global transformer attention
- **Efficient Fusion**: Adaptive feature fusion across different scales
- **Modular Design**: Easily extendable and customizable components

### Training Configuration
- **Image Size**: 640×640 pixels
- **Batch Size**: 8
- **Epochs**: 50
- **Number of Classes**: 20
- **Transformer Heads**: 8
- **Learning Rate**: 1e-3 with weight decay 1e-4

### Inference Configuration
- **Confidence Threshold**: 0.3
- **IoU Threshold**: 0.5 (for NMS)
- **Device**: CUDA (GPU) or CPU fallback

## 📊 Model Pipeline

1. **Input**: RGB images (640×640)
2. **Feature Extraction**: CNN backbone produces 3 feature maps at different scales
3. **Attention Enhancement**: Transformer blocks process each feature map
4. **Feature Fusion**: Multi-scale features are fused into unified representation
5. **Detection**: Three parallel heads predict:
   - Bounding boxes (coordinates)
   - Objectness scores (presence of objects)
   - Class probabilities (20 classes)
6. **Post-processing**: NMS filters overlapping detections
7. **Output**: Final detections with boxes, scores, and class labels

## 🔧 Installation

```bash
# Install required dependencies
pip install torch torchvision pillow numpy
```

## 💻 Usage

```bash
# Quick start - train the model
python hybriddetector/main.py
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

## 🤝 Contributing

Contributions are welcome! This modular architecture makes it easy to experiment with:
- Different backbone architectures
- Alternative fusion strategies
- Novel attention mechanisms
- Custom loss functions
- Enhanced data augmentation techniques

---

**Note**: This is a research and development project implementing cutting-edge object detection techniques using hybrid CNN-Transformer architectures.
