# hybriddetector/utils/config.py

class Config:
    # Device
    DEVICE = 'cuda'  # or 'cpu'

    # Training Hyperparameters
    BATCH_SIZE = 8
    IMG_SIZE = 640
    # Keep default training quick; override via CLI --epochs for longer runs.
    EPOCHS = 5
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    USE_AMP = True  # Mixed precision training (faster on modern GPUs)
    GRAD_ACCUM_STEPS = 1  # Increase if VRAM-limited (effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS)

    # DataLoader performance
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2

    # Speed/compute knobs
    FREEZE_CNN_EPOCHS = 0  # Freeze CNN backbone for first N epochs
    TRANSFORMER_LOW_RES_ONLY = True  # Apply transformer only on low-res (40x40) feature map
    TOKEN_POOL_FACTOR = 2  # AvgPool factor inside MHSA (reduces tokens by factor^2)

    # Model Architecture
    NUM_CLASSES = 20
    # Backbone: default to a pretrained ResNet for competitive performance.
    BACKBONE = 'resnet50'
    BACKBONE_PRETRAINED = True
    # Feature channels returned by the backbone (high/med/low): [C2, C3, C4] for ResNet.
    BACKBONE_CHANNELS = [256, 512, 1024]
    # Spatial strides (relative to input image) for the three backbone outputs.
    BACKBONE_STRIDES = [4, 8, 16]
    TRANSFORMER_HEADS = 8

    # Anchor priors (normalized w,h in [0,1]) for 3 anchors.
    # These are general-purpose; for best results, tune per-dataset.
    ANCHORS = [(0.04, 0.06), (0.10, 0.14), (0.20, 0.28)]

    # Detection Thresholds
    CONF_THRESH = 0.3
    IOU_THRESH = 0.5

    # Paths (YOLO format)
    DATA_DIR = './dataset/images/train'
    VAL_DIR = './dataset/images/val'
    LABELS_DIR = './dataset/labels/train'
    VAL_LABELS_DIR = './dataset/labels/val'
    DATA_YAML = './dataset/data.yaml'
    
    # Output Paths
    SAVE_DIR = './checkpoints'
    RESULTS_DIR = './results'
    PLOTS_DIR = './results/plots'
    VIS_DIR = './results/visualizations'
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 5  # Save checkpoint every N epochs
    SAVE_BEST = True  # Save best model based on loss
    RESUME_TRAINING = False  # Resume from checkpoint
    RESUME_CHECKPOINT = './checkpoints/latest_checkpoint.pth'
    
    # Evaluation
    EVAL_EVERY_N_EPOCHS = 5  # Run evaluation every N epochs
    SAVE_EVAL_PLOTS = True  # Save mAP, PR curves, confusion matrix
    
    # Inference
    SAVE_PREDICTIONS_JSON = True
    SAVE_PREDICTIONS_CSV = True
    SAVE_VISUALIZATIONS = True
    MAX_VIS_IMAGES = 50  # Maximum number of images to visualize
    
    # Class Names (COCO subset example - update for your dataset)
    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow'
    ]
