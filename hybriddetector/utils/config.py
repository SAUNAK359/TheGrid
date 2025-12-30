# hybriddetector/utils/config.py

class Config:
    # Device
    DEVICE = 'cuda'  # or 'cpu'

    # Training
    BATCH_SIZE = 8
    IMG_SIZE = 640
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Model
    NUM_CLASSES = 20
    BACKBONE_CHANNELS = [64, 256, 512]
    TRANSFORMER_HEADS = 8

    # Loss thresholds
    CONF_THRESH = 0.3
    IOU_THRESH = 0.5

    # Paths (YOLO format)
    DATA_DIR = './dataset/images/train'
    VAL_DIR = './dataset/images/val'
    LABELS_DIR = './dataset/labels/train'
    VAL_LABELS_DIR = './dataset/labels/val'
    DATA_YAML = './dataset/data.yaml'
    SAVE_DIR = './checkpoints'
