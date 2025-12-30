├── heads/                   # Detection heads
│   ├── __init__.py
│   ├── box_head.py           # Bounding box regression
│   ├── class_head.py         # Classification head
│   └── objectness_head.py    # Objectness head
│
├── losses/                  # Custom loss functions
│   ├── __init__.py
│   ├── bbox_loss.py          # IoU, GIoU, DIoU
│   ├── cls_loss.py           # Weighted cross-entropy / focal loss
│   └── obj_loss.py           # Objectness BCE / focal loss
│
├── dataset/                 # Dataset and dataloader
│   ├── __init__.py
│   ├── custom_dataset.py
│   └── transforms.py         # Augmentation and preprocessing
│
├── trainer/                 # Training utilities
│   ├── __init__.py
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation metrics (mAP, precision, recall)
│   └── scheduler.py         # Learning rate scheduler
│
├── inference/               # Inference utilities
│   ├── __init__.py
│   ├── predictor.py         # Forward pass + postprocessing (NMS)
│   └── visualize.py          # Visualize predictions
│
├── utils/                   # Helper functions
│   ├── __init__.py
│   ├── config.py            # Model / training config
│   ├── seed.py              # Seed setting for reproducibility
│   └── nms.py               # Custom NMS if needed
│
└── main.py  