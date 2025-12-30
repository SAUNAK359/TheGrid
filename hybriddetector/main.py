# hybriddetector/main.py

import torch
import torch.nn as nn
from torch import optim
from backbone import cnn_backbone, transformer, fusion
from heads import box_head, class_head, objectness_head
from dataset import custom_dataset, transforms
from trainer import train, scheduler, evaluate
from inference import predictor, visualize
from utils import config, seed

import os
from PIL import Image
import torchvision.transforms as T

# -------------------------
# 1. Set seed & device
# -------------------------
seed.set_seed(42)
DEVICE_TYPE = config.Config.DEVICE if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)

# -------------------------
# 2. Prepare Dataset
# -------------------------
train_transforms = transforms.get_transforms(train=True, img_size=config.Config.IMG_SIZE)
train_dataset = custom_dataset.CustomDataset(
    csv_file=config.Config.ANNOTATIONS,
    img_dir=config.Config.DATA_DIR,
    transform=train_transforms
)

# -------------------------
# 3. Build Model
# -------------------------
class HybridDetector(torch.nn.Module):
    def __init__(self):
        super(HybridDetector, self).__init__()
        # Backbone
        self.cnn = cnn_backbone.CNNBackbone()
        self.transformers = nn.ModuleList([
            transformer.TransformerBlock(ch, h, w, num_heads=config.Config.TRANSFORMER_HEADS)
            for ch, h, w in zip(config.Config.BACKBONE_CHANNELS, [320, 80, 40], [320, 80, 40])
        ])
        # Fusion
        self.fusion = fusion.FeatureFusion(
            channels_high=config.Config.BACKBONE_CHANNELS[0],
            channels_med=config.Config.BACKBONE_CHANNELS[1],
            channels_low=config.Config.BACKBONE_CHANNELS[2],
            out_channels=256
        )
        # Heads
        self.box_head = box_head.BoxHead()
        self.cls_head = class_head.ClassHead(num_classes=config.Config.NUM_CLASSES)
        self.obj_head = objectness_head.ObjectnessHead()

    def forward(self, x):
        f1, f2, f3 = self.cnn(x)
        f1 = self.transformers[0](f1)
        f2 = self.transformers[1](f2)
        f3 = self.transformers[2](f3)
        fused = self.fusion(f1, f2, f3)
        boxes = self.box_head(fused)
        obj_scores = self.obj_head(fused)
        class_probs = self.cls_head(fused)
        return {'boxes': boxes, 'objectness': obj_scores, 'class_probs': class_probs}

model = HybridDetector().to(DEVICE)

# -------------------------
# 4. Optimizer & Scheduler
# -------------------------
optimizer = optim.AdamW(model.parameters(), lr=config.Config.LR, weight_decay=config.Config.WEIGHT_DECAY)
steps_per_epoch = len(train_dataset) // config.Config.BATCH_SIZE
lr_scheduler = scheduler.get_scheduler(optimizer, max_lr=config.Config.LR, epochs=config.Config.EPOCHS, steps_per_epoch=steps_per_epoch)

# -------------------------
# 5. Trainer
# -------------------------
trainer = train.Trainer(model, train_dataset, optimizer, scheduler=lr_scheduler, device=DEVICE_TYPE, batch_size=config.Config.BATCH_SIZE)

for epoch in range(config.Config.EPOCHS):
    print(f"Epoch {epoch+1}/{config.Config.EPOCHS}")
    avg_loss = trainer.train_epoch()
    print(f"Average Loss: {avg_loss:.4f}")

# -------------------------
# 6. Inference Example
# -------------------------
# Load single image tensor for demo
image_path = os.path.join(config.Config.DATA_DIR, "demo.jpg")
img = Image.open(image_path).convert("RGB")
transform = T.Compose([T.Resize((config.Config.IMG_SIZE, config.Config.IMG_SIZE)), T.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

predictor_obj = predictor.Predictor(model, conf_thresh=config.Config.CONF_THRESH, iou_thresh=config.Config.IOU_THRESH, device=DEVICE_TYPE)
boxes, scores, labels = predictor_obj.predict(img_tensor)

# Visualize predictions
visualize.visualize_predictions(img_tensor[0], boxes[0], labels[0], scores[0])
