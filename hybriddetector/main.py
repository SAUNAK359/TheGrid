# hybriddetector/main.py

import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from backbone import cnn_backbone, transformer, fusion
from heads import box_head, class_head, objectness_head
from dataset import custom_dataset, transforms
from trainer import train, scheduler, evaluate
from inference import predictor, visualize
from utils import config, seed, metrics

import os
from PIL import Image
import torchvision.transforms as T

print("="*80)
print("🚀 Hybrid CNN-Transformer Object Detector Training")
print("="*80)

# -------------------------
# 1. Set seed & device
# -------------------------
seed.set_seed(42)
DEVICE_TYPE = config.Config.DEVICE if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
print(f"\n✓ Device: {DEVICE_TYPE}")
print(f"✓ Mixed Precision: {config.Config.USE_AMP}")

# -------------------------
# 2. Prepare Dataset
# -------------------------
print("\n" + "="*80)
print("📊 Loading Datasets")
print("="*80)

train_transforms = transforms.get_transforms(train=True, img_size=config.Config.IMG_SIZE)
train_dataset = custom_dataset.CustomDataset(
    csv_file=config.Config.LABELS_DIR,
    img_dir=config.Config.DATA_DIR,
    transform=train_transforms
)

# Validation dataset (if available)
try:
    val_transforms = transforms.get_transforms(train=False, img_size=config.Config.IMG_SIZE)
    val_dataset = custom_dataset.CustomDataset(
        csv_file=config.Config.VAL_LABELS_DIR,
        img_dir=config.Config.VAL_DIR,
        transform=val_transforms
    )
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
except:
    val_dataset = None
    print(f"✓ Training samples: {len(train_dataset)}")
    print("⚠ No validation dataset found")

# -------------------------
# 3. Build Model
# -------------------------
print("\n" + "="*80)
print("🏗️  Building Hybrid Detector Model")
print("="*80)

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

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Trainable parameters: {trainable_params:,}")

# -------------------------
# 4. Optimizer & Scheduler
# -------------------------
print("\n" + "="*80)
print("⚙️  Configuring Training")
print("="*80)

optimizer = optim.AdamW(model.parameters(), lr=config.Config.LR, weight_decay=config.Config.WEIGHT_DECAY)
steps_per_epoch = len(train_dataset) // config.Config.BATCH_SIZE
lr_scheduler = scheduler.get_scheduler(
    optimizer, 
    max_lr=config.Config.LR, 
    epochs=config.Config.EPOCHS, 
    steps_per_epoch=steps_per_epoch
)

print(f"✓ Optimizer: AdamW (lr={config.Config.LR}, wd={config.Config.WEIGHT_DECAY})")
print(f"✓ Batch size: {config.Config.BATCH_SIZE}")
print(f"✓ Epochs: {config.Config.EPOCHS}")
print(f"✓ Steps per epoch: {steps_per_epoch}")

# -------------------------
# 5. Initialize Trainer
# -------------------------
trainer = train.Trainer(
    model, 
    train_dataset, 
    optimizer, 
    scheduler=lr_scheduler, 
    device=DEVICE_TYPE, 
    batch_size=config.Config.BATCH_SIZE,
    use_amp=config.Config.USE_AMP,
    checkpoint_dir=config.Config.SAVE_DIR
)

# Resume training if specified
start_epoch = 0
if config.Config.RESUME_TRAINING and os.path.exists(config.Config.RESUME_CHECKPOINT):
    print(f"\n✓ Resuming from checkpoint: {config.Config.RESUME_CHECKPOINT}")
    checkpoint = trainer.load_checkpoint(config.Config.RESUME_CHECKPOINT)
    start_epoch = checkpoint['epoch'] + 1

# -------------------------
# 6. Training Loop
# -------------------------
print("\n" + "="*80)
print("🎓 Starting Training")
print("="*80)

for epoch in range(start_epoch, config.Config.EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{config.Config.EPOCHS}")
    print(f"{'='*80}")
    
    # Train one epoch
    loss_dict = trainer.train_epoch()
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Total Loss: {loss_dict['total']:.4f}")
    print(f"   BBox Loss:  {loss_dict['bbox']:.4f}")
    print(f"   Class Loss: {loss_dict['cls']:.4f}")
    print(f"   Obj Loss:   {loss_dict['obj']:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % config.Config.SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == config.Config.EPOCHS:
        is_best = loss_dict['total'] < trainer.best_loss
        if is_best:
            trainer.best_loss = loss_dict['total']
        
        trainer.save_checkpoint(epoch + 1, is_best=is_best)
    
    # Run evaluation
    if val_dataset and (epoch + 1) % config.Config.EVAL_EVERY_N_EPOCHS == 0:
        print(f"\n{'='*80}")
        print(f"📈 Running Evaluation")
        print(f"{'='*80}")
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config.Config.BATCH_SIZE, 
            shuffle=False
        )
        
        eval_results = evaluate.evaluate_model(
            model, 
            val_loader, 
            DEVICE, 
            config.Config.NUM_CLASSES,
            save_dir=config.Config.PLOTS_DIR
        )

# Plot and save loss curves
print("\n" + "="*80)
print("📉 Saving Training Curves")
print("="*80)

loss_history = trainer.get_loss_history()
metrics.plot_loss_curves(
    loss_history, 
    save_path=os.path.join(config.Config.PLOTS_DIR, 'loss_curves.png')
)

print(f"\n✓ Training complete!")
print(f"✓ Best loss: {trainer.best_loss:.4f}")
print(f"✓ Checkpoints saved to: {config.Config.SAVE_DIR}")

# -------------------------
# 7. Final Evaluation
# -------------------------
if val_dataset:
    print("\n" + "="*80)
    print("🎯 Final Model Evaluation")
    print("="*80)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.Config.BATCH_SIZE, 
        shuffle=False
    )
    
    final_eval = evaluate.evaluate_model(
        model, 
        val_loader, 
        DEVICE, 
        config.Config.NUM_CLASSES,
        save_dir=config.Config.RESULTS_DIR
    )
    
    print(f"\n✅ Final mAP@0.5: {final_eval['mAP']:.4f}")

# -------------------------
# 8. Demo Inference
# -------------------------
print("\n" + "="*80)
print("🔍 Running Demo Inference")
print("="*80)

# Load demo image (if exists)
demo_image_path = os.path.join(config.Config.DATA_DIR, "demo.jpg")
if os.path.exists(demo_image_path):
    img = Image.open(demo_image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((config.Config.IMG_SIZE, config.Config.IMG_SIZE)), 
        T.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Create predictor
    predictor_obj = predictor.Predictor(
        model, 
        conf_thresh=config.Config.CONF_THRESH, 
        iou_thresh=config.Config.IOU_THRESH, 
        device=DEVICE_TYPE
    )
    
    # Run prediction
    boxes, scores, labels = predictor_obj.predict(img_tensor)
    
    print(f"✓ Detected {len(boxes[0])} objects")
    
    # Visualize and save
    if config.Config.SAVE_VISUALIZATIONS and len(boxes[0]) > 0:
        visualize.save_detection_image(
            img_tensor[0],
            boxes[0],
            labels[0],
            scores[0],
            class_names=config.Config.CLASS_NAMES,
            save_path=os.path.join(config.Config.VIS_DIR, 'demo_result.jpg')
        )
        print(f"✓ Demo visualization saved to: {config.Config.VIS_DIR}/demo_result.jpg")
else:
    print(f"⚠ Demo image not found at {demo_image_path}")

print("\n" + "="*80)
print("✅ All Done!")
print("="*80)
print(f"\n📁 Results saved to:")
print(f"   Checkpoints: {config.Config.SAVE_DIR}")
print(f"   Plots: {config.Config.PLOTS_DIR}")
print(f"   Visualizations: {config.Config.VIS_DIR}")
print("\n" + "="*80)
