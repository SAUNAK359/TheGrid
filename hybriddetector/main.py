# hybriddetector/main.py

import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
from hybriddetector.backbone import cnn_backbone, transformer, fusion
from hybriddetector.heads import box_head, class_head, objectness_head
from hybriddetector.dataset import custom_dataset, transforms
from hybriddetector.trainer import train, scheduler, evaluate
from hybriddetector.inference import predictor, visualize
from hybriddetector.utils import config, seed, metrics

import os
from PIL import Image
import torchvision.transforms as T


def _get_device():
    seed.set_seed(42)
    device_type = config.Config.DEVICE if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    return device_type, device

class HybridDetector(torch.nn.Module):
    def __init__(self):
        super(HybridDetector, self).__init__()
        # Backbone
        self.cnn = cnn_backbone.CNNBackbone()

        # Transformer (speed-optimized): by default apply only on low-res feature map
        self.use_low_res_transformer_only = bool(getattr(config.Config, "TRANSFORMER_LOW_RES_ONLY", True))
        token_pool_factor = int(getattr(config.Config, "TOKEN_POOL_FACTOR", 1))

        if self.use_low_res_transformer_only:
            self.low_res_transformer = transformer.TransformerBlock(
                config.Config.BACKBONE_CHANNELS[2],
                40,
                40,
                num_heads=config.Config.TRANSFORMER_HEADS,
                token_pool_factor=token_pool_factor,
            )
            self.transformers = None
        else:
            self.low_res_transformer = None
            self.transformers = nn.ModuleList([
                transformer.TransformerBlock(
                    ch,
                    h,
                    w,
                    num_heads=config.Config.TRANSFORMER_HEADS,
                    token_pool_factor=token_pool_factor,
                )
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
        if self.use_low_res_transformer_only and self.low_res_transformer is not None:
            f3 = self.low_res_transformer(f3)
        elif self.transformers is not None:
            f1 = self.transformers[0](f1)
            f2 = self.transformers[1](f2)
            f3 = self.transformers[2](f3)
        # FeatureFusion returns (high, medium, low). Use medium (80x80 for 640 input)
        # as the detection feature map so head tensor shapes are consistent.
        _, fused_med, _ = self.fusion(f1, f2, f3)
        boxes = self.box_head(fused_med)
        obj_scores = self.obj_head(fused_med)
        class_probs = self.cls_head(fused_med)
        return {'boxes': boxes, 'objectness': obj_scores, 'class_probs': class_probs}


__all__ = ["HybridDetector"]
