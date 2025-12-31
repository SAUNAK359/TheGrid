# hybriddetector/main.py

import torch
import torch.nn as nn
from torch import optim
from pathlib import Path

from hybriddetector.backbone import cnn_backbone, transformer, fusion
from hybriddetector.backbone.resnet_backbone import ResNetBackbone
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
    def __init__(self, img_size: int | None = None):
        super(HybridDetector, self).__init__()
        if img_size is None:
            img_size = int(getattr(config.Config, "IMG_SIZE", 640))
        self.img_size = int(img_size)
        # Backbone
        backbone_name = str(getattr(config.Config, "BACKBONE", "resnet50")).lower().strip()
        backbone_pretrained = bool(getattr(config.Config, "BACKBONE_PRETRAINED", True))
        if backbone_name.startswith("resnet"):
            self.cnn = ResNetBackbone(name=backbone_name, pretrained=backbone_pretrained)
        else:
            self.cnn = cnn_backbone.CNNBackbone()

        # Transformer (speed-optimized): by default apply only on low-res feature map
        self.use_low_res_transformer_only = bool(getattr(config.Config, "TRANSFORMER_LOW_RES_ONLY", True))
        token_pool_factor = int(getattr(config.Config, "TOKEN_POOL_FACTOR", 1))

        strides = list(getattr(config.Config, "BACKBONE_STRIDES", [4, 8, 16]))
        if len(strides) != 3:
            strides = [4, 8, 16]

        if self.use_low_res_transformer_only:
            low_h = max(1, self.img_size // int(strides[2]))
            low_w = max(1, self.img_size // int(strides[2]))
            self.low_res_transformer = transformer.TransformerBlock(
                config.Config.BACKBONE_CHANNELS[2],
                low_h,
                low_w,
                num_heads=config.Config.TRANSFORMER_HEADS,
                token_pool_factor=token_pool_factor,
            )
            self.transformers = None
        else:
            self.low_res_transformer = None
            heights = [max(1, self.img_size // int(s)) for s in strides]
            widths = [max(1, self.img_size // int(s)) for s in strides]
            self.transformers = nn.ModuleList([
                transformer.TransformerBlock(
                    ch,
                    h,
                    w,
                    num_heads=config.Config.TRANSFORMER_HEADS,
                    token_pool_factor=token_pool_factor,
                )
                for ch, h, w in zip(config.Config.BACKBONE_CHANNELS, heights, widths)
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
        # FeatureFusion returns (high, medium, low) corresponding to P3/P4/P5.
        fused_high, fused_med, fused_low = self.fusion(f1, f2, f3)

        # Multi-scale heads (concatenate predictions across levels).
        boxes_h = self.box_head(fused_high)
        boxes_m = self.box_head(fused_med)
        boxes_l = self.box_head(fused_low)
        boxes = torch.cat([boxes_h, boxes_m, boxes_l], dim=1)

        obj_h = self.obj_head(fused_high)
        obj_m = self.obj_head(fused_med)
        obj_l = self.obj_head(fused_low)
        obj_scores = torch.cat([obj_h, obj_m, obj_l], dim=1)

        cls_h = self.cls_head(fused_high)
        cls_m = self.cls_head(fused_med)
        cls_l = self.cls_head(fused_low)
        class_probs = torch.cat([cls_h, cls_m, cls_l], dim=1)

        meta = {
            "levels": [
                {"h": int(fused_high.shape[-2]), "w": int(fused_high.shape[-1])},
                {"h": int(fused_med.shape[-2]), "w": int(fused_med.shape[-1])},
                {"h": int(fused_low.shape[-2]), "w": int(fused_low.shape[-1])},
            ]
        }

        return {'boxes': boxes, 'objectness': obj_scores, 'class_probs': class_probs, 'meta': meta}


__all__ = ["HybridDetector"]
