from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):
    """ResNet backbone that returns 3 feature maps for the detector.

    For an input image of size 640x640, the returned feature map strides are:
            - f_high: stride 8  (approx 80x80)   (P3)
            - f_med:  stride 16 (approx 40x40)  (P4)
            - f_low:  stride 32 (approx 20x20)  (P5)

        This is closer to modern YOLO defaults (multi-scale detection on P3/P4/P5).
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()

        name = str(name).lower().strip()
        if name not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
            raise ValueError(f"Unsupported ResNet backbone: {name}")

        weights = None
        if pretrained:
            if name == "resnet18":
                weights = torchvision.models.ResNet18_Weights.DEFAULT
            elif name == "resnet34":
                weights = torchvision.models.ResNet34_Weights.DEFAULT
            elif name == "resnet50":
                weights = torchvision.models.ResNet50_Weights.DEFAULT
            else:
                weights = torchvision.models.ResNet101_Weights.DEFAULT

        resnet = getattr(torchvision.models, name)(weights=weights)

        # Public metadata for downstream heads/fusion.
        # Returned feature maps are P3/P4/P5 with strides 8/16/32.
        self.strides = [8, 16, 32]
        if name in {"resnet18", "resnet34"}:
            self.out_channels = [128, 256, 512]
        else:
            self.out_channels = [512, 1024, 2048]

        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Stages
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        f_high = self.layer2(x)       # P3
        f_med = self.layer3(f_high)   # P4
        f_low = self.layer4(f_med)    # P5
        return f_high, f_med, f_low
