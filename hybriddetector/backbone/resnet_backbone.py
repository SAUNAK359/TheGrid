from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):
    """ResNet backbone that returns 3 feature maps for the detector.

    For an input image of size 640x640, the returned feature map strides are:
      - f_high: stride 4  (approx 160x160)
      - f_med:  stride 8  (approx 80x80)
      - f_low:  stride 16 (approx 40x40)

    This matches the detector's expectation that the *medium* map is 80x80
    for 640-sized inputs (used by heads).
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

        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Stages
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16

        # Drop layer4 to keep 3 levels (and keep 40x40 as low-res)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f_high = self.layer1(x)
        f_med = self.layer2(f_high)
        f_low = self.layer3(f_med)
        return f_high, f_med, f_low
