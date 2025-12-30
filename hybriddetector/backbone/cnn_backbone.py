# hybriddetector/backbone/cnn_backbone.py

import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    """
    Custom CNN backbone producing 3-level features: high, medium, low resolution
    """
    def __init__(self):
        super(CNNBackbone, self).__init__()
        # High-res features
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 640 -> 320
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Medium-res features
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, stride=4, padding=1),  # 320 -> 80
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Low-res features
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 80 -> 40
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 640, 640)
    model = CNNBackbone()
    f1, f2, f3 = model(dummy_input)
    print("High-res:", f1.shape)
    print("Med-res :", f2.shape)
    print("Low-res :", f3.shape)
