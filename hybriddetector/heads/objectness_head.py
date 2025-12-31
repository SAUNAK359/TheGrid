# hybriddetector/heads/objectness_head.py

import torch
import torch.nn as nn

from .blocks import ConvBNAct

class ObjectnessHead(nn.Module):
    """
    Predict objectness score for each anchor.
    Outputs: [B, num_anchors*H*W, 1]
    """
    def __init__(self, in_channels=256, num_anchors=3):
        super(ObjectnessHead, self).__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, act="silu"),
            ConvBNAct(in_channels, in_channels, kernel_size=3, act="silu"),
        )
        self.pred = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.stem(x)
        out = self.pred(x)                 # [B, num_anchors, H, W]
        out = out.permute(0,2,3,1).contiguous()  # [B, H, W, num_anchors]
        out = out.view(B, -1, 1)           # [B, H*W*num_anchors, 1]
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    head = ObjectnessHead(in_channels=256, num_anchors=3)
    obj_out = head(x)
    print("Objectness output shape:", obj_out.shape)  # [1, 80*80*3, 1]
