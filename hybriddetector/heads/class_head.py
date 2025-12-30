# hybriddetector/heads/class_head.py

import torch
import torch.nn as nn

class ClassHead(nn.Module):
    """
    Predict class probabilities for each anchor.
    Outputs: [B, num_anchors*H*W, num_classes]
    """
    def __init__(self, in_channels=256, num_anchors=3, num_classes=80):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv(x)                        # [B, num_anchors*num_classes, H, W]
        out = out.permute(0,2,3,1).contiguous()   # [B, H, W, num_anchors*num_classes]
        out = out.view(B, -1, self.num_classes)   # [B, H*W*num_anchors, num_classes]
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    head = ClassHead(in_channels=256, num_anchors=3, num_classes=20)
    cls_out = head(x)
    print("Class output shape:", cls_out.shape)  # [1, 80*80*3, 20]
