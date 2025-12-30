# hybriddetector/heads/box_head.py

import torch
import torch.nn as nn

class BoxHead(nn.Module):
    """
    Predict bounding boxes for each feature map cell.
    Outputs: [B, num_boxes, 4] -> [x_center, y_center, width, height]
    """
    def __init__(self, in_channels=256, num_anchors=3):
        super(BoxHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv(x)                  # [B, num_anchors*4, H, W]
        out = out.permute(0,2,3,1).contiguous()  # [B, H, W, num_anchors*4]
        out = out.view(B, -1, 4)            # [B, H*W*num_anchors, 4]
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    head = BoxHead(in_channels=256, num_anchors=3)
    boxes = head(x)
    print("Box output shape:", boxes.shape)  # [1, 80*80*3, 4]
