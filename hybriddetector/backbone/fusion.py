# hybriddetector/backbone/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, channels_high=64, channels_med=256, channels_low=512, out_channels=256):
        super(FeatureFusion, self).__init__()
        self.reduce_high = nn.Conv2d(channels_high, out_channels, 1)
        self.reduce_med  = nn.Conv2d(channels_med, out_channels, 1)
        self.reduce_low  = nn.Conv2d(channels_low, out_channels, 1)

        self.smooth_high = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_med  = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_low  = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, f_high, f_med, f_low):
        f_high_r = self.reduce_high(f_high)
        f_med_r  = self.reduce_med(f_med)
        f_low_r  = self.reduce_low(f_low)

        # Fuse low->med
        f_low_up = F.interpolate(f_low_r, size=f_med_r.shape[2:], mode='nearest')
        f_med_fused = f_med_r + f_low_up
        f_med_smooth = self.smooth_med(f_med_fused)

        # Fuse med->high
        f_med_up = F.interpolate(f_med_smooth, size=f_high_r.shape[2:], mode='nearest')
        f_high_fused = f_high_r + f_med_up
        f_high_smooth = self.smooth_high(f_high_fused)

        f_low_smooth = self.smooth_low(f_low_r)
        return f_high_smooth, f_med_smooth, f_low_smooth


if __name__ == "__main__":
    B = 1
    f_high = torch.randn(B,64,320,320)
    f_med  = torch.randn(B,256,80,80)
    f_low  = torch.randn(B,512,40,40)
    fusion = FeatureFusion()
    out_high, out_med, out_low = fusion(f_high, f_med, f_low)
    print(out_high.shape, out_med.shape, out_low.shape)
