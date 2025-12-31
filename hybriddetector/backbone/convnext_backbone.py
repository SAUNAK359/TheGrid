from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone that returns 3 feature maps for the detector.

    Returns (f_high, f_med, f_low) with strides (8, 16, 32) relative to input.

    For ConvNeXt-Tiny/Small/Base/Large the typical stage dims are:
      - stride 4:  C1
      - stride 8:  C2  -> f_high
      - stride 16: C3  -> f_med
      - stride 32: C4  -> f_low

    This module is intentionally lightweight and avoids torchvision-version-specific
    assumptions beyond the standard convnext feature layout.
    """

    SUPPORTED = {"convnext_tiny", "convnext_small", "convnext_base", "convnext_large"}

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()

        name = str(name).lower().strip()
        if name not in self.SUPPORTED:
            raise ValueError(f"Unsupported ConvNeXt backbone: {name}")

        import torchvision

        weights = None
        if pretrained:
            # Best-effort across torchvision versions.
            weights_map = {
                "convnext_tiny": "ConvNeXt_Tiny_Weights",
                "convnext_small": "ConvNeXt_Small_Weights",
                "convnext_base": "ConvNeXt_Base_Weights",
                "convnext_large": "ConvNeXt_Large_Weights",
            }
            w_name = weights_map[name]
            w_cls = getattr(torchvision.models, w_name, None)
            if w_cls is not None:
                try:
                    weights = w_cls.DEFAULT
                except Exception:
                    weights = None

        model_fn = getattr(torchvision.models, name, None)
        if model_fn is None:
            raise RuntimeError(
                f"torchvision.models has no '{name}'. Update torchvision or choose a different backbone."
            )

        convnext = model_fn(weights=weights)

        # torchvision ConvNeXt has .features as a Sequential with 8 blocks:
        # [0]=stem(stride4), [1]=stage0, [2]=down->8, [3]=stage1,
        # [4]=down->16, [5]=stage2, [6]=down->32, [7]=stage3
        self.features = convnext.features

        # Strides and output channels are stable for torchvision ConvNeXt.
        self.strides = [8, 16, 32]
        if name == "convnext_tiny":
            self.out_channels = [192, 384, 768]
        elif name == "convnext_small":
            self.out_channels = [192, 384, 768]
        elif name == "convnext_base":
            self.out_channels = [256, 512, 1024]
        else:
            self.out_channels = [384, 768, 1536]

    def forward(self, x: torch.Tensor):
        # Run through features and capture stage outputs.
        # f_high: after stage1 (stride 8)
        # f_med : after stage2 (stride 16)
        # f_low : after stage3 (stride 32)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        f_high = self.features[3](x)
        x = self.features[4](f_high)
        f_med = self.features[5](x)
        x = self.features[6](f_med)
        f_low = self.features[7](x)
        return f_high, f_med, f_low
