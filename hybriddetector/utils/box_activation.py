from __future__ import annotations

import torch


def squash01_tanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Squash logits to [0, 1] using a tanh mapping.

    This is a sigmoid-free alternative that still produces bounded outputs.
    """
    y = 0.5 * (torch.tanh(x) + 1.0)
    return y.clamp(min=0.0, max=1.0)


def decode_boxes_cxcywh(raw: torch.Tensor, eps_wh: float = 1e-6) -> torch.Tensor:
    """Decode raw box head outputs into normalized (xc, yc, w, h) in [0,1].

    Uses tanh-based squashing for all 4 channels and ensures w/h are positive.
    """
    out = squash01_tanh(raw)
    # Ensure strictly positive size to avoid degenerate boxes downstream.
    out[..., 2:4] = out[..., 2:4].clamp(min=eps_wh, max=1.0)
    return out
