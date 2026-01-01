# hybriddetector/losses/cls_loss.py

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Use -1 as background/ignore label for dense anchor targets.
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(self, logits, targets):
        """
        logits: [B, N, num_classes]
        targets: [B, N] long tensor
        """
        # Compute CE for all anchors; ignored targets (-1) contribute 0.
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        ce_loss = self.ce(flat_logits, flat_targets)  # [B*N]

        # Focal scaling (note: ignored anchors have ce_loss==0 -> focal term == 0).
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # IMPORTANT: do not average over all anchors (would be dominated by background/ignored).
        # Normalize over valid (non-ignored) anchors so the classification head actually learns.
        valid = flat_targets != -1
        if valid.any():
            loss = loss[valid]
        else:
            # No positives in this batch.
            loss = loss.sum() * 0.0

        if self.reduction == 'mean':
            denom = max(1, int(loss.numel()))
            return loss.sum() / float(denom)
        if self.reduction == 'sum':
            return loss.sum()
        return loss

if __name__ == "__main__":
    logits = torch.randn(2,3,5)   # B=2, N=3, num_classes=5
    targets = torch.tensor([[0,1,2],[2,3,4]])
    loss_fn = FocalLoss()
    print("Focal loss:", loss_fn(logits, targets))
