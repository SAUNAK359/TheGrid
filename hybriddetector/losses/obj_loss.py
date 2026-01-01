# hybriddetector/losses/obj_loss.py

import torch
import torch.nn as nn

class ObjectnessLoss(nn.Module):
    """
    Binary Cross-Entropy / Focal loss for objectness prediction
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(ObjectnessLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        """
        logits: [B, N, 1]
        targets: [B, N, 1] float tensor 0/1
        """
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss  # [B,N,1]

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction != 'mean':
            return loss

        # Mean-reduction over all anchors heavily favors the trivial solution (obj≈0 everywhere)
        # because negatives vastly outnumber positives. Balance pos/neg contributions.
        flat_loss = loss.view(-1)
        flat_t = targets.view(-1)
        pos = flat_t > 0.5
        neg = ~pos

        if pos.any() and neg.any():
            pos_loss = flat_loss[pos].mean()
            neg_loss = flat_loss[neg].mean()
            # Weight negatives so pos and neg contribute similarly.
            pos_count = float(pos.sum().item())
            neg_count = float(neg.sum().item())
            neg_weight = pos_count / max(1.0, neg_count)
            return pos_loss + neg_weight * neg_loss

        if pos.any():
            return flat_loss[pos].mean()
        # No positives in batch.
        return flat_loss[neg].mean() if neg.any() else flat_loss.sum() * 0.0

if __name__ == "__main__":
    logits = torch.randn(2,3,1)
    targets = torch.tensor([[[1],[0],[1]], [[0],[1],[0]]], dtype=torch.float32)
    loss_fn = ObjectnessLoss()
    print("Objectness loss:", loss_fn(logits, targets))
