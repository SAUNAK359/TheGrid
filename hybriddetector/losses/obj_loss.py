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
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        return loss

if __name__ == "__main__":
    logits = torch.randn(2,3,1)
    targets = torch.tensor([[[1],[0],[1]], [[0],[1],[0]]], dtype=torch.float32)
    loss_fn = ObjectnessLoss()
    print("Objectness loss:", loss_fn(logits, targets))
