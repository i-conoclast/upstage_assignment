import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        else: 
            return focal_loss.sum()


