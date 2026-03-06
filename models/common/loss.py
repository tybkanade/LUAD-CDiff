# models/prototype.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBuffer(nn.Module):
    """
    Maintain class prototypes with EMA update.
    """
    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.99):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum

        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, feat_dim)
        )
        self.register_buffer(
            "initialized",
            torch.zeros(num_classes, dtype=torch.bool)
        )

    @torch.no_grad()
    def update(self, feat: torch.Tensor, label: torch.Tensor):
        """
        feat:  [B, D] (建议用 x0_real)
        label: [B]
        """
        for f, y in zip(feat, label):
            y = int(y.item())
            if not self.initialized[y]:
                self.prototypes[y] = f.detach()
                self.initialized[y] = True
            else:
                self.prototypes[y] = (
                    self.momentum * self.prototypes[y]
                    + (1 - self.momentum) * f.detach()
                )

    def forward(self):
        return self.prototypes


def prototype_softmax_loss(
    x: torch.Tensor,           # [B, D]
    label: torch.Tensor,       # [B]
    prototypes: torch.Tensor,  # [C, D]
):
    """
    Soft prototype loss: distance-based classification
    """
    # x: [B, D], proto: [C, D]
    dists = torch.cdist(x, prototypes)     # [B, C]
    loss = F.cross_entropy(-dists, label)
    return loss


def prototype_margin_loss(
    x: torch.Tensor,
    label: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 1.0
):
    B, D = x.shape
    loss = 0.0

    for i in range(B):
        y = label[i]
        pos = F.mse_loss(x[i], prototypes[y])

        neg_dists = []
        for c in range(prototypes.size(0)):
            if c != y:
                neg_dists.append(F.mse_loss(x[i], prototypes[c]))

        neg = torch.min(torch.stack(neg_dists))
        loss = loss + F.relu(pos - neg + margin)

    return loss / B

def angular_align_loss(x_gen, x_real, label, target_class=0, eps=1e-6):
    """
    x_gen:  [B, D]
    x_real: [B, D]  (detach outside!)
    label:  [B]
    """
    # cosine similarity
    cos_sim = F.cosine_similarity(x_gen, x_real, dim=-1)  # [B]
    loss = 1.0 - cos_sim                                  # [B]

    # 只对 target_class 生效
    mask = (label == target_class).float()                # [B]

    if mask.sum() < 1:
        # batch 里没有 0 类，直接返回 0，不影响训练
        return loss.mean() * 0.0

    loss = (loss * mask).sum() / (mask.sum() + eps)
    return loss



class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2:
            targets = targets.argmax(dim=-1)
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss


