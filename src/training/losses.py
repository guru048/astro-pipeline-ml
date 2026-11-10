"""
Loss functions for Siamese Network training.

ContrastiveLoss  – margin-based pairwise loss (Hadsell et al., 2006).
SiameseLoss      – α · contrastive + (1−α) · cross-entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    L = y · d² + (1−y) · max(margin − d, 0)²

    where d = ||emb1 − emb2||₂ and y = 1 for same-class pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        d = F.pairwise_distance(emb1, emb2)
        loss = pair_labels * d.pow(2) + (1 - pair_labels) * F.relu(self.margin - d).pow(2)
        return loss.mean()


class SiameseLoss(nn.Module):
    """Combined contrastive + cross-entropy loss."""

    def __init__(
        self,
        n_classes: int,
        margin: float = 1.0,
        alpha: float = 0.4,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.contrastive = ContrastiveLoss(margin)
        self.ce          = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha       = alpha

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        pair_labels: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_cont = self.contrastive(emb1, emb2, pair_labels)
        l_ce   = self.ce(logits, targets)
        total  = self.alpha * l_cont + (1 - self.alpha) * l_ce
        return total, l_cont, l_ce
