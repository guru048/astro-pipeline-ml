"""
Siamese Network for astronomical source classification.

Architecture
------------
Shared encoder: MLP with BatchNorm → ReLU → Dropout, final L2-normalised embedding.
Classifier head: single linear layer on top of the embedding.

Training uses a combined loss:
  L = α · ContrastiveLoss(pair) + (1 − α) · CrossEntropyLoss(individual)

At inference only the encoder + head are used; no pairs needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SiameseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.2,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder    = SiameseEncoder(input_dim, hidden_dim, embed_dim, n_layers, dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-source forward pass. Returns (embedding, logits)."""
        emb    = self.encoder(x)
        logits = self.classifier(emb)
        return emb, logits

    def forward_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair forward pass for contrastive training. Returns (emb1, emb2)."""
        return self.encoder(x1), self.encoder(x2)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
