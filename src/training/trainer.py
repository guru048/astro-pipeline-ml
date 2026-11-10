"""
Siamese Network trainer with 5-fold stratified cross-validation.

Usage
-----
from src.training.trainer import train_cv
metrics = train_cv(X, y)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.siamese_net import SiameseNet
from src.training.losses import SiameseLoss
from src.evaluation.metrics import evaluate, ClassificationMetrics


@dataclass
class TrainConfig:
    hidden_dim:  int   = 128
    embed_dim:   int   = 64
    n_layers:    int   = 3
    dropout:     float = 0.2
    lr:          float = 1e-3
    weight_decay: float = 1e-4
    batch_size:  int   = 512
    max_epochs:  int   = 200
    patience:    int   = 20
    margin:      float = 1.0
    alpha:       float = 0.4
    n_folds:     int   = 5
    seed:        int   = 42
    n_classes:   int   = 4


@dataclass
class FoldResult:
    fold:    int
    metrics: ClassificationMetrics
    y_true:  np.ndarray
    y_pred:  np.ndarray
    y_prob:  np.ndarray


@dataclass
class CVResult:
    fold_results:    list[FoldResult] = field(default_factory=list)
    aggregate:       ClassificationMetrics | None = None

    def macro_f1(self) -> float:
        return self.aggregate.macro_f1 if self.aggregate else 0.0


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / (counts + 1)
    w = w / w.sum() * n_classes
    return torch.FloatTensor(w)


def _train_epoch(
    model: SiameseNet,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: SiameseLoss,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> float:
    model.train()
    perm = torch.randperm(len(X))
    total_loss = 0.0
    n_batches  = 0
    for i in range(0, len(X), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X[idx], y[idx]
        n = len(xb)

        # Random intra-batch pairs
        i1 = torch.randint(0, n, (n,))
        i2 = torch.randint(0, n, (n,))
        pair_labels = (yb[i1] == yb[i2]).float()

        optimizer.zero_grad()
        emb1, emb2 = model.forward_pair(xb[i1], xb[i2])
        _, logits  = model(xb)
        loss, _, _ = criterion(emb1, emb2, pair_labels, logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(model: SiameseNet, X: torch.Tensor, y: np.ndarray):
    model.eval()
    embs, logits = model(X)
    preds = logits.argmax(1).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    return evaluate(y, preds, probs)


def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
    cfg: TrainConfig,
) -> FoldResult:
    torch.manual_seed(cfg.seed)
    n_input = X_tr.shape[1]
    weights = _class_weights(y_tr, cfg.n_classes)

    model = SiameseNet(
        input_dim=n_input,
        hidden_dim=cfg.hidden_dim,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        n_classes=cfg.n_classes,
    )
    criterion = SiameseLoss(
        n_classes=cfg.n_classes,
        margin=cfg.margin,
        alpha=cfg.alpha,
        class_weights=weights,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.max_epochs, eta_min=cfg.lr * 0.01
    )

    Xtr = torch.FloatTensor(X_tr)
    ytr = torch.LongTensor(y_tr)
    Xvl = torch.FloatTensor(X_vl)

    best_f1, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(cfg.max_epochs):
        _train_epoch(model, Xtr, ytr, criterion, optimizer, cfg.batch_size)
        scheduler.step()

        if epoch % 5 == 0 or epoch == cfg.max_epochs - 1:
            m = _validate(model, Xvl, y_vl)
            if m.macro_f1 > best_f1:
                best_f1 = m.macro_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience // 5:
                    break

    model.load_state_dict(best_state)
    final = _validate(model, Xvl, y_vl)
    model.eval()
    with torch.no_grad():
        _, logits = model(Xvl)
        preds = logits.argmax(1).numpy()
        probs = torch.softmax(logits, dim=-1).numpy()

    return FoldResult(fold=0, metrics=final, y_true=y_vl, y_pred=preds, y_prob=probs)


def train_cv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> CVResult:
    """Run n-fold stratified CV, return aggregated metrics."""
    if cfg is None:
        cfg = TrainConfig()

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    cv  = CVResult()

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"  fold {fold + 1}/{cfg.n_folds} ...", end=" ", flush=True)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr_idx]).astype(np.float32)
        X_vl   = scaler.transform(X[vl_idx]).astype(np.float32)

        result      = train_fold(X_tr, y[tr_idx], X_vl, y[vl_idx], cfg)
        result.fold = fold
        cv.fold_results.append(result)

        if verbose:
            print(f"macro F1 = {result.metrics.macro_f1:.4f}")

    all_true = np.concatenate([r.y_true for r in cv.fold_results])
    all_pred = np.concatenate([r.y_pred for r in cv.fold_results])
    all_prob = np.concatenate([r.y_prob for r in cv.fold_results])
    cv.aggregate = evaluate(all_true, all_pred, all_prob)
    return cv
