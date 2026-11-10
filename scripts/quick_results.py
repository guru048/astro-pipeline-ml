"""
Quick end-to-end demo of the astro-pipeline-ml pipeline.

Stages
------
1. Generate a synthetic catalogue (100 K sources, 4 classes).
2. Apply quality-control cuts (S/N, FWHM, ellipticity, etc.).
3. Cross-match catalogue A vs catalogue B using parallel KDTree workers.
4. Train a Siamese Network classifier with 5-fold stratified CV.
5. Report per-class and aggregate metrics; save to results/quick_results.json.

Run from project root:
    python3 scripts/quick_results.py
No package install required.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, roc_auc_score,
)

torch.manual_seed(42)
np.random.seed(42)

CLASSES      = {0: "star", 1: "galaxy", 2: "artifact", 3: "cosmic_ray"}
N_CLASSES    = 4
N_SOURCES    = 100_000
N_FOLDS      = 5
MAX_EPOCHS   = 150
PATIENCE     = 25
BATCH_SIZE   = 512
HIDDEN_DIM   = 128
EMBED_DIM    = 64
N_LAYERS     = 3
DROPOUT      = 0.2
LR           = 1e-3
MARGIN       = 1.0
ALPHA        = 0.4          # weight of contrastive loss

FEATURE_COLS = [
    "snr", "fwhm_arcsec", "ellipticity", "flux_radius",
    "class_star", "kron_radius", "mag_auto", "mag_err",
    "a_b_ratio", "g_r_color", "r_i_color", "background",
]


# ─── 1. Catalogue generation ──────────────────────────────────────────────────

def _gen_stars(n, rng):
    snr  = np.clip(rng.lognormal(2.5, 1.0, n), 5, 2000)
    return np.column_stack([
        snr,
        np.clip(rng.normal(1.2, 0.15, n), 0.8, 2.0),
        np.clip(rng.beta(1.5, 15, n), 0.0, 0.35),
        np.clip(rng.normal(2.0, 0.25, n), 1.0, 4.0),
        np.clip(rng.beta(12, 1.2, n), 0.6, 1.0),
        np.clip(rng.normal(3.5, 0.4, n), 2.0, 6.0),
        rng.uniform(18, 26, n),
        np.clip(1.0857 / snr, 0.0005, 0.5),
        np.clip(rng.normal(1.05, 0.05, n), 1.0, 1.4),
        np.clip(rng.normal(0.5, 0.3, n), -0.5, 2.0),
        np.clip(rng.normal(0.2, 0.2, n), -0.5, 1.5),
        np.clip(rng.lognormal(4.5, 0.3, n), 50, 500),
    ])

def _gen_galaxies(n, rng):
    snr = np.clip(rng.lognormal(2.0, 1.2, n), 3, 500)
    return np.column_stack([
        snr,
        np.clip(rng.lognormal(0.8, 0.4, n), 0.9, 12.0),
        np.clip(rng.beta(2.5, 3, n), 0.0, 0.92),
        np.clip(rng.lognormal(1.6, 0.5, n), 2.5, 40.0),
        np.clip(rng.beta(1.2, 10, n), 0.0, 0.5),
        np.clip(rng.lognormal(2.0, 0.5, n), 3.0, 30.0),
        rng.uniform(17, 27, n),
        np.clip(1.0857 / snr, 0.0005, 0.5),
        np.clip(rng.lognormal(0.3, 0.3, n), 1.0, 4.0),
        np.clip(rng.normal(0.7, 0.4, n), -0.3, 2.5),
        np.clip(rng.normal(0.35, 0.25, n), -0.3, 1.8),
        np.clip(rng.lognormal(4.5, 0.3, n), 50, 500),
    ])

def _gen_artifacts(n, rng):
    snr = np.clip(rng.uniform(5, 500, n), 5, 500)
    return np.column_stack([
        snr,
        np.clip(rng.uniform(0.5, 10, n), 0.5, 15.0),
        np.clip(rng.uniform(0.4, 0.95, n), 0.3, 0.98),
        np.clip(rng.uniform(3, 60, n), 2, 80.0),
        rng.uniform(0.1, 0.9, n),
        np.clip(rng.uniform(2, 40, n), 1.5, 60.0),
        rng.uniform(15, 25, n),
        np.clip(1.0857 / snr, 0.0005, 0.5),
        np.clip(rng.uniform(1.0, 8.0, n), 1.0, 10.0),
        rng.uniform(-1.0, 3.0, n),
        rng.uniform(-1.0, 2.5, n),
        np.clip(rng.lognormal(5.5, 0.5, n), 100, 2000),
    ])

def _gen_cosmic_rays(n, rng):
    snr = np.clip(rng.lognormal(4.0, 0.8, n), 20, 10000)
    return np.column_stack([
        snr,
        np.clip(rng.normal(0.4, 0.1, n), 0.1, 1.0),
        np.clip(rng.uniform(0.0, 0.95, n), 0.0, 0.98),
        np.clip(rng.normal(0.7, 0.15, n), 0.3, 2.0),
        rng.uniform(0.1, 0.9, n),
        np.clip(rng.normal(1.5, 0.3, n), 0.5, 4.0),
        rng.uniform(14, 22, n),
        np.clip(1.0857 / snr, 0.0005, 0.5),
        np.clip(rng.uniform(1.0, 5.0, n), 1.0, 8.0),
        rng.uniform(-2.0, 4.0, n),
        rng.uniform(-2.0, 3.0, n),
        np.clip(rng.lognormal(4.5, 0.3, n), 50, 500),
    ])

def generate_catalogue(n=N_SOURCES, seed=42, field=10.0):
    rng = np.random.default_rng(seed)
    fracs  = [0.40, 0.50, 0.07, 0.03]
    counts = [int(f * n) for f in fracs]
    counts[-1] = n - sum(counts[:-1])

    X = np.vstack([_gen_stars(counts[0], rng), _gen_galaxies(counts[1], rng),
                   _gen_artifacts(counts[2], rng), _gen_cosmic_rays(counts[3], rng)])
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    ra  = rng.uniform(0.0, field, n)
    dec = rng.uniform(0.0, field, n)

    idx = rng.permutation(n)
    X, y, ra, dec = X[idx], y[idx], ra[idx], dec[idx]

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["label"] = y.astype(int)
    df["ra"] = ra; df["dec"] = dec
    df["source_id"] = np.arange(n)
    return df


# ─── 2. Quality-control cuts ──────────────────────────────────────────────────

def apply_qc(df):
    mask = (
        (df["snr"]          >= 5.0)  &
        (df["snr"]          <= 5000) &
        (df["fwhm_arcsec"]  >= 0.5)  &
        (df["fwhm_arcsec"]  <= 8.0)  &
        (df["ellipticity"]  <= 0.95) &
        (df["flux_radius"]  >= 0.3)  &
        (df["flux_radius"]  <= 60.0) &
        (df["mag_err"]      <= 0.3)  &
        (df["background"]   <= 1500)
    )
    return df[mask].reset_index(drop=True)


# ─── 3. Parallel cross-match ──────────────────────────────────────────────────

def _match_strip(args):
    strip_ra, strip_dec, strip_ids, b_ra, b_dec, b_ids, radius_deg = args
    if len(strip_ra) == 0:
        return []
    tree = KDTree(np.column_stack([b_ra, b_dec]))
    coords = np.column_stack([strip_ra, strip_dec])
    dists, idxs = tree.query(coords, k=1, distance_upper_bound=radius_deg)
    matched = dists < radius_deg
    return [(strip_ids[i], b_ids[idxs[i]], dists[i] * 3600)
            for i in range(len(strip_ids)) if matched[i]]

def cross_match(cat_a, cat_b, radius_arcsec=1.0, n_workers=4):
    radius_deg = radius_arcsec / 3600.0
    n_strips   = n_workers
    ra_min, ra_max = cat_a["ra"].min(), cat_a["ra"].max()
    edges = np.linspace(ra_min, ra_max, n_strips + 1)

    args_list = []
    for i in range(n_strips):
        lo, hi = edges[i], edges[i + 1]
        m = (cat_a["ra"] >= lo) & (cat_a["ra"] <= hi)
        sub = cat_a[m]
        args_list.append((
            sub["ra"].values, sub["dec"].values, sub["source_id"].values,
            cat_b["ra"].values, cat_b["dec"].values, cat_b["source_id"].values,
            radius_deg,
        ))

    pairs = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_match_strip, args_list):
            pairs.extend(result)

    if pairs:
        matched_df = pd.DataFrame(pairs, columns=["id_a", "id_b", "sep_arcsec"])
        matched_df = matched_df.sort_values("sep_arcsec").drop_duplicates("id_b")
        n_matched  = len(matched_df)
        med_sep    = matched_df["sep_arcsec"].median()
    else:
        n_matched = 0
        med_sep   = 0.0

    return {
        "n_a": len(cat_a), "n_b": len(cat_b),
        "n_matched": n_matched,
        "n_unmatched_a": len(cat_a) - n_matched,
        "n_unmatched_b": len(cat_b) - n_matched,
        "median_sep_arcsec": float(med_sep),
        "match_rate": n_matched / len(cat_a),
    }


# ─── 4. Siamese Network ───────────────────────────────────────────────────────

class SiameseEncoder(nn.Module):
    def __init__(self, in_dim, hidden, embed, n_layers, drop):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop)]
        layers.append(nn.Linear(hidden, embed))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class SiameseNet(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN_DIM, embed=EMBED_DIM,
                 n_layers=N_LAYERS, drop=DROPOUT, n_cls=N_CLASSES):
        super().__init__()
        self.enc = SiameseEncoder(in_dim, hidden, embed, n_layers, drop)
        self.clf = nn.Linear(embed, n_cls)

    def forward(self, x):
        e = self.enc(x)
        return e, self.clf(e)

    def forward_pair(self, x1, x2):
        return self.enc(x1), self.enc(x2)


# ─── 5. Training ──────────────────────────────────────────────────────────────

def train_fold(X_tr, y_tr, X_val, y_val):
    n_input = X_tr.shape[1]
    counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    weights = torch.FloatTensor(1.0 / (counts + 1))
    weights = weights / weights.sum() * N_CLASSES

    model = SiameseNet(n_input)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS, eta_min=LR * 0.01)
    ce    = nn.CrossEntropyLoss(weight=weights)

    Xtr = torch.FloatTensor(X_tr); ytr = torch.LongTensor(y_tr)
    Xvl = torch.FloatTensor(X_val); yvl = torch.LongTensor(y_val)

    best_f1, best_state, patience_ctr = 0.0, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx]

            # Pair sampling within batch
            n  = len(xb)
            i1 = torch.randint(0, n, (n,))
            i2 = torch.randint(0, n, (n,))
            pl = (yb[i1] == yb[i2]).float()

            opt.zero_grad()
            e1, e2 = model.forward_pair(xb[i1], xb[i2])
            d      = F.pairwise_distance(e1, e2)
            l_cont = (pl * d.pow(2) + (1 - pl) * F.relu(MARGIN - d).pow(2)).mean()

            _, logits = model(xb)
            l_ce  = ce(logits, yb)
            loss  = ALPHA * l_cont + (1 - ALPHA) * l_ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                _, vl = model(Xvl)
                preds = vl.argmax(1).numpy()
            vf1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE // 5:
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embs, logits = model(Xvl)
        preds  = logits.argmax(1).numpy()
        probs  = torch.softmax(logits, dim=-1).numpy()
    return preds, probs, y_val


def run_cv(X, y):
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    all_preds, all_probs, all_true = [], [], []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        print(f"  fold {fold + 1}/{N_FOLDS} ...", end=" ", flush=True)
        t0 = time.time()

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr_idx])
        X_vl   = scaler.transform(X[vl_idx])

        preds, probs, y_vl = train_fold(X_tr, y[tr_idx], X_vl, y[vl_idx])
        all_preds.append(preds)
        all_probs.append(probs)
        all_true.append(y_vl)
        f1 = f1_score(y_vl, preds, average="macro", zero_division=0)
        print(f"macro F1 = {f1:.4f}  ({time.time()-t0:.1f}s)")

    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_preds)
    y_prob  = np.concatenate(all_probs)

    per_f1  = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                       average=None, zero_division=0)
    per_p   = precision_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                               average=None, zero_division=0)
    per_r   = recall_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                            average=None, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auc":         float(auc),
        **{f"precision_{CLASSES[i]}": float(per_p[i]) for i in range(N_CLASSES)},
        **{f"recall_{CLASSES[i]}":    float(per_r[i]) for i in range(N_CLASSES)},
        **{f"f1_{CLASSES[i]}":        float(per_f1[i]) for i in range(N_CLASSES)},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    output = {}

    # Stage 1: catalogue generation
    print(f"\n[1/4] Generating catalogue ({N_SOURCES:,} sources)...")
    t0   = time.time()
    cat  = generate_catalogue(n=N_SOURCES, seed=42)
    print(f"      Done in {time.time()-t0:.1f}s. Class distribution:")
    for label, name in CLASSES.items():
        n = (cat["label"] == label).sum()
        print(f"        {name:12s}: {n:6,}  ({n/len(cat):.1%})")

    # Stage 2: quality cuts
    print("\n[2/4] Applying quality-control cuts...")
    t0      = time.time()
    cat_qc  = apply_qc(cat)
    removed = len(cat) - len(cat_qc)
    print(f"      {len(cat_qc):,} sources retained ({removed:,} removed, {removed/len(cat):.1%}).")
    output["qc"] = {
        "n_input": len(cat),
        "n_output": len(cat_qc),
        "fraction_removed": removed / len(cat),
    }

    # Stage 3: cross-match
    print("\n[3/4] Cross-matching catalogue A vs catalogue B (parallel, 4 workers)...")
    # Catalogue B: same field, 95% overlap + 5% new sources, 5% missing
    rng  = np.random.default_rng(99)
    keep = rng.random(len(cat_qc)) < 0.95
    cat_b = cat_qc[keep].copy()
    cat_b["ra"]  += rng.normal(0, 0.0001, len(cat_b))   # small positional offset
    cat_b["dec"] += rng.normal(0, 0.0001, len(cat_b))
    cat_b["source_id"] = np.arange(len(cat_b))
    t0  = time.time()
    xm  = cross_match(cat_qc, cat_b, radius_arcsec=1.0, n_workers=4)
    dt  = time.time() - t0
    print(f"      Matched {xm['n_matched']:,}/{xm['n_a']:,} "
          f"({xm['match_rate']:.1%}) in {dt:.1f}s. "
          f"Median sep = {xm['median_sep_arcsec']:.4f}\"")
    output["cross_match"] = xm

    # Stage 4: Siamese Network, 5-fold CV
    print(f"\n[4/4] Training Siamese Network ({N_FOLDS}-fold CV, up to {MAX_EPOCHS} epochs/fold)...")
    X = cat_qc[FEATURE_COLS].values.astype(np.float32)
    y = cat_qc["label"].values.astype(int)
    cv_metrics = run_cv(X, y)

    print("\n── Results ────────────────────────────────")
    print(f"  Accuracy:          {cv_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {cv_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {cv_metrics['macro_f1']:.4f}")
    print(f"  Macro AUC:         {cv_metrics['macro_auc']:.4f}")
    print()
    for name in CLASSES.values():
        p = cv_metrics[f"precision_{name}"]
        r = cv_metrics[f"recall_{name}"]
        f = cv_metrics[f"f1_{name}"]
        print(f"  {name:12s}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    output["classification"] = cv_metrics

    path = os.path.join("results", "quick_results.json")
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
