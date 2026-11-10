"""
Unit and integration tests for the astro-pipeline-ml pipeline.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Data / QC ────────────────────────────────────────────────────────────────

def test_catalogue_generation():
    from src.data.catalogue_generator import generate_catalogue
    df = generate_catalogue(n=1000, seed=0)
    assert len(df) == 1000
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1, 2, 3})
    assert df["snr"].min() >= 0


def test_quality_cuts_remove_bad_sources():
    from src.data.catalogue_generator import generate_catalogue
    from src.data.quality_cuts import apply_quality_cuts, QCConfig
    df = generate_catalogue(n=5000, seed=1)
    # Inject a bad source
    df.loc[0, "snr"] = 0.5
    df.loc[1, "ellipticity"] = 0.99
    df.loc[2, "background"] = 2000.0
    qc_df, report = apply_quality_cuts(df)
    assert len(qc_df) < len(df)
    assert report.n_removed > 0
    assert (qc_df["snr"] >= 5.0).all()
    assert (qc_df["ellipticity"] <= 0.95).all()


def test_qc_config_custom():
    from src.data.catalogue_generator import generate_catalogue
    from src.data.quality_cuts import apply_quality_cuts, QCConfig
    df = generate_catalogue(n=1000, seed=2)
    cfg = QCConfig(snr_min=20.0)
    qc_df, _ = apply_quality_cuts(df, cfg)
    assert (qc_df["snr"] >= 20.0).all()


# ─── Cross-match ──────────────────────────────────────────────────────────────

def test_cross_match_basic():
    from src.data.cross_match import cross_match
    import pandas as pd
    # Perfect match
    n = 200
    rng = np.random.default_rng(7)
    ra  = rng.uniform(0, 5, n)
    dec = rng.uniform(0, 5, n)
    cat_a = pd.DataFrame({"ra": ra, "dec": dec, "source_id": np.arange(n)})
    cat_b = cat_a.copy()
    matched, result = cross_match(cat_a, cat_b, radius_arcsec=1.0, n_workers=2)
    assert result.n_matched > n * 0.9  # expect near-perfect match


def test_cross_match_no_overlap():
    from src.data.cross_match import cross_match
    import pandas as pd
    n = 100
    cat_a = pd.DataFrame({"ra": np.ones(n)*0, "dec": np.ones(n)*0, "source_id": np.arange(n)})
    cat_b = pd.DataFrame({"ra": np.ones(n)*90, "dec": np.ones(n)*0, "source_id": np.arange(n)})
    _, result = cross_match(cat_a, cat_b, radius_arcsec=1.0, n_workers=2)
    assert result.n_matched == 0


# ─── Feature engineering ──────────────────────────────────────────────────────

def test_add_derived_features():
    from src.data.catalogue_generator import generate_catalogue
    from src.features.feature_engineering import add_derived_features, get_feature_columns
    df = generate_catalogue(n=500, seed=3)
    df_fe = add_derived_features(df)
    expected = get_feature_columns(include_derived=True)
    for col in expected:
        assert col in df_fe.columns
    assert not df_fe[expected].isnull().any().any()


# ─── Siamese Network ──────────────────────────────────────────────────────────

def test_siamese_net_forward():
    import torch
    from src.models.siamese_net import SiameseNet
    model = SiameseNet(input_dim=12)
    x = torch.randn(8, 12)
    emb, logits = model(x)
    assert emb.shape == (8, 64)
    assert logits.shape == (8, 4)
    assert abs(emb.norm(dim=-1).mean().item() - 1.0) < 1e-4  # L2-normalised


def test_siamese_net_pair():
    import torch
    from src.models.siamese_net import SiameseNet
    model = SiameseNet(input_dim=12)
    x1 = torch.randn(8, 12)
    x2 = torch.randn(8, 12)
    e1, e2 = model.forward_pair(x1, x2)
    assert e1.shape == e2.shape == (8, 64)


def test_contrastive_loss():
    import torch
    from src.training.losses import ContrastiveLoss
    loss_fn = ContrastiveLoss(margin=1.0)
    e1 = torch.randn(16, 64)
    e2 = torch.randn(16, 64)
    labels = torch.zeros(16)
    labels[:8] = 1
    loss = loss_fn(e1, e2, labels)
    assert loss.item() >= 0


def test_siamese_loss():
    import torch
    from src.training.losses import SiameseLoss
    loss_fn = SiameseLoss(n_classes=4)
    e1  = torch.randn(16, 64)
    e2  = torch.randn(16, 64)
    pl  = torch.randint(0, 2, (16,)).float()
    logits = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    total, l_cont, l_ce = loss_fn(e1, e2, pl, logits, targets)
    assert total.item() >= 0


# ─── Metrics ──────────────────────────────────────────────────────────────────

def test_evaluate_metrics():
    from src.evaluation.metrics import evaluate
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, 200)
    y_pred = rng.integers(0, 4, 200)
    m = evaluate(y_true, y_pred)
    assert 0.0 <= m.accuracy <= 1.0
    assert 0.0 <= m.macro_f1 <= 1.0
    assert len(m.per_class_f1) == 4


# ─── End-to-end smoke test ────────────────────────────────────────────────────

def test_e2e_smoke():
    """Small end-to-end run: generate → QC → feature eng → 2-fold CV."""
    import torch
    from src.data.catalogue_generator import generate_catalogue
    from src.data.quality_cuts import apply_quality_cuts
    from src.features.feature_engineering import add_derived_features, get_feature_columns
    from src.training.trainer import train_cv, TrainConfig

    cat = generate_catalogue(n=2000, seed=99)
    cat_qc, _ = apply_quality_cuts(cat)
    cat_fe = add_derived_features(cat_qc)
    X = cat_fe[get_feature_columns()].values.astype("float32")
    y = cat_fe["label"].values.astype(int)

    cfg = TrainConfig(n_folds=2, max_epochs=10, patience=5, hidden_dim=32,
                      embed_dim=16, n_layers=2, batch_size=64)
    cv = train_cv(X, y, cfg, verbose=False)
    assert cv.aggregate.accuracy > 0.0
    assert cv.aggregate.macro_f1 >= 0.0
