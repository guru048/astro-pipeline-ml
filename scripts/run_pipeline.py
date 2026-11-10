"""
Full pipeline runner.

Steps
-----
1. Load catalogue (parquet or generate if not found).
2. Apply quality-control cuts.
3. Add derived features.
4. Cross-match against a second catalogue (simulated offset copy).
5. Train Siamese Network with 5-fold CV.
6. Save results to JSON.

Usage
-----
python scripts/run_pipeline.py --config configs/pipeline.yaml --output results/
"""

import argparse, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd

from src.data.catalogue_generator import generate_catalogue
from src.data.quality_cuts import apply_quality_cuts, QCConfig
from src.data.cross_match import cross_match
from src.features.feature_engineering import add_derived_features, get_feature_columns
from src.training.trainer import train_cv, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", default=None)
    parser.add_argument("--config",    default="configs/pipeline.yaml")
    parser.add_argument("--output",    default="results/")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    os.makedirs(args.output, exist_ok=True)
    output = {}
    t0 = time.time()

    # 1. Catalogue
    if args.catalogue and os.path.exists(args.catalogue):
        print(f"Loading catalogue from {args.catalogue}")
        cat = pd.read_parquet(args.catalogue)
    else:
        c = cfg["catalogue"]
        print(f"Generating {c['n_sources']:,} sources ...")
        cat = generate_catalogue(n=c["n_sources"], seed=c["seed"],
                                 field_size_deg=c["field_size_deg"])
    print(f"  {len(cat):,} sources loaded.")

    # 2. QC
    qc_cfg = QCConfig(**cfg.get("quality_cuts", {}))
    cat_qc, qc_report = apply_quality_cuts(cat, qc_cfg)
    print(qc_report)
    output["qc"] = qc_report.__dict__

    # 3. Features
    cat_fe = add_derived_features(cat_qc)

    # 4. Cross-match (simulate cat_b as offset copy)
    import numpy as np
    rng = np.random.default_rng(99)
    keep = rng.random(len(cat_qc)) < 0.95
    cat_b = cat_qc[keep].copy()
    cat_b["ra"]  += rng.normal(0, 0.0001, len(cat_b))
    cat_b["dec"] += rng.normal(0, 0.0001, len(cat_b))
    cat_b["source_id"] = np.arange(len(cat_b))
    xm_cfg = cfg.get("cross_match", {})
    _, xm_result = cross_match(cat_qc, cat_b, **xm_cfg)
    print(xm_result)
    output["cross_match"] = xm_result.__dict__

    # 5. Train
    feat_cols = get_feature_columns(include_derived=True)
    X = cat_fe[feat_cols].values.astype("float32")
    y = cat_fe["label"].values.astype(int)

    tr_cfg = TrainConfig(**{k: v for k, v in cfg.get("training", {}).items()
                            if hasattr(TrainConfig, k)})
    cv = train_cv(X, y, tr_cfg, verbose=True)
    output["classification"] = cv.aggregate.to_dict()
    output["elapsed_seconds"] = time.time() - t0

    out_path = os.path.join(args.output, "results.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Macro F1: {cv.aggregate.macro_f1:.4f}")


if __name__ == "__main__":
    main()
