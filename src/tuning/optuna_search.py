"""
Optuna hyperparameter optimisation for the Siamese Network.

Searches over:
  hidden_dim, embed_dim, n_layers, lr, dropout, margin, alpha

Usage
-----
python -m src.tuning.optuna_search --n-trials 50 --n-folds 3
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.training.trainer import TrainConfig, train_fold


def _objective(trial, X: np.ndarray, y: np.ndarray, n_folds: int = 3, seed: int = 42):
    cfg = TrainConfig(
        hidden_dim   = trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        embed_dim    = trial.suggest_categorical("embed_dim",  [32, 64, 128]),
        n_layers     = trial.suggest_int("n_layers",  2, 5),
        lr           = trial.suggest_float("lr",      1e-4, 1e-2, log=True),
        dropout      = trial.suggest_float("dropout", 0.1, 0.5),
        margin       = trial.suggest_float("margin",  0.5, 2.0),
        alpha        = trial.suggest_float("alpha",   0.1, 0.7),
        max_epochs   = 100,
        patience     = 15,
        n_folds      = n_folds,
        seed         = seed,
    )

    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    f1s  = []
    for tr_idx, vl_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr_idx]).astype("float32")
        X_vl   = scaler.transform(X[vl_idx]).astype("float32")
        result = train_fold(X_tr, y[tr_idx], X_vl, y[vl_idx], cfg)
        f1s.append(result.metrics.macro_f1)
        trial.report(float(np.mean(f1s)), step=len(f1s))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(f1s))


def run_hpo(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_folds:  int = 3,
    seed:     int = 42,
    output:   str = "results/best_hpo.json",
) -> dict:
    if not HAS_OPTUNA:
        raise ImportError("optuna is required. pip install optuna")

    sampler  = optuna.samplers.TPESampler(seed=seed)
    pruner   = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study    = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda t: _objective(t, X, y, n_folds, seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "n_trials": len(study.trials),
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"Best macro F1 = {best['value']:.4f}")
    print(f"Best params  = {best['params']}")
    print(f"Saved to {output}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds",  type=int, default=3)
    parser.add_argument("--n-sources", type=int, default=50_000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   default="results/best_hpo.json")
    args = parser.parse_args()

    from src.data.catalogue_generator import generate_catalogue
    from src.data.quality_cuts import apply_quality_cuts
    from src.features.feature_engineering import add_derived_features, get_feature_columns

    print(f"Generating catalogue ({args.n_sources:,} sources)...")
    cat     = generate_catalogue(n=args.n_sources, seed=args.seed)
    cat_qc, _ = apply_quality_cuts(cat)
    cat_fe  = add_derived_features(cat_qc)
    feat_cols = get_feature_columns(include_derived=True)
    X = cat_fe[feat_cols].values.astype("float32")
    y = cat_fe["label"].values.astype(int)

    run_hpo(X, y, n_trials=args.n_trials, n_folds=args.n_folds,
            seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
