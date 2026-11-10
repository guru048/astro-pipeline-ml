# astro-pipeline-ml

An end-to-end, reproducible pipeline for processing large-scale astronomical catalogues and classifying sources with a Siamese Network. It handles the full journey from raw photometric data through quality control, parallel catalogue cross-matching, and deep-learning-based source separation — all in a single codebase ready to run on a laptop or scale to an HPC cluster via SLURM.

---

## What it does

| Stage | What happens |
|---|---|
| **Catalogue generation** | Synthetic 1 M-row catalogue (stars, galaxies, artefacts, cosmic rays) with realistic photometric distributions |
| **Quality-control cuts** | Sequential S/N, FWHM, ellipticity, flux-radius and background-level filters; configurable thresholds |
| **Feature engineering** | 12 raw + 12 derived features (log-scaled SNR, compactness, colour indices, concentration, stellarity-morphology coupling) |
| **Parallel cross-matching** | RA-strip decomposition + KDTree per strip, dispatched to `ProcessPoolExecutor` workers; scales linearly with CPU count |
| **Siamese Network** | Shared MLP encoder → L2-normalised embedding → classifier head; trained with combined contrastive + cross-entropy loss |
| **5-fold stratified CV** | Cosine-annealed Adam, early stopping, class-balanced weights |
| **Optuna HPO** | TPE sampler with median pruning over 7 hyperparameters |
| **SLURM scripts** | Array-job training (one job per fold) and full pipeline jobs ready to submit |
| **Docker** | Single-command reproducibility |

---

## Results (100 K sources, 5-fold CV)

These numbers come from running `python scripts/quick_results.py` on a laptop CPU.

| Metric | Value |
|---|---|
| Accuracy | **99.99 %** |
| Balanced Accuracy | **99.98 %** |
| Macro F1 | **99.97 %** |
| Macro AUC (OvR) | **100.00 %** |

**Per-class breakdown**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| star | 1.0000 | 1.0000 | 1.0000 |
| galaxy | 0.9999 | 0.9998 | 0.9998 |
| artifact | 0.9987 | 0.9993 | 0.9990 |
| cosmic\_ray | 1.0000 | 1.0000 | 1.0000 |

**Pipeline stats**

| Step | Result |
|---|---|
| Sources generated | 100,000 |
| Retained after QC | 77,433 (77.4 %) |
| Removed by QC | 22,567 (22.6 %) |
| Cross-match rate | 93.1 % |
| Median separation | 0.42 arcsec |

---

## Project structure

```
astro-pipeline-ml/
├── configs/
│   └── pipeline.yaml          # All tunable parameters in one place
├── scripts/
│   ├── quick_results.py       # Self-contained end-to-end demo (no install)
│   ├── generate_catalogue.py  # Save 1 M-row catalogue as parquet
│   └── run_pipeline.py        # Full pipeline runner
├── slurm/
│   ├── run_pipeline.sh        # SLURM: full pipeline job
│   └── run_training.sh        # SLURM: array job (one per CV fold)
├── src/
│   ├── data/
│   │   ├── catalogue_generator.py   # Synthetic catalogue (4 classes, 12 features)
│   │   ├── quality_cuts.py          # Sequential QC filters + QCReport
│   │   └── cross_match.py           # Parallel KDTree cross-matching
│   ├── features/
│   │   └── feature_engineering.py  # 12 derived features
│   ├── models/
│   │   └── siamese_net.py           # Encoder + classifier head
│   ├── training/
│   │   ├── losses.py                # ContrastiveLoss, SiameseLoss
│   │   └── trainer.py               # 5-fold CV trainer, early stopping
│   ├── evaluation/
│   │   └── metrics.py               # ClassificationMetrics dataclass
│   └── tuning/
│       └── optuna_search.py         # Optuna HPO entry point
├── tests/
│   └── test_pipeline.py       # Unit + integration tests (pytest)
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Quick start

```bash
git clone https://github.com/Ritikk048/astro-pipeline-ml.git
cd astro-pipeline-ml
pip install -r requirements.txt
python scripts/quick_results.py
```

This runs the full 4-stage pipeline on 100 K sources and prints the results above in about 2 minutes on a laptop CPU.

### Docker

```bash
docker build -t astro-pipeline .
docker run --rm astro-pipeline
```

### Full 1 M-source run

```bash
pip install -e .
python scripts/generate_catalogue.py --n-sources 1000000 --output data/catalogue.parquet
python scripts/run_pipeline.py --catalogue data/catalogue.parquet --config configs/pipeline.yaml
```

### HPC / SLURM

```bash
sbatch slurm/run_pipeline.sh
# or launch one job per CV fold in parallel:
sbatch slurm/run_training.sh
```

### Hyperparameter optimisation

```bash
pip install optuna
python -m src.tuning.optuna_search --n-trials 50 --n-folds 3 --n-sources 100000
```

---

## Model architecture

```
Input (12 raw + 12 derived features)
        │
  ┌─────▼──────────────────────┐
  │  Shared Encoder             │
  │  Linear → BN → ReLU → Drop  │  × 3 layers
  │  Linear → embed_dim (64)    │
  │  L2 normalise               │
  └─────┬──────────────────────┘
        │  embedding (unit hypersphere)
  ┌─────▼─────┐
  │ Classifier │   Linear(64 → 4)
  └─────┬─────┘
        │
      logits
```

**Training loss:**

```
L = 0.4 × ContrastiveLoss(pair) + 0.6 × CrossEntropyLoss(individual)
```

ContrastiveLoss (Hadsell et al., 2006):
same-class pairs minimise embedding distance; different-class pairs are pushed apart by margin 1.0.

---

## Techniques demonstrated

- **Large-scale data processing** — 1 M-row catalogue pipeline in pure Python/NumPy/pandas
- **Parallel computing** — `ProcessPoolExecutor` with RA-strip decomposition; throughput scales with CPU count
- **Siamese Networks** — combined contrastive + cross-entropy loss; L2-normalised embeddings on the unit hypersphere
- **Stratified cross-validation** — 5-fold CV with per-fold StandardScaler to prevent data leakage
- **Class imbalance** — inverse-frequency class weights in cross-entropy
- **Cosine annealing** — LR schedule with warm restarts for stable convergence
- **Optuna HPO** — TPE sampler + MedianPruner over 7 hyperparameters
- **HPC / SLURM** — array jobs for parallel fold training
- **Docker** — single-command reproducibility

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Configuration

All thresholds and hyperparameters live in [`configs/pipeline.yaml`](configs/pipeline.yaml). Override any value on the command line or by editing the file — no code changes needed.

---

## License

MIT © Guru
