#!/bin/bash
#SBATCH --job-name=astro-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

set -euo pipefail

module load python/3.11
source venv/bin/activate

echo "=== Generating 1M-source catalogue ==="
python scripts/generate_catalogue.py \
    --n-sources 1000000 \
    --output data/catalogue.parquet

echo "=== Running full pipeline ==="
python scripts/run_pipeline.py \
    --catalogue data/catalogue.parquet \
    --config configs/pipeline.yaml \
    --output results/

echo "Done. Results in results/"
