#!/bin/bash
#SBATCH --job-name=astro-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --array=0-4         # one job per CV fold

set -euo pipefail

module load python/3.11
source venv/bin/activate

FOLD=${SLURM_ARRAY_TASK_ID}
echo "=== Training fold ${FOLD} ==="

python scripts/run_pipeline.py \
    --catalogue data/catalogue_qc.parquet \
    --config configs/pipeline.yaml \
    --fold "${FOLD}" \
    --output "results/fold_${FOLD}/"

echo "Fold ${FOLD} done."
