#!/bin/bash
#SBATCH --job-name=fm_benchmark
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=results/fm_benchmark_%j.log
#SBATCH --error=results/fm_benchmark_%j.err

set -euo pipefail

cd ~/org/the-benchmark

mkdir -p results

# Add nvidia pip wheel libs to LD_LIBRARY_PATH so JAX/XLA can find cuDNN etc.
_NVIDIA="${HOME}/org/the-benchmark/.venv/lib/python3.14/site-packages/nvidia"
for d in "${_NVIDIA}"/*/lib; do
    [ -d "${d}" ] && export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Date: $(date)"
echo ""

.venv/bin/python -m benchmark.run_forward_model \
    --n-warmup 5 \
    --n-repeat 50 \
    --output results/forward_model_${SLURM_JOB_ID}.json \
    --save-images

echo ""
echo "Done: $(date)"
