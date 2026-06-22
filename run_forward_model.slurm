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

# NOTE: no `module load cudatoolkit` and no LD_LIBRARY_PATH editing.
# JAX is installed via pip `jax[cuda12]` and bundles its own CUDA 12.x + cuDNN
# wheels, which it locates automatically. Injecting a system CUDA toolkit or
# reordering LD_LIBRARY_PATH causes mismatched-library errors (e.g. the cuDNN
# 9.5.1-vs-9.8.0 `dnn_support` failure). Let the venv's wheels self-configure.

cd ~/org/the-benchmark
mkdir -p results

# --- Fail fast: confirm JAX actually sees the GPU and cuDNN works -----------
.venv/bin/python -c "
import jax, jax.numpy as jnp, jax.scipy.signal as jsp
devs = jax.devices()
print('devices:', devs)
assert devs[0].platform == 'gpu', f'expected a GPU, got {devs}'
x, k = jnp.ones((10, 10)), jnp.ones((3, 3))
out = jax.jit(lambda a, b: jsp.convolve2d(a, b, mode='same'))(x, k)
jax.block_until_ready(out)
print('conv test ok:', out.shape)
"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "Date:   $(date)"
echo ""

# --- Full benchmark ---------------------------------------------------------
.venv/bin/python -m benchmark.run_forward_model \
    --n-warmup 5 \
    --n-repeat 50 \
    --output results/forward_model_${SLURM_JOB_ID}.json \
    --save-images

echo ""
echo "Done: $(date)"
