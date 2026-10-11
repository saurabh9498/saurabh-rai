#!/bin/bash
# Hourly incremental retrain on a 32-node H100 cluster (BCM-provisioned, Slurm-scheduled).
# Designed job script; NOT run from this repo.
#
# Why Slurm, not Kubernetes: the embedding tables are sharded across all 256 GPUs and every
# step runs collectives over all ranks, so the job needs all 32 nodes at once or it cannot
# start (gang scheduling). Slurm allocates nodes all-or-nothing natively.
#
# Failure handling: BCM health checks drain a failing node in Slurm; --requeue restarts the
# job on the remaining nodes plus a hot spare, and training resumes from the newest
# checkpoint on the parallel filesystem.
#SBATCH --job-name=dlrm-hourly-retrain
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=00:58:00
#SBATCH --output=/shared/logs/retrain-%j.out

set -euo pipefail

export CKPT_DIR=/weka/checkpoints/dlrm          # parallel filesystem mounted on every node
export CKPT_EVERY_MIN=3                        # deadline-driven cadence (training/recovery_math.py)
export NCCL_IB_HCA=mlx5                        # InfiniBand for inter-node collectives
export NCCL_SOCKET_IFNAME=ib0

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun torchrun \
  --nnodes="$SLURM_NNODES" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$HEAD:29500" --rdzv-id="$SLURM_JOB_ID" \
  train_incremental.py \
    --resume-from-latest "$CKPT_DIR" \
    --checkpoint-every-min "$CKPT_EVERY_MIN" \
    --async-gds-checkpoint \
    --data-snapshot "iceberg://lake/clickstream@latest" \
    --replay-window-days 7 \
    --epochs 1 --lr-scale 0.1
