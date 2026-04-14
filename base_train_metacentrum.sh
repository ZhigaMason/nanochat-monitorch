#!/bin/bash
#PBS -N nanochat-base-train
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=40gb:gpu_cap=sm_80:mem=64gb
#PBS -l walltime=24:00:00
#PBS -q gpu

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

cd /storage/brno2/home/$(whoami)/nanochat-monitorch

source .venv/bin/activate

python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --depth=12 --device-batch-size=4 --window-pattern=L --run=monitorch-small

# kindly stolen from https://github.com/karpathy/nanochat/discussions/677
