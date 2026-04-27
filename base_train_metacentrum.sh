#!/bin/bash
#PBS -N nanochat-base-train
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=40gb:gpu_cap=sm_80:mem=64gb
#PBS -l walltime=24:00:00
#PBS -q gpu

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Hyperparameters: pass via `qsub -v KEY=VAL,...`. Defaults match the current sweep.
SEED="${SEED:-42}"
TAG="${TAG:-default}"
VE_GATE_RELU="${VE_GATE_RELU:-0}"               # 1 = enable, 0 = disable (base_train.py default: off)
VE_DROPOUT="${VE_DROPOUT:-0.0}"
VE_GATE_MOMENTUM_START="${VE_GATE_MOMENTUM_START:-0.85}"
VE_GATE_MOMENTUM_PEAK="${VE_GATE_MOMENTUM_PEAK:-0.97}"
VE_GATE_MOMENTUM_FINAL="${VE_GATE_MOMENTUM_FINAL:-0.90}"
MONITOR="${MONITOR:-0}"                         # 1 = enable monitorch logging, 0 = disable

echo "=== Job hyperparameters ==="
echo "  SEED                    = ${SEED}"
echo "  TAG                     = ${TAG}"
echo "  VE_GATE_RELU            = ${VE_GATE_RELU}"
echo "  VE_DROPOUT              = ${VE_DROPOUT}"
echo "  VE_GATE_MOMENTUM_START  = ${VE_GATE_MOMENTUM_START}"
echo "  VE_GATE_MOMENTUM_PEAK   = ${VE_GATE_MOMENTUM_PEAK}"
echo "  VE_GATE_MOMENTUM_FINAL  = ${VE_GATE_MOMENTUM_FINAL}"
echo "  MONITOR                 = ${MONITOR}"
echo "==========================="

cd /storage/brno2/home/$(whoami)/nanochat-monitorch

source .venv/bin/activate

python -m nanochat.dataset -n 8
python -m scripts.tok_train

EXTRA_ARGS=()
if [[ "${VE_GATE_RELU}" == "1" || "${VE_GATE_RELU,,}" == "true" ]]; then
  EXTRA_ARGS+=(--ve-gate-relu)
fi
if [[ "${MONITOR}" == "1" || "${MONITOR,,}" == "true" ]]; then
  EXTRA_ARGS+=(--monitor)
fi

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --depth=12 --device-batch-size=4 --window-pattern=L \
  --run="monitorch-small-${TAG}-seed${SEED}" \
  --model-tag="d12-${TAG}-seed${SEED}" \
  --log-file="logs-${TAG}-seed${SEED}.pkl" \
  --seed="${SEED}" \
  --ve-dropout="${VE_DROPOUT}" \
  --ve-gate-momentum-start="${VE_GATE_MOMENTUM_START}" \
  --ve-gate-momentum-peak="${VE_GATE_MOMENTUM_PEAK}" \
  --ve-gate-momentum-final="${VE_GATE_MOMENTUM_FINAL}" \
  "${EXTRA_ARGS[@]}"

# kindly stolen from https://github.com/karpathy/nanochat/discussions/677
