#!/bin/bash
# Submit N base training jobs to MetaCentrum, one per seed.
# Each job gets its own --run name, --model-tag and --log-file (parametrized by
# SEED + TAG). All hyperparameters are forwarded to the PBS script via `qsub -v`.
#
# Usage:
#   ./dispatch_seeds_metacentrum.sh [options]
#
# Options:
#   --seeds START-END                Seed range, inclusive (default: 1-10)
#   --tag NAME                       Tag for run/model-tag/log file (default: default)
#   --ve-gate-relu BOOL              0/1 (default: 0)
#   --ve-dropout FLOAT               (default: 0.0)
#   --ve-gate-momentum-start FLOAT   (default: 0.85)
#   --ve-gate-momentum-peak  FLOAT   (default: 0.97)
#   --ve-gate-momentum-final FLOAT   (default: 0.90)
#   --monitor BOOL                   0/1, enable monitorch logging (default: 0)
#   -h, --help                       Show this help
#
# Example:
#   ./dispatch_seeds_metacentrum.sh --seeds 1-10 --tag dropout01 --ve-dropout 0.1
#
# Note: each job runs `python -m nanochat.dataset` and `tok_train` independently.
# Make sure those steps are idempotent / data is already cached, otherwise run a
# single prep job first and submit the seed jobs once it finishes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/base_train_metacentrum.sh"

# Defaults (kept in sync with base_train_metacentrum.sh)
START=1
END=10
TAG="default"
VE_GATE_RELU=0
VE_DROPOUT=0.0
VE_GATE_MOMENTUM_START=0.85
VE_GATE_MOMENTUM_PEAK=0.97
VE_GATE_MOMENTUM_FINAL=0.90
MONITOR=0

usage() { sed -n '2,21p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      IFS='-' read -r START END <<< "$2"; shift 2 ;;
    --tag)                      TAG="$2"; shift 2 ;;
    --ve-gate-relu)             VE_GATE_RELU="$2"; shift 2 ;;
    --ve-dropout)               VE_DROPOUT="$2"; shift 2 ;;
    --ve-gate-momentum-start)   VE_GATE_MOMENTUM_START="$2"; shift 2 ;;
    --ve-gate-momentum-peak)    VE_GATE_MOMENTUM_PEAK="$2"; shift 2 ;;
    --ve-gate-momentum-final)   VE_GATE_MOMENTUM_FINAL="$2"; shift 2 ;;
    --monitor)                  MONITOR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

VARS="TAG=${TAG}"
VARS+=",VE_GATE_RELU=${VE_GATE_RELU}"
VARS+=",VE_DROPOUT=${VE_DROPOUT}"
VARS+=",VE_GATE_MOMENTUM_START=${VE_GATE_MOMENTUM_START}"
VARS+=",VE_GATE_MOMENTUM_PEAK=${VE_GATE_MOMENTUM_PEAK}"
VARS+=",VE_GATE_MOMENTUM_FINAL=${VE_GATE_MOMENTUM_FINAL}"
VARS+=",MONITOR=${MONITOR}"

echo "Submitting seeds ${START}..${END} (tag=${TAG})"
echo "  VE_GATE_RELU=${VE_GATE_RELU} VE_DROPOUT=${VE_DROPOUT} MONITOR=${MONITOR}"
echo "  VE_GATE_MOMENTUM start/peak/final = ${VE_GATE_MOMENTUM_START}/${VE_GATE_MOMENTUM_PEAK}/${VE_GATE_MOMENTUM_FINAL}"
for seed in $(seq "${START}" "${END}"); do
  jobid=$(qsub -v "${VARS},SEED=${seed}" -N "nanochat-${TAG}-seed${seed}" "${PBS_SCRIPT}")
  echo "  seed=${seed} -> ${jobid}"
done
