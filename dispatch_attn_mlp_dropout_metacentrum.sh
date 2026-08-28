#!/bin/bash
# Control experiment for the --ve-dropout sweep: apply dropout to the attention
# probabilities and the MLP hidden activations instead of the value embeddings,
# at the SAME rates and over the SAME seeds. Any difference in val bpb is then
# attributable to WHERE the dropout is applied rather than how much of it there is.
#
# Submits one job per (rate, seed) pair: 3 rates x 10 seeds = 30 jobs by default.
# Both --attn-dropout and --mlp-dropout are set to the same rate within a config.
#
# Usage:
#   ./dispatch_attn_mlp_dropout_metacentrum.sh [options]
#
# Options:
#   --rates "R1 R2 ..."   Dropout rates to sweep (default: "0.1 0.3 0.5")
#   --seeds START-END     Seed range, inclusive (default: 1-10)
#   --tag-prefix NAME     Prefix for run/model-tag/log names (default: attnmlp)
#   --monitor BOOL        0/1, enable monitorch logging (default: 0)
#   --dry-run             Print the qsub commands without submitting
#   -h, --help            Show this help
#
# Example:
#   ./dispatch_attn_mlp_dropout_metacentrum.sh --rates "0.1 0.3 0.5" --seeds 1-10
#
# NOTE: attention dropout runs on the SDPA path only - FA3 has no dropout support
# and base_train will assert rather than silently drop the regularization. The PBS
# script requests gpu_cap=sm_80 (A100), which is the SDPA path, so this is fine.
#
# NOTE: each job runs `python -m nanochat.dataset` and `tok_train` independently.
# Make sure those steps are idempotent / data is already cached, otherwise run a
# single prep job first and submit the seed jobs once it finishes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/base_train_metacentrum.sh"

RATES="0.1 0.3 0.5"
START=1
END=10
TAG_PREFIX="attnmlp"
MONITOR=0
DRY_RUN=0

usage() { sed -n '2,30p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rates)      RATES="$2"; shift 2 ;;
    --seeds)      IFS='-' read -r START END <<< "$2"; shift 2 ;;
    --tag-prefix) TAG_PREFIX="$2"; shift 2 ;;
    --monitor)    MONITOR="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Turn a rate into a tag-safe suffix matching the ve sweep's naming: 0.1 -> 01, 0.05 -> 005
rate_suffix() { echo "$1" | tr -d '.'; }

n_rates=$(echo "${RATES}" | wc -w)
n_seeds=$(( END - START + 1 ))
echo "Submitting ${n_rates} rate(s) x ${n_seeds} seed(s) = $(( n_rates * n_seeds )) jobs"
echo "  rates   = ${RATES}"
echo "  seeds   = ${START}..${END}"
echo "  monitor = ${MONITOR}"
echo

for rate in ${RATES}; do
  TAG="${TAG_PREFIX}_do$(rate_suffix "${rate}")"
  echo "--- ${TAG} (attn=${rate}, mlp=${rate}) ---"

  VARS="TAG=${TAG}"
  VARS+=",VE_GATE_RELU=0"
  VARS+=",VE_DROPOUT=0.0"
  VARS+=",ATTN_DROPOUT=${rate}"
  VARS+=",MLP_DROPOUT=${rate}"
  VARS+=",VE_GATE_MOMENTUM_START=0.85"
  VARS+=",VE_GATE_MOMENTUM_PEAK=0.97"
  VARS+=",VE_GATE_MOMENTUM_FINAL=0.90"
  VARS+=",MONITOR=${MONITOR}"

  for seed in $(seq "${START}" "${END}"); do
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  qsub -v ${VARS},SEED=${seed} -N nanochat-${TAG}-seed${seed} ${PBS_SCRIPT}"
    else
      jobid=$(qsub -v "${VARS},SEED=${seed}" -N "nanochat-${TAG}-seed${seed}" "${PBS_SCRIPT}")
      echo "  seed=${seed} -> ${jobid}"
    fi
  done
done
