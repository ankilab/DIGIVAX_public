#!/bin/bash
# GPU job for one array task. Submitted by run.sh, which sets partition, gres,
# time, array range and log path on the sbatch command line and passes
# RUNNER_DIR, OUT_DIR, NUM_TASKS, VERIFY and PASS2 via --export. run.sh --direct runs
# this same file with bash on an already allocated GPU node.
#
#SBATCH --job-name=digivax_run
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -uo pipefail
: "${RUNNER_DIR:?RUNNER_DIR not set — start the pipeline with run.sh}"
: "${OUT_DIR:?OUT_DIR not set — start the pipeline with run.sh}"
source "$RUNNER_DIR/config.env"

PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY — run install.sh first." >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "DIGIVAX run | task ${SLURM_ARRAY_TASK_ID:-0}/${NUM_TASKS:-1} | $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VERIFY_FLAG=()
[[ "${VERIFY:-1}" == "0" ]] && VERIFY_FLAG=(--no-verify)
[[ "${PASS2:-0}" == "1" ]] && VERIFY_FLAG+=(--pass2)

cd "$RUNNER_DIR"
py -m digivax_runner.infer --out-dir "$OUT_DIR" \
  --base-model "$BASE_MODEL_DIR" --adapter "$ADAPTER_DIR" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "${NUM_TASKS:-1}" "${VERIFY_FLAG[@]}"
RC=$?
echo "finished $(date) rc=$RC"
exit $RC
