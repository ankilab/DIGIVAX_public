#!/bin/bash
# Run the DIGIVAX extraction pipeline on a folder of vaccination-certificate images.
#
#   ./run.sh [folder] [options]
#
#   folder        name of a subfolder of images/ (e.g. test2), or any path;
#                 default: images/test1
#   -n N          only the first N images (sorted by file name)
#   -o DIR        output folder (default: output/<timestamp>_<folder name>);
#                 pass an existing one to resume — finished images are skipped
#   -p K          split over K parallel 1-GPU jobs (default: 1 per 450 images)
#   -t HH:MM:SS   walltime per job (default: estimated from the image count)
#   --pass2       re-read images whose DB lookup was ambiguous (second VLM pass)
#   --no-verify   skip the database verification
#   --direct      run here instead of submitting (when already on a GPU node)
#   -h            this help
#
# Pipeline per image: fine-tuned Qwen3.5-9B (timopt prompt) → reference-DB
# verification of every sticker → with --pass2, a second VLM pass where a
# lookup was ambiguous. The script submits a Slurm job, waits until a GPU is free and the
# job has finished, then prints where the results are. Ctrl-C cancels the job;
# images finished so far are kept, and `-o <output folder>` resumes the rest.
set -euo pipefail

# The whole script is one { … } block: bash parses it completely before running
# it, so editing run.sh while a run is still being watched cannot break that run.
{

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/config.env"
PY="$VENV_DIR/bin/python"

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die()   { echo "ERROR: $*" >&2; exit 1; }

# --- arguments ---------------------------------------------------------------
IMAGE_DIR=""; N=""; OUT=""; PARALLEL=""; WALLTIME=""; VERIFY=1; PASS2=0; DIRECT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    -p) PARALLEL="$2"; shift 2 ;;
    -t) WALLTIME="$2"; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    --pass2) PASS2=1; shift ;;
    --direct) DIRECT=1; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) [[ -z "$IMAGE_DIR" ]] || die "only one image folder allowed"; IMAGE_DIR="$1"; shift ;;
  esac
done
IMAGE_DIR="${IMAGE_DIR:-test1}"
# A bare name refers to a subfolder of images/; anything else is a normal path.
[[ -d "$RUNNER_DIR/images/$IMAGE_DIR" && "$IMAGE_DIR" != /* ]] && IMAGE_DIR="$RUNNER_DIR/images/$IMAGE_DIR"
[[ -d "$IMAGE_DIR" ]] || die "image folder not found: $IMAGE_DIR"
[[ -z "$N" || "$N" =~ ^[1-9][0-9]*$ ]] || die "-n needs a positive integer"
[[ -z "$PARALLEL" || "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || die "-p needs a positive integer"
[[ $VERIFY == 1 || $PASS2 == 0 ]] || die "--pass2 needs the database verification; drop --no-verify"

# --- installation present? ---------------------------------------------------
[[ -x "$PY" ]] || die "venv missing ($VENV_DIR) — run $RUNNER_DIR/install.sh first"
compgen -G "$BASE_MODEL_DIR/*.safetensors" > /dev/null \
  || die "base model missing ($BASE_MODEL_DIR) — run install.sh"
[[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] \
  || die "adapter missing ($ADAPTER_DIR) — run install.sh"

IMAGE_DIR="$(realpath "$IMAGE_DIR")"
OUT="$(realpath -m "${OUT:-$RUNNER_DIR/output/$(date +%Y%m%d_%H%M%S)_$(basename "$IMAGE_DIR")}")"
TECH="$OUT/technical"
if [[ $DIRECT -eq 0 && "$OUT" == /tmp/* || $DIRECT -eq 0 && "$OUT" == /dev/shm/* ]]; then
  die "output folder $OUT is node-local — the GPU job cannot write there; use a folder under \$WORK or \$HOME"
fi
mkdir -p "$TECH/logs"

# --- select images -----------------------------------------------------------
read -r SELECTED REMAINING < <(cd "$RUNNER_DIR" && py -m digivax_runner.prepare \
                                 "$IMAGE_DIR" "$OUT" ${N:+-n "$N"} \
                                 $([[ $DIRECT -eq 1 ]] && echo --allow-local)) || true
if [[ -z "${SELECTED:-}" ]]; then
  rmdir "$TECH/logs" "$TECH" "$OUT" 2>/dev/null || true   # drop the folder we just created if still empty
  exit 1
fi

summarize() {
  (cd "$RUNNER_DIR" && py -m digivax_runner.summarize "$OUT" > /dev/null)
  py - "$OUT" "$PASS2" <<'EOF'
import json, sys
s = json.load(open(f"{sys.argv[1]}/technical/summary.json"))
print(f"  images      : {s['images_done']}/{s['images_selected']} done"
      f"{', %d failed' % s['images_failed'] if s['images_failed'] else ''}"
      f"{', %d not processed' % s['images_not_processed'] if s['images_not_processed'] else ''}")
print(f"  records     : {s['records']}  ({s['images_no_records']} image(s) without records)")
print(f"  verification: {s['images_verified']} image(s) checked against the DB"
      + (f", second pass applied on {s['pass2_applied']}" if sys.argv[2] == "1" else "")
      if s["images_verified"] else "  verification: off")
sys.exit(0 if s["images_done"] == s["images_selected"] else 1)
EOF
}

finish() {
  echo
  echo "Results: $OUT"
  local rc=0; summarize || rc=$?
  echo "  FHIR        : $OUT/fhir/<image>.json"
  echo "  label       : $OUT/standard/<image>.json"
  echo "  card        : $OUT/digital_card/<image>.png"
  echo "  table       : $TECH/results.csv"
  [[ -s "$TECH/failed.txt" ]] && echo "  problems    : $TECH/failed.txt  (rerun with -o $OUT to retry)"
  echo "  logs        : $TECH/logs/"
  exit $rc
}

echo "DIGIVAX extraction"
echo "  images : $SELECTED selected from $IMAGE_DIR ($((SELECTED - REMAINING)) already done)"
echo "  output : $OUT"
echo "  model  : $(basename "$BASE_MODEL_DIR") + $(basename "$ADAPTER_DIR")  |  verification: $([[ $VERIFY == 1 ]] && echo on || echo off)$([[ $PASS2 == 1 ]] && echo "  |  pass 2: on")"

cat > "$TECH/run_config.json" <<EOF
{
  "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "image_dir": "$IMAGE_DIR", "n": "${N:-all}",
  "images_selected": $SELECTED, "images_remaining": $REMAINING,
  "base_model": "$BASE_MODEL_DIR", "adapter": "$ADAPTER_DIR", "prompt_set": "timopt",
  "verification": $([[ $VERIFY == 1 ]] && echo true || echo false),
  "pass2": $([[ $PASS2 == 1 ]] && echo true || echo false),
  "runner_commit": "$(git -C "$RUNNER_DIR" rev-parse --short HEAD 2>/dev/null || echo none)"
}
EOF

if [[ "$REMAINING" -eq 0 ]]; then
  echo "Nothing to do — every selected image already has a prediction."
  finish
fi

export RUNNER_DIR OUT_DIR="$OUT" VERIFY PASS2

# --- direct mode (already on a GPU node) -------------------------------------
if [[ $DIRECT -eq 1 ]]; then
  echo "Running directly on $(hostname)"
  NUM_TASKS=1 bash "$RUNNER_DIR/job.sbatch" 2>&1 | tee "$TECH/logs/direct_$(date +%Y%m%d_%H%M%S).log" || true
  finish
fi

command -v sbatch > /dev/null || die "sbatch not found — run on the Alex login node, or use --direct on a GPU node"

# --- submit ------------------------------------------------------------------
TASKS="${PARALLEL:-$(( (REMAINING + MAX_IMAGES_PER_TASK - 1) / MAX_IMAGES_PER_TASK ))}"
(( TASKS > REMAINING )) && TASKS=$REMAINING
PER_TASK=$(( (REMAINING + TASKS - 1) / TASKS ))
if [[ -z "$WALLTIME" ]]; then
  SECS=$(( SETUP_SECONDS + PER_TASK * SECONDS_PER_IMAGE ))
  if (( SECS > 86400 )); then
    SECS=86400
    echo "  NOTE   : $PER_TASK images per job may exceed the 24 h limit — use -p to split" \
         "further, or rerun with -o $OUT afterwards to finish the rest."
  fi
  MINS=$(( (SECS + 59) / 60 ))
  WALLTIME=$(printf '%02d:%02d:00' $((MINS / 60)) $((MINS % 60)))
fi

echo "  submit : sending the job to Slurm …"
# shellcheck disable=SC2086
JID=$(sbatch --parsable \
        --partition="$SLURM_PARTITION" --gres="$SLURM_GRES" --cpus-per-task="$SLURM_CPUS" \
        --time="$WALLTIME" --array="0-$((TASKS - 1))" \
        --output="$TECH/logs/slurm_%A_%a.log" \
        --export="ALL,RUNNER_DIR=$RUNNER_DIR,OUT_DIR=$OUT,NUM_TASKS=$TASKS,VERIFY=$VERIFY,PASS2=$PASS2" \
        $SLURM_EXTRA "$RUNNER_DIR/job.sbatch")
JID="${JID%%;*}"
echo "  job    : $JID — $TASKS × 1 GPU ($SLURM_PARTITION), ~$PER_TASK image(s) each, walltime $WALLTIME"
echo "  log    : $TECH/logs/slurm_${JID}_<task>.log"
echo
echo "Watching the job (Ctrl-C cancels it; finished images are kept)"

# watch.py prints the phases (waiting for a GPU, loading the model, each image) and
# returns when the job has left the queue. Ctrl-C reaches it directly (exit 130);
# the no-op trap only keeps bash alive so the job can be cancelled below.
trap ':' INT
WRC=0
(cd "$RUNNER_DIR" && py -m digivax_runner.watch "$OUT" "$JID" "$TASKS") || WRC=$?
trap - INT
if (( WRC == 130 )); then
  echo "Ctrl-C — cancelling job $JID …"
  scancel "$JID" || echo "  scancel failed — cancel it yourself: scancel $JID"
  echo "Cancelled. To process the remaining images later: $0 $IMAGE_DIR -o $OUT"
  finish
fi
(( WRC == 0 )) || exit "$WRC"

# sacct is not available to users on Alex, so read each task's outcome from its log:
# job.sbatch ends with "finished … rc=N"; Slurm writes "*** JOB … CANCELLED …" when it kills a task.
for ((t = 0; t < TASKS; t++)); do
  LOG="$TECH/logs/slurm_${JID}_${t}.log"
  if [[ ! -f "$LOG" ]]; then
    echo "  task $t: never started (cancelled while pending?)"
  elif grep -q "^finished .* rc=0$" "$LOG"; then
    echo "  task $t: completed"
  elif MSG=$(grep -m1 -o "CANCELLED AT .*" "$LOG"); then
    echo "  task $t: killed — $MSG  (log: $LOG)"
  elif MSG=$(grep -m1 -o "rc=[0-9]*$" "$LOG"); then
    echo "  task $t: finished with errors ($MSG) — see $LOG"
  else
    echo "  task $t: ended without finishing (out of memory or node failure?) — see $LOG"
  fi
done
finish
}
