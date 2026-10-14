#!/bin/bash
# One-time setup on the Alex login node (needs internet; no GPU).
#
#   ./install.sh            create venv, fetch base model + adapter, self-check
#   ./install.sh --force    recreate the venv from scratch
#
# Idempotent: steps that are already done are skipped. Paths: config.env.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUNNER_DIR/config.env"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

say() { printf '\n== %s\n' "$*"; }

mkdir -p "$DIGIVAX_HOME" "$HF_HOME" "$PIP_CACHE_DIR"

# --- 1. Python venv ----------------------------------------------------------
PY="$VENV_DIR/bin/python"
if [[ $FORCE -eq 1 && -d "$VENV_DIR" ]]; then
  say "Removing existing venv $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ -x "$PY" ]] && py -c "import torch, transformers, peft, fla" 2>/dev/null; then
  say "venv OK: $VENV_DIR"
else
  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3.12 || command -v python3)"
  fi
  say "Creating venv $VENV_DIR with $PYTHON_BIN ($("$PYTHON_BIN" -V))"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  py -m pip install --upgrade pip
  say "Installing torch (CUDA 13.0 build) — a few GB, takes a while"
  py -m pip install torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu130
  say "Installing requirements.txt"
  py -m pip install -r "$RUNNER_DIR/requirements.txt"
fi

# --- 2. Base model -----------------------------------------------------------
if compgen -G "$BASE_MODEL_DIR/*.safetensors" > /dev/null; then
  say "Base model OK: $BASE_MODEL_DIR"
else
  say "Downloading $BASE_MODEL_REPO → $BASE_MODEL_DIR (~19 GB)"
  py -c "from huggingface_hub import snapshot_download as d; d('$BASE_MODEL_REPO', local_dir='$BASE_MODEL_DIR')"
fi

# --- 3. Fine-tuned adapter ---------------------------------------------------
if [[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  say "Adapter OK: $ADAPTER_DIR"
else
  if [[ ! -f "$ADAPTER_SOURCE/adapter_model.safetensors" ]]; then
    echo "ERROR: adapter not found at ADAPTER_SOURCE=$ADAPTER_SOURCE" >&2
    echo "       Point ADAPTER_SOURCE (config.env) at a LoRA adapter directory." >&2
    exit 1
  fi
  say "Copying adapter $ADAPTER_SOURCE → $ADAPTER_DIR"
  mkdir -p "$ADAPTER_DIR"
  cp -L "$ADAPTER_SOURCE/adapter_config.json" "$ADAPTER_SOURCE/adapter_model.safetensors" "$ADAPTER_DIR/"
fi

# --- 4. Self-check (CPU only) ------------------------------------------------
say "Self-check"
cd "$RUNNER_DIR"
py - <<EOF
import json, torch, transformers, peft
from transformers import AutoProcessor
from digivax_runner.mcp_tools import lookup_vaccine
print(f"torch {torch.__version__} (CUDA {torch.version.cuda})  transformers {transformers.__version__}  peft {peft.__version__}")
AutoProcessor.from_pretrained("$BASE_MODEL_DIR", trust_remote_code=True)
print("processor loads from $BASE_MODEL_DIR")
cfg = json.load(open("$ADAPTER_DIR/adapter_config.json"))
print(f"adapter: LoRA r={cfg['r']} alpha={cfg['lora_alpha']}, {len(cfg['target_modules'])} target module types")
assert lookup_vaccine("Priorix")["found"], "reference DB lookup failed"
print("reference DB OK")
EOF

say "Installation complete. Run:  $RUNNER_DIR/run.sh <image_folder> [-n N]"
