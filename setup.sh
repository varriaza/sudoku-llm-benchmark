#!/usr/bin/env bash
set -euo pipefail

NVIDIA=0
ROCM=0
APPLE=0

for arg in "$@"; do
  case $arg in
    --nvidia) NVIDIA=1 ;;
    --rocm)   ROCM=1 ;;
    --apple)  APPLE=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── uv ────────────────────────────────────────────────────────────────────────
echo "==> Checking for uv..."
if ! command -v uv &>/dev/null; then
  echo "    uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "    uv $(uv --version)"

# ── Python env ────────────────────────────────────────────────────────────────
echo "==> Setting up Python environment..."
uv venv --python 3.11 --clear
uv pip install -e ".[dev]"
echo "    Python environment ready."

# ── Validate base install ─────────────────────────────────────────────────────
echo "==> Running smoke tests..."
uv run pytest tests/ -q --tb=short
echo "    All tests passed."

# ── GPU providers ─────────────────────────────────────────────────────────────
if [ $NVIDIA -eq 1 ]; then
  echo "==> Setting up Nvidia GPU support..."
  if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
  fi
  echo "    nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  uv run python -c "
from sudoku_bench.gpu_monitor import GPUMonitor
import time
m = GPUMonitor(poll_interval=0.1)
m.start(); time.sleep(0.2); stats = m.stop()
if stats.gpu_name:
    print(f'    GPU detected: {stats.gpu_name} ({stats.gpu_max_vram_mb} MB VRAM)')
else:
    print('    WARNING: nvidia-smi found but no GPU data returned.')
"
  echo "    NVIDIA=true" > .gpu_provider
  echo "    Nvidia setup complete."
fi

if [ $ROCM -eq 1 ]; then
  echo "==> Setting up AMD ROCm GPU support..."
  if ! command -v rocm-smi &>/dev/null; then
    echo "ERROR: rocm-smi not found. Install ROCm drivers first."
    exit 1
  fi
  echo "    rocm-smi found: $(rocm-smi --version 2>&1 | head -1)"
  echo "    NOTE: ROCm monitoring backend not yet implemented."
  echo "    ROCM=true" > .gpu_provider
  echo "    ROCm setup complete (monitoring stub)."
fi

if [ $APPLE -eq 1 ]; then
  echo "==> Setting up Apple Silicon GPU support..."
  if ! system_profiler SPDisplaysDataType &>/dev/null; then
    echo "ERROR: system_profiler not found. This flag requires macOS."
    exit 1
  fi
  echo "    Apple Silicon detected."
  echo "    NOTE: Apple Metal monitoring backend not yet implemented."
  echo "    APPLE=true" > .gpu_provider
  echo "    Apple Silicon setup complete (monitoring stub)."
fi

echo ""
echo "Setup complete. To generate puzzles:"
echo "  uv run sudoku-gen config.example.yaml"
echo ""
echo "To run the benchmark:"
echo "  uv run sudoku-bench config.yaml"
