#!/usr/bin/env bash
# Save only the three options we change so restoring them doesn't use eval
_setup_e=$(set -o | awk '/^errexit /{print $2}')
_setup_u=$(set -o | awk '/^nounset /{print $2}')
_setup_p=$(set -o | awk '/^pipefail /{print $2}')
set -euo pipefail

NVIDIA=0
ROCM=0
APPLE=0

for arg in "$@"; do
  case $arg in
    --nvidia)   NVIDIA=1 ;;
    --rocm)     ROCM=1 ;;
    --apple)    APPLE=1 ;;
    *) echo "Unknown option: $arg"; [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1 ;;
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

# ── Native llama-server (built from source) ───────────────────────────────────
echo "==> Setting up native llama-server..."
if command -v llama-server &>/dev/null; then
  echo "    llama-server already installed."
else
  for tool in git cmake; do
    if ! command -v "$tool" &>/dev/null; then
      echo "ERROR: $tool is required to build llama-server. Install it first."
      [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1
    fi
  done

  LLAMA_REPO="ggml-org/llama.cpp"
  echo "    Fetching latest release tag..."
  LLAMA_TAG=$(curl -fsSL "https://api.github.com/repos/${LLAMA_REPO}/releases/latest" \
    | grep '"tag_name"' | head -1 | sed 's/.*"\(b[0-9]*\)".*/\1/')

  if [ -z "$LLAMA_TAG" ]; then
    echo "ERROR: Could not fetch llama.cpp release info."
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1
  fi

  LLAMA_LIB_DIR="$HOME/.local/lib/llama.cpp"
  LLAMA_BIN_DIR="$HOME/.local/bin"
  mkdir -p "$LLAMA_LIB_DIR" "$LLAMA_BIN_DIR"

  _src_dir=$(mktemp -d)
  echo "    Cloning llama.cpp ${LLAMA_TAG}..."
  git clone --depth=1 --branch "$LLAMA_TAG" \
    "https://github.com/${LLAMA_REPO}.git" "$_src_dir"

  # Add CUDA toolkit to PATH if not already there
  if ! command -v nvcc &>/dev/null; then
    for _cuda_dir in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
      if [ -x "$_cuda_dir/nvcc" ]; then
        export PATH="$_cuda_dir:$PATH"
        break
      fi
    done
  fi

  CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=ON"
  if command -v nvidia-smi &>/dev/null; then
    if command -v nvcc &>/dev/null; then
      echo "    Building with CUDA support ($(nvcc --version | grep release | awk '{print $5}' | tr -d ,))..."
      CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
    else
      echo "    WARNING: nvidia-smi found but nvcc not found — building CPU-only."
      echo "    Install the CUDA toolkit to enable GPU support."
    fi
  else
    echo "    Building CPU-only..."
  fi

  cmake -B "$_src_dir/build" -S "$_src_dir" $CMAKE_FLAGS
  cmake --build "$_src_dir/build" --config Release \
    -j"$(nproc)" --target llama-server

  find "$_src_dir/build" -name "llama-server" -type f \
    -exec cp {} "$LLAMA_LIB_DIR/llama-server" \;
  find "$_src_dir/build" \( -name "*.so" -o -name "*.so.*" \) \( -type f -o -type l \) \
    -exec cp -P {} "$LLAMA_LIB_DIR/" \;
  chmod +x "$LLAMA_LIB_DIR/llama-server"
  rm -rf "$_src_dir"

  # Wrapper so llama-server is on PATH with LD_LIBRARY_PATH pointing to its libs
  cat > "$LLAMA_BIN_DIR/llama-server" << 'WRAPPER'
#!/usr/bin/env bash
LLAMA_DIR="$HOME/.local/lib/llama.cpp"
exec env LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$LLAMA_DIR/llama-server" "$@"
WRAPPER
  chmod +x "$LLAMA_BIN_DIR/llama-server"

  if ! echo "$PATH" | grep -qF "$LLAMA_BIN_DIR"; then
    echo "    NOTE: Add to your shell profile:"
    echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
    export PATH="$LLAMA_BIN_DIR:$PATH"
  fi
fi
echo "    llama-server ready."

# ── Validate base install ─────────────────────────────────────────────────────
echo "==> Running smoke tests..."
uv run pytest tests/ -q --tb=short
echo "    All tests passed."

# ── GPU providers ─────────────────────────────────────────────────────────────
if [ $NVIDIA -eq 1 ]; then
  echo "==> Setting up Nvidia GPU support..."
  if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1
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
  uv pip install vllm
  echo "    NVIDIA=true" > .gpu_provider
  echo "    Nvidia setup complete."
fi

if [ $ROCM -eq 1 ]; then
  echo "==> Setting up AMD ROCm GPU support..."
  if ! command -v rocm-smi &>/dev/null; then
    echo "ERROR: rocm-smi not found. Install ROCm drivers first."
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1
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
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 1 || exit 1
  fi
  echo "    Apple Silicon detected."
  echo "    NOTE: Apple Metal monitoring backend not yet implemented."
  echo "    APPLE=true" > .gpu_provider
  echo "    Apple Silicon setup complete (monitoring stub)."
fi

echo ""
echo "Setup complete. To generate puzzles:"
echo "  uv run sudoku-gen config.yaml"
echo ""
echo "To run the benchmark:"
echo "  uv run sudoku-bench config.yaml"
echo ""
echo "Flags: --nvidia  (vLLM + nvidia-smi monitoring)"
echo "       --rocm    (AMD ROCm, monitoring stub)"
echo "       --apple   (Apple Silicon, monitoring stub)"

# Restore only the options we changed (avoids eval, which pollutes history)
[ "$_setup_e" = "off" ] && set +e
[ "$_setup_u" = "off" ] && set +u
[ "$_setup_p" = "off" ] && set +o pipefail
unset _setup_e _setup_u _setup_p
