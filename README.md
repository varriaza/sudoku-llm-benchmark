# sudoku-llm-benchmark

Benchmark local LLMs on sudoku puzzles to compare performance across model sizes, quantization levels, and hardware constraints. Answers questions like: "Is a Q8 27B model better than a Q4 70B model on 16GB VRAM?"

## How it works

1. **Puzzle bank** — pre-generated puzzles (4x4 through 16x16, configurable difficulties) stored as a static JSON file. Every model gets the exact same puzzles.
2. **Test runner** — holds a conversation with the LLM over an OpenAI-compatible API. The model submits grids, the harness validates them against sudoku rules, and sends back feedback (rule violations only — no hints). Repeats until solved or context window is exhausted.
3. **Metrics** — one CSV row per puzzle run, capturing accuracy, token usage, VRAM, and system RAM.

## Setup

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
./setup.sh
```

For GPU monitoring, pass a provider flag:

```bash
./setup.sh --nvidia    # Nvidia (nvidia-smi)
./setup.sh --rocm      # AMD (rocm-smi)
./setup.sh --apple     # Apple Silicon
```

## Configuration

Copy the example config and edit it:

```bash
cp configs/example.yaml configs/my_model.yaml
```

### Model

```yaml
model:
  api_base: "http://localhost:8000/v1"
  name: "meta-llama/Llama-3.1-8B-Instruct"  # omit to auto-detect
  # context_window: 32768                     # set if backend doesn't report it
```

For GGUF repos, use `repo_id:quant_type` to select a specific quantization. Many GGUF repos lack a `config.json`, so use `--hf-config-path` pointing to the base model (found in the repo's tags/README on HuggingFace):

```yaml
model:
  api_base: "http://localhost:8000/v1"
  name: "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
serve:
  command: ["vllm", "serve", "{model}", "--hf-config-path", "Qwen/Qwen3.5-9B"]
  startup_timeout: 120
```

`context_window` is required. It's auto-detected for Ollama. For vLLM and other backends that don't report it, set it manually.

### Auto-starting the server

Add a `serve:` block and `sudoku-bench` will start the server for you, wait until it's ready, and shut it down when the benchmark finishes (including on Ctrl-C or errors):

```yaml
serve:
  command: ["vllm", "serve", "{model}"]  # {model} is replaced with model.name
  startup_timeout: 120  # seconds to wait for server to be ready
```

Remove the `serve:` block if you prefer to manage the server yourself.

### Puzzles

```yaml
puzzles:
  - box_rows: 3
    box_cols: 3
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 5
```

`box_rows` × `box_cols` determines the board size (e.g. 3×3 boxes → 9×9 board). `diffs` follows the `py-sudoku` convention: `0.4` means 40% of cells are empty (higher = harder).

Supported board sizes: 4×4, 6×6, 9×9, 12×12, 16×16.

## Usage

**Step 1 — Generate the puzzle bank** (run once, or when you change the puzzle config):

```bash
uv run sudoku-gen configs/my_model.yaml
```

**Step 2 — Run the benchmark:**

```bash
uv run sudoku-bench configs/my_model.yaml
```

If you have a `serve:` block in your config, this is the only command you need — the server starts and stops automatically. Otherwise, make sure your model server is running at `api_base` before this step.

Results are appended to the CSV file set in `benchmark.results_file` (default: `results/benchmark.csv`).

## Running tests

```bash
uv run pytest
```

## Output

One CSV row per puzzle run. Key columns:

| Column | Description |
|--------|-------------|
| `model_name` | Auto-detected model identifier |
| `model_params` | Parameter count (e.g. `70B`) |
| `model_quant` | Quantization level (e.g. `Q4_K_M`) |
| `board_size` | e.g. `9x9` |
| `difficulty` | Difficulty float (0–1) |
| `solved` | Whether the model solved the puzzle |
| `best_pct_correct` | Highest % of correct cells during the run |
| `total_tokens` | Total tokens used |
| `context_pct_used` | Context window % used at end |
| `total_turns` | Number of feedback rounds |
| `max_vram_mb` | Peak VRAM usage (Nvidia only) |
| `spilled_to_ram` | Whether the model exceeded VRAM |
| `total_ram_mb` | Peak VRAM + peak system RAM combined |

## Supported backends

Any server exposing an OpenAI-compatible `/v1/chat/completions` endpoint:

| Backend | Auto-detect | GPU metrics |
|---------|-------------|-------------|
| [Ollama](https://ollama.com) | Model name, context window | nvidia-smi |
| [vLLM](https://github.com/vllm-project/vllm) | Model name | `/metrics` endpoint (KV-cache usage, spill detection) |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) (server mode) | Model name | nvidia-smi |

For vLLM, GPU metrics come from its Prometheus `/metrics` endpoint rather than nvidia-smi, giving more accurate spill detection via KV-cache CPU usage.
