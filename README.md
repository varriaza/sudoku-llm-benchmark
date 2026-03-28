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

Key fields:

```yaml
model:
  api_base: "http://localhost:11434/v1"  # your model server
  name: "llama3:70b-q4_K_M"             # omit to auto-detect
  # context_window: 32768               # set if backend doesn't report it

puzzles:
  - box_rows: 3
    box_cols: 3
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 5
```

`diffs` follows the `py-sudoku` convention: `0.4` means 40% of cells are empty (higher = harder).

## Usage

**Step 1 — Generate the puzzle bank** (once, or when you change the puzzle config):

```bash
uv run sudoku-gen configs/my_model.yaml
```

**Step 2 — Run the benchmark** (requires a model server running at `api_base`):

```bash
uv run sudoku-bench configs/my_model.yaml
```

Results are appended to the CSV file configured in `benchmark.results_file` (default: `results/benchmark.csv`).

## Running tests

```bash
uv run pytest
```

## Output

One CSV row per puzzle run. Key columns:

| Column | Description |
|--------|-------------|
| `model_name` | Auto-detected model identifier |
| `board_size` | e.g. `9x9` |
| `difficulty` | Difficulty float (0–1) |
| `solved` | Whether the model solved the puzzle |
| `best_pct_correct` | Highest % of correct cells during the run |
| `total_tokens` | Total tokens used |
| `context_pct_used` | Context window % used at end |
| `total_turns` | Number of feedback rounds |
| `max_vram_mb` | Peak VRAM usage |
| `spilled_to_ram` | Whether the model exceeded VRAM |

## Supported backends

Any server exposing an OpenAI-compatible `/v1/chat/completions` endpoint:

- [Ollama](https://ollama.com)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (server mode)

Model name, parameter count, quantization, and context window are auto-detected where the backend supports it. Set `model.context_window` in your config if your backend doesn't report it.
