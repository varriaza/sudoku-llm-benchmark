# CLAUDE.md — sudoku-llm-benchmark

A tool that benchmarks LLMs on Sudoku puzzles and records 41 metrics per run to a CSV.

---

# VERY IMPORANT: After Making Changes — Update This File (CLAUDE.md)

If you add files, change the workflow, rename columns, or add new config options, update the relevant section above so the next agent starts with accurate context.

---

# Key Information

## Key Files

```
sudoku-llm-benchmark/
├── CLAUDE.md                      ← you are here; update after making changes
├── README.md                      ← full setup/usage docs
├── results/COLUMNS.md             ← reference for all 41 CSV output columns
├── configs/
│   ├── simple_sanity_check.yaml   ← fastest benchmark (1 puzzle, 2×2 board)
│   ├── full_sanity_check.yaml     ← medium benchmark (2×2, 3 diffs × 3 puzzles)
│   └── example.yaml               ← full benchmark template (5 board sizes)
├── puzzles/puzzles.json           ← pre-generated puzzle bank (do not edit)
├── results/                       ← CSV output lands here (gitignored except COLUMNS.md)
├── src/sudoku_bench/
│   ├── runner.py                  ← main benchmark loop (entry point: sudoku-bench)
│   ├── puzzle_bank.py             ← puzzle generation (entry point: sudoku-gen)
│   ├── config.py                  ← YAML config parsing (ModelConfig, BenchmarkConfig)
│   ├── metrics.py                 ← PuzzleMetrics dataclass + CSV append logic
│   ├── model_info.py              ← auto-detects model name/params/quant/backend
│   ├── parser.py                  ← parses LLM text output into a Board; strips <think> blocks, handles *-prefixed tokens and code fences
│   ├── validator.py               ← validates Sudoku rules, returns Violation list
│   ├── formatter.py               ← formats Board into text for LLM prompt
│   └── feedback.py                ← generates feedback messages from violations
├── tests/
│   ├── test_config.py             ← BenchmarkConfig defaults and YAML parsing
│   └── ...                        ← other tests mirror src/ structure
```

---

## Run a Sanity Check Benchmark

```bash
# Quickest possible run: 1 puzzle, 2×2 board
uv run sudoku-bench configs/simple_sanity_check.yaml

# Medium run: 2×2 board, 3 difficulties, 3 puzzles each
uv run sudoku-bench configs/full_sanity_check.yaml
```

Both configs include a `serve:` block that **automatically downloads, starts, and stops llama-server** — no manual server setup needed. The model is downloaded from HuggingFace on first run.

Results are written to `results/simple_sanity_benchmark.csv` (or whichever `results_file` is set in the config).

---

## Run pytest Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_validator.py

# Run with coverage
uv run pytest --cov=sudoku_bench
```

Tests live in `tests/` and mirror the `src/sudoku_bench/` structure. No conftest.py — fixtures are inline per file.

### Write a New Test

1. Create or edit a file in `tests/` (e.g. `tests/test_parser.py`)
2. Import from `sudoku_bench.*` directly (pythonpath is set to `src/` in pyproject.toml)
3. Use plain `pytest` functions — no special base class needed

```python
from sudoku_bench.parser import parse_board

def test_my_case():
    result = parse_board("1 2\n3 4", board_size=2)
    assert result is not None
```

---

## Find New Results

```bash
ls -lt results/*.csv   # most recent files first
```

Each CSV row is one puzzle run. Columns are documented in `results/COLUMNS.md`. Key columns:

| Column | Meaning |
|---|---|
| `model_name` | Model identifier |
| `board_size` | e.g. `4` = 4×4 board |
| `difficulty` | 0.0–1.0 (fraction of cells removed) |
| `solved` | `true`/`false` |
| `best_pct_correct` | Best board accuracy across all turns |
| `total_turns` | How many LLM turns were used |
| `total_tokens_used` | Total tokens consumed |
| `total_seconds` | Wall-clock run time |

---

## Create a New Config

Copy an existing config and edit:

```yaml
model:
  api_base: "http://localhost:8080/v1"
  name: "my-model-name"
  context_window: 32768

# Technically not required but you should include this unless told otherwise.
serve:
  # llama-server is installed by: bash setup.sh --llamacpp
  # Add --n-gpu-layers 99 to offload all layers to GPU
  command: [
    "llama-server",
    "--hf-repo", "unsloth/Qwen3.5-9B-GGUF",
    "--hf-file", "Qwen3.5-9B-Q4_K_M.gguf",
    "--port", "8080",
    "--ctx-size", "32768",
    "--metrics"
  ]
  startup_timeout: 120  # seconds to wait for server to be ready

puzzles:
  - box_rows: 2
    box_cols: 2
    diffs: [0.25]
    tests_per_diff: 1

benchmark:
  results_file: "results/my_model.csv"
  puzzle_bank_file: "puzzles/puzzles.json"
  context_buffer_tokens: 500
  max_turns_per_puzzle: 200
  save_llm_output: false   # set true to write full LLM I/O to results/llm_output_<timestamp>.txt
```

---


