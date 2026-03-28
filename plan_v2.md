# LLM Sudoku Benchmark — Plan v2

## Goal

Benchmark local LLMs on sudoku puzzles to compare performance across model sizes, quantization levels, and hardware constraints. Answer questions like: "Is a Q8 27B model better than a Q4 70B model on 16GB VRAM?"

## Architecture

Three components:

### 1. Puzzle Generator (one-time setup)

- Uses `py-sudoku` to pre-generate all puzzles and solutions
- Run once — every model gets the exact same puzzles
- Puzzle bank is a static artifact (checked into repo or distributed alongside)
- If config changes add new sizes/difficulties, re-run appends without touching existing puzzles
- Stores both the initial board (with givens) and the solution

### 2. Test Runner (per model)

Manages the conversational loop with the LLM over an OpenAI-compatible chat completions API:

1. Sends system prompt + initial grid to LLM
2. LLM responds with reasoning + updated grid
3. Harness parses the submitted grid
4. Validates against sudoku rules (not the solution) and sends feedback
5. If 100% correct and fully filled → puzzle complete, record metrics
6. If not → send feedback, loop to step 2
7. If next turn won't fit in context window → stop, record final state

### 3. Metrics Collector

- Polls `nvidia-smi` at configurable intervals during each puzzle run
- Captures token counts from API responses (including thinking tokens)
- Monitors system RAM for spillover detection
- Writes one CSV row per puzzle run

## Puzzle Format

### Grid Representation

Padded, aligned text grid. Givens marked with `*`:

```
 5  3* .  | .  7* .  | .  .  .
 6  .  .  | 1* 9  5  | .  .  .
 .  9* 8  | .  .  .  | .  6  .
---------- +---------- +----------
 8  .  .  | .  6* .  | .  .  3*
 4* .  .  | 8* .  3  | .  .  1*
 7  .  .  | .  2  .  | .  .  6
---------- +---------- +----------
 .  6  .  | .  .  .  | 2  8* .
 .  .  .  | 4  1* 9* | .  .  5*
 .  .  .  | .  8  .  | .  7* 9*
```

- Cell width adjusts for board size (2 chars for 1-9, 3 chars for 10-16)
- `.` represents empty cells
- `*` marks pre-filled givens that must not be changed

### Board Sizes

| Board | Box Size | Number Range |
|-------|----------|-------------|
| 4x4   | 2x2     | 1-4         |
| 6x6   | 2x3     | 1-6         |
| 9x9   | 3x3     | 1-9         |
| 12x12 | 3x4     | 1-12        |
| 16x16 | 4x4     | 1-16        |

## Configuration

YAML config file:

```yaml
model:
  api_base: "http://localhost:11434"
  name: "llama3:70b-q4_K_M"  # optional, auto-detected if omitted

puzzles:
  - box_rows: 2
    box_cols: 2
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 3
  - box_rows: 2
    box_cols: 3
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 3
  - box_rows: 3
    box_cols: 3
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 5
  - box_rows: 3
    box_cols: 4
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 3
  - box_rows: 4
    box_cols: 4
    diffs: [0.25, 0.5, 0.75]
    tests_per_diff: 3
```

- `diffs` values follow `py-sudoku` convention: 0.4 means 40% of cells are empty
- Higher difficulty = harder puzzle

## Model Auto-Detection

At startup, the harness:

1. Probes the server to detect backend type (Ollama vs vLLM vs other)
2. Queries model info endpoint (e.g. Ollama `/api/show`, vLLM `/v1/models`)
3. Extracts: model name, parameter count, quantization level, context window size
4. Falls back to config overrides for anything missing
5. Warns if critical fields couldn't be determined

## Conversation Design

### System Prompt

Tells the LLM:
- You are solving a sudoku puzzle
- Board dimensions and valid number range (e.g. "This is a 9x9 board with 3x3 boxes, valid numbers are 1-9")
- Rules: each row, column, and box must contain each number exactly once
- Cells marked with `*` are pre-filled and must not be changed
- Submit your current grid in the same format when you want to check progress
- Feedback will only report rule violations (duplicates, modified givens, out-of-range values)
- A fully correct board with no empty cells will end the challenge
- Do not write code — reason through the puzzle logically

### Feedback Rules

The validator only reports what a human could see by checking the rules:
- **Row/column/box duplicates** — "Row 5 has duplicate 3s"
- **Modified givens** — "R1C2 was a given (3), you changed it — reverted"
- **Out-of-range values** — "R4C7 has value 12, valid range is 1-9"
- **Completion status** — "68/81 cells filled"

No hints about correct values. The LLM must figure those out itself.

### Malformed Submissions

If the LLM submits a grid that can't be parsed (wrong dimensions, non-numeric values, missing rows, garbled formatting):
- Do not count it as a scored turn
- Respond with: "Your board couldn't be parsed. Please resubmit in this format:" followed by an example grid matching the current board size
- Track total malformed submissions per puzzle in metrics

### Termination Conditions

1. **Success** — LLM submits a fully correct, complete board
2. **Context exhaustion** — next turn won't fit in the model's context window

No early stopping across puzzles — every model runs every puzzle.

## Metrics & Output

Single CSV file, one row per puzzle run, appended across benchmark runs.

| Column | Description |
|--------|-------------|
| model_name | Auto-detected model identifier |
| model_params | Parameter count |
| model_quant | Quantization level |
| gpu_name | GPU model |
| gpu_max_vram_mb | GPU's total VRAM capacity |
| board_size | e.g. "9x9" |
| difficulty | The difficulty float |
| puzzle_id | Unique puzzle identifier |
| solved | Boolean |
| best_pct_correct | Highest completion % during the run |
| final_pct_correct | Completion % at end |
| best_num_errors | Fewest rule violations during the run |
| final_num_errors | Rule violations at end |
| total_tokens | Prompt + completion + thinking tokens |
| context_tokens_used | Total context tokens at end |
| context_pct_used | Context window % used at end |
| total_turns | Number of feedback rounds |
| total_seconds | Wall clock time |
| avg_vram_mb | Average VRAM usage |
| max_vram_mb | Peak VRAM usage |
| spilled_to_ram | Boolean — model exceeded VRAM |
| avg_sys_ram_mb | Average system RAM used |
| max_sys_ram_mb | Peak system RAM used |
| total_ram_mb | Peak VRAM + peak system RAM combined |
| malformed_submissions | Count of unparseable grid submissions |

## GPU Monitoring

- Nvidia GPU support via `nvidia-smi` polling
- Architecture allows adding AMD (rocm-smi) and Apple Silicon (Metal) backends later
- Polls at configurable interval during inference
- Tracks VRAM usage, system RAM usage, and detects VRAM spillover

## API Interface

Targets the OpenAI-compatible `/v1/chat/completions` endpoint. This is exposed by:
- Ollama
- vLLM
- llama.cpp server
- Most other local serving tools

The benchmark is agnostic to the serving backend — swap freely without changes.

## Development Approach

**Test-driven development.** Unit tests cover all core logic before implementation:

- **Grid parser** — parse LLM text output into board state, handle malformed input
- **Grid formatter** — render boards with padding, `*` markers, box dividers across all sizes
- **Validator** — duplicate detection, given-modification detection, out-of-range detection, completion check
- **Feedback generator** — produce correct feedback messages from validation results
- **Puzzle generator** — verify deterministic generation, correct difficulty levels, solution validity
- **Metrics collector** — verify CSV output format, correct aggregation of best/final stats

All core logic (parsing, validation, formatting, feedback) must be deterministically testable with no LLM or GPU dependency. The LLM and GPU monitoring layers are integration-level concerns tested separately.

## Setup Script

A `setup.sh` script that bootstraps the project environment:

### Base setup (`./setup.sh`)
- Checks for and installs `uv` if missing
- Installs the required Python version via `uv`
- Creates a virtual environment and installs Python dependencies
- Validates that everything is working

### GPU provider flags
- `./setup.sh --nvidia` — verifies `nvidia-smi` is available, installs any Python packages needed for Nvidia monitoring
- `./setup.sh --rocm` — verifies `rocm-smi` is available, installs AMD-specific dependencies
- `./setup.sh --apple` — verifies Apple Silicon Metal tools are available, installs macOS-specific dependencies

Each GPU flag:
1. Checks that the provider's CLI tools are installed and accessible
2. Installs any provider-specific Python packages
3. Runs a quick smoke test (e.g. can we query GPU info?)
4. Fails with a clear error message if requirements aren't met

Only one GPU provider is active per setup. The chosen provider is saved to a local config so the benchmark knows which monitoring backend to use at runtime.

## Tech Stack

- **Language:** Python
- **Testing:** pytest
- **Package management:** uv
- **Puzzle generation:** `py-sudoku`
- **LLM communication:** OpenAI-compatible API (via `openai` Python SDK or raw HTTP)
- **GPU monitoring:** `nvidia-smi` (with rocm-smi and Apple Metal planned)
- **Config:** YAML
- **Output:** CSV
