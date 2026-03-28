# Sudoku LLM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark that runs local LLMs through sudoku puzzles via a conversational loop and records accuracy + hardware metrics to a CSV, enabling performance-per-VRAM comparisons across model sizes and quantizations.

**Architecture:** A static puzzle bank (generated once with py-sudoku) is shared across all model runs. A test runner manages a multi-turn chat loop over OpenAI-compatible `/v1/chat/completions`, parsing each LLM response for a board submission, validating it against sudoku rules, and sending feedback. A GPU monitor polls `nvidia-smi` in the background. All results append to a single CSV.

**Tech Stack:** Python 3.11+, uv, pytest, py-sudoku, openai SDK, PyYAML, psutil

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/sudoku_bench/board.py` | `Board` dataclass: cells, givens, box dims |
| `src/sudoku_bench/formatter.py` | `Board` → padded text grid with `*` markers |
| `src/sudoku_bench/parser.py` | LLM response text → `Board` or `None` |
| `src/sudoku_bench/validator.py` | Detect duplicates, modified givens, out-of-range |
| `src/sudoku_bench/feedback.py` | Violation list → feedback string |
| `src/sudoku_bench/puzzle_bank.py` | Generate/load/save puzzle bank JSON |
| `src/sudoku_bench/config.py` | Load + validate YAML config |
| `src/sudoku_bench/model_info.py` | Probe Ollama/vLLM for model metadata |
| `src/sudoku_bench/gpu_monitor.py` | Background nvidia-smi polling |
| `src/sudoku_bench/metrics.py` | Accumulate run stats → append CSV row |
| `src/sudoku_bench/runner.py` | Conversational loop + CLI entry point |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/sudoku_bench/__init__.py`
- Create: `config.example.yaml`
- Create: `puzzles/.gitkeep`
- Create: `results/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/sudoku_bench tests puzzles results docs/superpowers/plans docs/superpowers/specs
touch src/sudoku_bench/__init__.py
touch puzzles/.gitkeep results/.gitkeep
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sudoku-bench"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "py-sudoku>=1.0.3",
    "openai>=1.0.0",
    "pyyaml>=6.0",
    "psutil>=5.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]

[project.scripts]
sudoku-bench = "sudoku_bench.runner:main"
sudoku-gen = "sudoku_bench.puzzle_bank:main"

[tool.hatch.build.targets.wheel]
packages = ["src/sudoku_bench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `config.example.yaml`**

```yaml
model:
  api_base: "http://localhost:11434/v1"
  name: "llama3:70b-q4_K_M"   # optional — auto-detected if omitted

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

benchmark:
  gpu_poll_interval: 1.0          # seconds between nvidia-smi polls
  results_file: "results/benchmark.csv"
  puzzle_bank_file: "puzzles/puzzles.json"
  context_buffer_tokens: 500      # reserve this many tokens before context is "full"
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
results/
*.csv
.env
```

- [ ] **Step 5: Install dependencies with uv**

```bash
uv venv
uv pip install -e ".[dev]"
```

Expected: no errors, `pytest --collect-only` runs with 0 tests found.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config.example.yaml .gitignore src/ tests/ puzzles/.gitkeep results/.gitkeep
git commit -m "chore: project skeleton with pyproject.toml"
```

---

## Task 2: Board Dataclass

**Files:**
- Create: `src/sudoku_bench/board.py`
- Create: `tests/test_board.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_board.py`:

```python
import pytest
from sudoku_bench.board import Board


def test_board_size_9x9():
    cells = [[None] * 9 for _ in range(9)]
    board = Board(cells=cells, givens=frozenset(), box_rows=3, box_cols=3)
    assert board.size == 9


def test_board_size_4x4():
    cells = [[None] * 4 for _ in range(4)]
    board = Board(cells=cells, givens=frozenset(), box_rows=2, box_cols=2)
    assert board.size == 4


def test_board_size_6x6():
    cells = [[None] * 6 for _ in range(6)]
    board = Board(cells=cells, givens=frozenset(), box_rows=2, box_cols=3)
    assert board.size == 6


def test_board_size_12x12():
    cells = [[None] * 12 for _ in range(12)]
    board = Board(cells=cells, givens=frozenset(), box_rows=3, box_cols=4)
    assert board.size == 12


def test_board_size_16x16():
    cells = [[None] * 16 for _ in range(16)]
    board = Board(cells=cells, givens=frozenset(), box_rows=4, box_cols=4)
    assert board.size == 16


def test_board_is_given():
    cells = [[1, None], [None, 2]]
    board = Board(cells=cells, givens=frozenset({(0, 0), (1, 1)}), box_rows=2, box_cols=2)
    assert board.is_given(0, 0) is True
    assert board.is_given(0, 1) is False
    assert board.is_given(1, 1) is True


def test_board_cells_filled_count():
    cells = [[1, None, 3, None],
             [None, 2, None, 4],
             [3, None, None, 2],
             [None, 4, 1, None]]
    board = Board(cells=cells, givens=frozenset(), box_rows=2, box_cols=2)
    assert board.cells_filled == 6
    assert board.total_cells == 16


def test_board_copy_with_cells():
    cells = [[1, None], [None, 2]]
    board = Board(cells=cells, givens=frozenset({(0, 0)}), box_rows=2, box_cols=2)
    new_cells = [[1, 3], [4, 2]]
    copy = board.copy_with_cells(new_cells)
    assert copy.cells == new_cells
    assert copy.givens == board.givens
    assert copy.box_rows == board.box_rows
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_board.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `sudoku_bench.board` doesn't exist yet.

- [ ] **Step 3: Implement `src/sudoku_bench/board.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Board:
    cells: list[list[Optional[int]]]
    givens: frozenset[tuple[int, int]]
    box_rows: int
    box_cols: int

    @property
    def size(self) -> int:
        return self.box_rows * self.box_cols

    def is_given(self, row: int, col: int) -> bool:
        return (row, col) in self.givens

    @property
    def cells_filled(self) -> int:
        return sum(1 for row in self.cells for cell in row if cell is not None)

    @property
    def total_cells(self) -> int:
        return self.size * self.size

    def copy_with_cells(self, new_cells: list[list[Optional[int]]]) -> Board:
        return Board(
            cells=[row[:] for row in new_cells],
            givens=self.givens,
            box_rows=self.box_rows,
            box_cols=self.box_cols,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_board.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/board.py tests/test_board.py
git commit -m "feat: Board dataclass with size, givens, fill count"
```

---

## Task 3: Grid Formatter

**Files:**
- Create: `src/sudoku_bench/formatter.py`
- Create: `tests/test_formatter.py`

The format uses fixed-width cells. For boards with size ≤ 9, each cell is 3 characters (` N*`, ` N `, ` . `). For boards with size 10-16, each cell is 4 characters (` NN*`, ` NN `, `  . `). Columns of boxes are separated by ` | `. Rows of boxes are separated by a horizontal rule of dashes and `+` signs matching the row width.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_formatter.py`:

```python
import pytest
from sudoku_bench.board import Board
from sudoku_bench.formatter import format_board, cell_width


def make_board(cells, givens_set, box_rows, box_cols):
    return Board(
        cells=cells,
        givens=frozenset(givens_set),
        box_rows=box_rows,
        box_cols=box_cols,
    )


def test_cell_width_small_board():
    assert cell_width(4) == 3   # 4x4 board, values 1-4
    assert cell_width(9) == 3   # 9x9 board, values 1-9


def test_cell_width_large_board():
    assert cell_width(12) == 4  # 12x12 board, values 1-12
    assert cell_width(16) == 4  # 16x16 board, values 1-16


def test_format_4x4_empty_board():
    cells = [[None] * 4 for _ in range(4)]
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    result = format_board(board)
    lines = result.strip().split("\n")
    # 4x4 with 2x2 boxes: 4 data rows + 1 separator row = 5 lines
    assert len(lines) == 5
    # separator appears after row index 1 (0-indexed)
    sep_line = lines[2]
    assert "+" in sep_line
    assert all(c in "-+ " for c in sep_line)


def test_format_4x4_given_marked_with_star():
    cells = [[1, 2, 3, 4],
             [3, 4, 1, 2],
             [2, 1, 4, 3],
             [4, 3, 2, 1]]
    board = make_board(cells, {(0, 0), (1, 2)}, box_rows=2, box_cols=2)
    result = format_board(board)
    lines = result.strip().split("\n")
    # Row 0: cell (0,0) is given "1" → should contain "1*"
    assert "1*" in lines[0]
    # Row 1: cell (1,2) is given "1" → should contain "1*"
    # row 1 is lines[1], but cell (1,2) is in column 2 (second box)
    assert "1*" in lines[1]


def test_format_4x4_non_given_no_star():
    cells = [[1, 2, 3, 4],
             [3, 4, 1, 2],
             [2, 1, 4, 3],
             [4, 3, 2, 1]]
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    result = format_board(board)
    assert "*" not in result


def test_format_9x9_row_count():
    cells = [[None] * 9 for _ in range(9)]
    board = make_board(cells, set(), box_rows=3, box_cols=3)
    result = format_board(board)
    lines = result.strip().split("\n")
    # 9 data rows + 2 separator rows = 11 lines
    assert len(lines) == 11


def test_format_9x9_empty_cell_is_dot():
    cells = [[None] * 9 for _ in range(9)]
    board = make_board(cells, set(), box_rows=3, box_cols=3)
    result = format_board(board)
    for line in result.strip().split("\n"):
        if "+" not in line:
            assert "." in line


def test_format_16x16_row_count():
    cells = [[None] * 16 for _ in range(16)]
    board = make_board(cells, set(), box_rows=4, box_cols=4)
    result = format_board(board)
    lines = result.strip().split("\n")
    # 16 data rows + 3 separator rows = 19 lines
    assert len(lines) == 19


def test_format_16x16_two_digit_value():
    cells = [[None] * 16 for _ in range(16)]
    cells[0][0] = 12
    board = make_board(cells, {(0, 0)}, box_rows=4, box_cols=4)
    result = format_board(board)
    assert "12*" in result


def test_format_roundtrip_givens_preserved():
    """Given cells show * and non-given cells don't."""
    cells = [[5, None, None], [None, 3, None], [None, None, 7]]
    # Use a 3x3 with 1x3 boxes — not standard sudoku, just testing format
    # Actually let's use a real board: 9x9 partial
    cells9 = [[None] * 9 for _ in range(9)]
    cells9[0][0] = 5
    cells9[4][4] = 3
    board = make_board(cells9, {(0, 0)}, box_rows=3, box_cols=3)
    result = format_board(board)
    # (0,0) is given → "5*" present
    assert "5*" in result
    # (4,4) is not given → "3 " present (followed by space or |), "3*" NOT present
    assert "3*" not in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_formatter.py -v
```

Expected: `ImportError` for `sudoku_bench.formatter`.

- [ ] **Step 3: Implement `src/sudoku_bench/formatter.py`**

```python
from __future__ import annotations
from sudoku_bench.board import Board


def cell_width(board_size: int) -> int:
    """Return the character width of a single cell (number + star/space)."""
    return 3 if board_size <= 9 else 4


def _format_cell(value: int | None, is_given: bool, width: int) -> str:
    """Format a single cell to exactly `width` characters."""
    num_width = width - 1  # space for the star/space suffix
    if value is None:
        return " " * (num_width - 1) + ". "
    suffix = "*" if is_given else " "
    return str(value).rjust(num_width) + suffix


def format_board(board: Board) -> str:
    size = board.size
    cw = cell_width(size)
    box_rows = board.box_rows
    box_cols = board.box_cols
    # boxes_across: how many box-columns span the board horizontally
    # Each box is box_cols cells wide → boxes_across = size / box_cols
    boxes_across = size // box_cols

    # Width of one box section: box_cols cells (each `cw` chars wide)
    box_section_width = box_cols * cw

    # Separator line: dashes for each box section, joined with " + "
    sep = " + ".join(["-" * box_section_width] * boxes_across)

    lines: list[str] = []
    for r in range(size):
        # Insert horizontal separator before each new box row (except the first)
        if r > 0 and r % box_rows == 0:
            lines.append(sep)

        # Build one data row: cells grouped into box sections, separated by " | "
        box_sections: list[str] = []
        for bx in range(boxes_across):
            col_start = bx * box_cols
            col_end = col_start + box_cols
            section = "".join(
                _format_cell(board.cells[r][c], board.is_given(r, c), cw)
                for c in range(col_start, col_end)
            )
            box_sections.append(section)
        lines.append(" | ".join(box_sections))

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_formatter.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/formatter.py tests/test_formatter.py
git commit -m "feat: grid formatter with * givens, box dividers, scaled cell width"
```

---

## Task 4: Grid Parser

**Files:**
- Create: `src/sudoku_bench/parser.py`
- Create: `tests/test_parser.py`

The parser extracts a board from free-form LLM text. It looks for a grid block (lines containing cells separated by `|` and/or whitespace), strips `*` markers, and reconstructs a `Board`. It returns `None` if parsing fails (wrong number of rows/cols, non-numeric values, etc.).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parser.py`:

```python
import pytest
from sudoku_bench.board import Board
from sudoku_bench.parser import parse_board


def test_parse_simple_4x4():
    text = """
 1  2  |  3  4
 3  4  |  1  2
-------+-------
 2  1  |  4  3
 4  3  |  2  1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells[0] == [1, 2, 3, 4]
    assert board.cells[3] == [4, 3, 2, 1]


def test_parse_strips_star_markers():
    text = """
 1* 2  |  3* 4
 3  4* |  1  2
-------+-------
 2  1  |  4  3
 4  3  |  2  1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells[0][0] == 1
    assert board.cells[0][2] == 3


def test_parse_preserves_givens_from_stars():
    text = """
 1* 2  |  3* 4
 3  4* |  1  2
-------+-------
 2  1  |  4  3
 4  3  |  2  1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert (0, 0) in board.givens
    assert (0, 2) in board.givens
    assert (1, 1) in board.givens
    assert (0, 1) not in board.givens


def test_parse_empty_cells_as_none():
    text = """
 1* .  |  .  4
 .  4* |  1  .
-------+-------
 2  .  |  .  3
 .  3  |  2  .
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells[0][1] is None
    assert board.cells[0][2] is None
    assert board.cells[1][0] is None


def test_parse_embedded_in_prose():
    text = """
I've been working through the puzzle and here's my current board:

 1* 2  |  3* 4
 3  4* |  1  2
-------+-------
 2  1  |  4  3
 4  3  |  2  1

I think row 3 needs more work.
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells[0] == [1, 2, 3, 4]


def test_parse_returns_none_wrong_row_count():
    text = """
 1  2  |  3  4
 3  4  |  1  2
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is None


def test_parse_returns_none_wrong_col_count():
    text = """
 1  2  3  |  4
 3  4  1  |  2
-----------+---
 2  1  4  |  3
 4  3  2  |  1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is None


def test_parse_returns_none_non_numeric():
    text = """
 1  X  |  3  4
 3  4  |  1  2
-------+-------
 2  1  |  4  3
 4  3  |  2  1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is None


def test_parse_9x9():
    row = " 1  2  3  |  4  5  6  |  7  8  9"
    sep = "-----------+-----------+-----------"
    grid_lines = [row] * 3 + [sep] + [row] * 3 + [sep] + [row] * 3
    text = "\n".join(grid_lines)
    board = parse_board(text, box_rows=3, box_cols=3)
    assert board is not None
    assert len(board.cells) == 9
    assert len(board.cells[0]) == 9


def test_parse_16x16_two_digit_values():
    # Build a minimal 16x16 text where all cells are "16"
    row_cells = "  16" * 4
    row = f"{row_cells} | {row_cells} | {row_cells} | {row_cells}"
    sep = "-" * len(row.split("|")[0]) + "+" + "-" * 10  # rough separator
    lines = []
    for block in range(4):
        if block > 0:
            lines.append(sep)
        for _ in range(4):
            lines.append(row)
    text = "\n".join(lines)
    board = parse_board(text, box_rows=4, box_cols=4)
    assert board is not None
    assert board.cells[0][0] == 16
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parser.py -v
```

Expected: `ImportError` for `sudoku_bench.parser`.

- [ ] **Step 3: Implement `src/sudoku_bench/parser.py`**

```python
from __future__ import annotations
import re
from typing import Optional
from sudoku_bench.board import Board


def _is_separator_line(line: str) -> bool:
    """True if the line is a box-row separator (dashes and + signs only)."""
    stripped = line.strip()
    return bool(stripped) and all(c in "-+" for c in stripped.replace(" ", ""))


def _parse_data_line(line: str) -> list[tuple[Optional[int], bool]] | None:
    """
    Parse one data row into a list of (value, is_given) pairs.
    Returns None if any token is non-numeric (other than '.' for empty).
    """
    # Strip box-column separators
    cleaned = line.replace("|", " ")
    tokens = cleaned.split()
    result: list[tuple[Optional[int], bool]] = []
    for token in tokens:
        is_given = token.endswith("*")
        raw = token.rstrip("*")
        if raw == ".":
            result.append((None, is_given))
        elif raw.isdigit():
            result.append((int(raw), is_given))
        else:
            return None
    return result if result else None


def parse_board(text: str, box_rows: int, box_cols: int) -> Optional[Board]:
    """
    Extract a Board from free-form LLM text.
    Returns None if no valid board of the expected size can be found.
    """
    size = box_rows * box_cols
    lines = text.splitlines()

    # Collect data lines (skip separator lines and blank lines)
    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_separator_line(stripped):
            continue
        parsed = _parse_data_line(stripped)
        if parsed is not None and len(parsed) == size:
            data_lines.append(stripped)
        elif parsed is not None and len(parsed) != size:
            # Wrong column count — might be prose that happens to have numbers;
            # collect anyway and fail at validation
            data_lines.append(stripped)

    # Find the first contiguous window of `size` valid rows
    window: list[list[tuple[Optional[int], bool]]] = []
    for line in data_lines:
        parsed = _parse_data_line(line)
        if parsed is None:
            window = []
            continue
        if len(parsed) != size:
            window = []
            continue
        window.append(parsed)
        if len(window) == size:
            break
    else:
        if len(window) != size:
            return None

    cells: list[list[Optional[int]]] = []
    givens: set[tuple[int, int]] = set()

    for r, row_data in enumerate(window):
        row: list[Optional[int]] = []
        for c, (val, is_given) in enumerate(row_data):
            row.append(val)
            if is_given:
                givens.add((r, c))
        cells.append(row)

    return Board(
        cells=cells,
        givens=frozenset(givens),
        box_rows=box_rows,
        box_cols=box_cols,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/parser.py tests/test_parser.py
git commit -m "feat: LLM response parser — extracts board, handles * givens and prose"
```

---

## Task 5: Validator

**Files:**
- Create: `src/sudoku_bench/validator.py`
- Create: `tests/test_validator.py`

The validator checks a submitted board against sudoku rules using the *original* board's givens. It never reveals correct values — only rule violations.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validator.py`:

```python
import pytest
from sudoku_bench.board import Board
from sudoku_bench.validator import validate, Violation, ViolationType


def make_board(cells, givens_set, box_rows=3, box_cols=3):
    return Board(
        cells=[row[:] for row in cells],
        givens=frozenset(givens_set),
        box_rows=box_rows,
        box_cols=box_cols,
    )


# --- Helpers ---

def solved_4x4():
    return [[1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]]


# --- Row duplicate tests ---

def test_no_violations_complete_4x4():
    cells = solved_4x4()
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    assert violations == []


def test_row_duplicate_detected():
    cells = solved_4x4()
    cells[0][2] = 1  # duplicate 1 in row 0
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    row_dups = [v for v in violations if v.type == ViolationType.ROW_DUPLICATE]
    assert len(row_dups) == 1
    assert row_dups[0].row == 0
    assert row_dups[0].value == 1


def test_col_duplicate_detected():
    cells = solved_4x4()
    cells[2][0] = 1  # duplicate 1 in col 0
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    col_dups = [v for v in violations if v.type == ViolationType.COL_DUPLICATE]
    assert len(col_dups) == 1
    assert col_dups[0].col == 0
    assert col_dups[0].value == 1


def test_box_duplicate_detected():
    cells = solved_4x4()
    cells[1][0] = 1  # duplicate 1 in top-left 2x2 box
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    box_dups = [v for v in violations if v.type == ViolationType.BOX_DUPLICATE]
    assert len(box_dups) == 1
    assert box_dups[0].value == 1


def test_modified_given_detected():
    cells = solved_4x4()
    original_cells = solved_4x4()
    cells[0][0] = 9  # change given at (0,0)
    board = make_board(cells, {(0, 0)}, box_rows=2, box_cols=2)
    original = make_board(original_cells, {(0, 0)}, box_rows=2, box_cols=2)
    violations = validate(board, original=original)
    modified = [v for v in violations if v.type == ViolationType.MODIFIED_GIVEN]
    assert len(modified) == 1
    assert modified[0].row == 0
    assert modified[0].col == 0
    assert modified[0].expected == 1
    assert modified[0].got == 9


def test_out_of_range_detected():
    cells = solved_4x4()
    cells[2][2] = 9  # 9 is out of range for 4x4
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    oor = [v for v in violations if v.type == ViolationType.OUT_OF_RANGE]
    assert len(oor) == 1
    assert oor[0].row == 2
    assert oor[0].col == 2
    assert oor[0].value == 9


def test_empty_cells_not_a_violation():
    cells = [[None, None, None, None]] * 4
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    assert violations == []


def test_completion_pct_complete_board():
    cells = solved_4x4()
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    assert violations == []
    # fully filled and valid → board is complete
    assert board.cells_filled == 16
    assert board.total_cells == 16


def test_multiple_violations_all_reported():
    cells = solved_4x4()
    cells[0][2] = 1   # row dup
    cells[2][0] = 1   # col dup
    board = make_board(cells, set(), box_rows=2, box_cols=2)
    violations = validate(board)
    assert len(violations) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_validator.py -v
```

Expected: `ImportError` for `sudoku_bench.validator`.

- [ ] **Step 3: Implement `src/sudoku_bench/validator.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from sudoku_bench.board import Board


class ViolationType(Enum):
    ROW_DUPLICATE = "row_duplicate"
    COL_DUPLICATE = "col_duplicate"
    BOX_DUPLICATE = "box_duplicate"
    MODIFIED_GIVEN = "modified_given"
    OUT_OF_RANGE = "out_of_range"


@dataclass
class Violation:
    type: ViolationType
    row: Optional[int] = None
    col: Optional[int] = None
    value: Optional[int] = None
    expected: Optional[int] = None
    got: Optional[int] = None
    positions: list[tuple[int, int]] = field(default_factory=list)


def validate(board: Board, original: Optional[Board] = None) -> list[Violation]:
    """
    Validate a submitted board against sudoku rules.
    If `original` is provided, also check for modified givens.
    Empty cells (None) are not violations.
    """
    size = board.size
    violations: list[Violation] = []

    # Out-of-range check
    for r in range(size):
        for c in range(size):
            val = board.cells[r][c]
            if val is not None and (val < 1 or val > size):
                violations.append(Violation(
                    type=ViolationType.OUT_OF_RANGE,
                    row=r, col=c, value=val,
                ))

    # Modified given check
    if original is not None:
        for (r, c) in original.givens:
            original_val = original.cells[r][c]
            submitted_val = board.cells[r][c]
            if submitted_val != original_val:
                violations.append(Violation(
                    type=ViolationType.MODIFIED_GIVEN,
                    row=r, col=c,
                    expected=original_val,
                    got=submitted_val,
                ))

    # Row duplicate check
    for r in range(size):
        seen: dict[int, list[int]] = {}
        for c in range(size):
            val = board.cells[r][c]
            if val is not None:
                seen.setdefault(val, []).append(c)
        for val, cols in seen.items():
            if len(cols) > 1:
                violations.append(Violation(
                    type=ViolationType.ROW_DUPLICATE,
                    row=r, value=val,
                    positions=[(r, c) for c in cols],
                ))

    # Column duplicate check
    for c in range(size):
        seen: dict[int, list[int]] = {}
        for r in range(size):
            val = board.cells[r][c]
            if val is not None:
                seen.setdefault(val, []).append(r)
        for val, rows in seen.items():
            if len(rows) > 1:
                violations.append(Violation(
                    type=ViolationType.COL_DUPLICATE,
                    col=c, value=val,
                    positions=[(r, c) for r in rows],
                ))

    # Box duplicate check
    # boxes_down: how many box-rows span the board vertically = size / box_rows
    # boxes_across: how many box-cols span the board horizontally = size / box_cols
    boxes_down = board.size // board.box_rows
    boxes_across = board.size // board.box_cols
    for br in range(boxes_down):
        for bc in range(boxes_across):
            seen: dict[int, list[tuple[int, int]]] = {}
            for r in range(br * board.box_rows, (br + 1) * board.box_rows):
                for c in range(bc * board.box_cols, (bc + 1) * board.box_cols):
                    val = board.cells[r][c]
                    if val is not None:
                        seen.setdefault(val, []).append((r, c))
            for val, positions in seen.items():
                if len(positions) > 1:
                    violations.append(Violation(
                        type=ViolationType.BOX_DUPLICATE,
                        value=val,
                        positions=positions,
                    ))

    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validator.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/validator.py tests/test_validator.py
git commit -m "feat: validator — row/col/box duplicates, modified givens, out-of-range"
```

---

## Task 6: Feedback Generator

**Files:**
- Create: `src/sudoku_bench/feedback.py`
- Create: `tests/test_feedback.py`

The feedback generator converts a list of violations + fill count into a plain-English string for the LLM. It never reveals correct values.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feedback.py`:

```python
import pytest
from sudoku_bench.board import Board
from sudoku_bench.validator import Violation, ViolationType
from sudoku_bench.feedback import generate_feedback


def test_feedback_no_violations_incomplete():
    violations = []
    msg = generate_feedback(violations, cells_filled=45, total_cells=81)
    assert "45/81" in msg
    assert "violation" not in msg.lower() or "no" in msg.lower()


def test_feedback_complete_no_violations():
    violations = []
    msg = generate_feedback(violations, cells_filled=81, total_cells=81)
    assert "correct" in msg.lower() or "complete" in msg.lower() or "solved" in msg.lower()


def test_feedback_row_duplicate():
    violations = [Violation(
        type=ViolationType.ROW_DUPLICATE,
        row=4, value=3,
        positions=[(4, 1), (4, 6)],
    )]
    msg = generate_feedback(violations, cells_filled=60, total_cells=81)
    assert "row 5" in msg.lower()  # 1-indexed in output
    assert "3" in msg


def test_feedback_col_duplicate():
    violations = [Violation(
        type=ViolationType.COL_DUPLICATE,
        col=2, value=7,
        positions=[(0, 2), (5, 2)],
    )]
    msg = generate_feedback(violations, cells_filled=60, total_cells=81)
    assert "column 3" in msg.lower()  # 1-indexed
    assert "7" in msg


def test_feedback_modified_given():
    violations = [Violation(
        type=ViolationType.MODIFIED_GIVEN,
        row=0, col=1,
        expected=3, got=5,
    )]
    msg = generate_feedback(violations, cells_filled=81, total_cells=81)
    assert "r1c2" in msg.lower()  # 1-indexed
    assert "given" in msg.lower()
    assert "3" in msg  # shows original value


def test_feedback_out_of_range():
    violations = [Violation(
        type=ViolationType.OUT_OF_RANGE,
        row=3, col=6, value=12,
    )]
    msg = generate_feedback(violations, cells_filled=50, total_cells=81, board_size=9)
    assert "r4c7" in msg.lower()  # 1-indexed
    assert "12" in msg
    assert "9" in msg  # shows valid range


def test_feedback_multiple_violations():
    violations = [
        Violation(type=ViolationType.ROW_DUPLICATE, row=0, value=1, positions=[(0,0),(0,3)]),
        Violation(type=ViolationType.COL_DUPLICATE, col=1, value=2, positions=[(1,1),(4,1)]),
    ]
    msg = generate_feedback(violations, cells_filled=40, total_cells=81)
    # Should mention both
    assert "row" in msg.lower()
    assert "column" in msg.lower()


def test_feedback_fill_count_always_present():
    violations = [Violation(type=ViolationType.ROW_DUPLICATE, row=0, value=1, positions=[])]
    msg = generate_feedback(violations, cells_filled=30, total_cells=81)
    assert "30/81" in msg
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_feedback.py -v
```

Expected: `ImportError` for `sudoku_bench.feedback`.

- [ ] **Step 3: Implement `src/sudoku_bench/feedback.py`**

```python
from __future__ import annotations
from sudoku_bench.validator import Violation, ViolationType


def generate_feedback(
    violations: list[Violation],
    cells_filled: int,
    total_cells: int,
    board_size: int | None = None,
) -> str:
    """
    Convert a list of violations + fill count into a feedback string for the LLM.
    Uses 1-indexed rows/columns in the output.
    """
    lines: list[str] = []

    for v in violations:
        if v.type == ViolationType.ROW_DUPLICATE:
            lines.append(f"Row {v.row + 1} has duplicate {v.value}s.")
        elif v.type == ViolationType.COL_DUPLICATE:
            lines.append(f"Column {v.col + 1} has duplicate {v.value}s.")
        elif v.type == ViolationType.BOX_DUPLICATE:
            lines.append(f"A box has duplicate {v.value}s.")
        elif v.type == ViolationType.MODIFIED_GIVEN:
            lines.append(
                f"R{v.row + 1}C{v.col + 1} was a given ({v.expected}), "
                f"you changed it — reverted."
            )
        elif v.type == ViolationType.OUT_OF_RANGE:
            range_str = f"1-{board_size}" if board_size else "the valid range"
            lines.append(
                f"R{v.row + 1}C{v.col + 1} has value {v.value}, "
                f"valid range is {range_str}."
            )

    fill_line = f"{cells_filled}/{total_cells} cells filled."

    if not violations and cells_filled == total_cells:
        return f"Correct! Puzzle complete. {fill_line}"

    if violations:
        return "\n".join(lines) + f"\n{fill_line}"
    else:
        return f"No rule violations. {fill_line}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_feedback.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/feedback.py tests/test_feedback.py
git commit -m "feat: feedback generator — rule violation messages, fill count"
```

---

## Task 7: Puzzle Bank

**Files:**
- Create: `src/sudoku_bench/puzzle_bank.py`
- Create: `tests/test_puzzle_bank.py`

Generates puzzles using `py-sudoku`, stores them in a JSON file. IDs are deterministic: `{size}x{size}_d{diff:.2f}_{seq:04d}`. The `main()` function is the `sudoku-gen` CLI entry point.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_puzzle_bank.py`:

```python
import json
import pytest
from pathlib import Path
from sudoku_bench.puzzle_bank import (
    generate_puzzle,
    load_bank,
    save_bank,
    append_puzzles,
    PuzzleRecord,
)
from sudoku_bench.board import Board


def test_generate_puzzle_returns_record():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    assert isinstance(record, PuzzleRecord)
    assert record.box_rows == 2
    assert record.box_cols == 2
    assert record.difficulty == 0.5
    assert record.id == "4x4_d0.50_0001"


def test_generate_puzzle_board_has_correct_size():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    assert len(record.board) == 4
    assert len(record.board[0]) == 4


def test_generate_puzzle_solution_is_complete():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    # Solution has no None values
    assert all(v is not None for row in record.solution for v in row)


def test_generate_puzzle_solution_valid_range():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    for row in record.solution:
        for v in row:
            assert 1 <= v <= 4


def test_generate_puzzle_givens_match_solution():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    for r, c in record.givens:
        assert record.board[r][c] == record.solution[r][c]


def test_generate_puzzle_empty_cells_in_board():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.75, seq=1)
    empty_count = sum(1 for row in record.board for v in row if v is None)
    total = 4 * 4
    # At least 1 cell should be empty with difficulty 0.75
    assert empty_count > 0


def test_puzzle_id_format():
    r = generate_puzzle(box_rows=3, box_cols=3, difficulty=0.25, seq=42)
    assert r.id == "9x9_d0.25_0042"


def test_save_and_load_bank(tmp_path):
    records = [
        generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=i)
        for i in range(1, 4)
    ]
    path = tmp_path / "puzzles.json"
    save_bank(records, path)
    loaded = load_bank(path)
    assert len(loaded) == 3
    assert loaded[0].id == records[0].id


def test_load_bank_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nonexistent.json"
    loaded = load_bank(path)
    assert loaded == []


def test_append_puzzles_no_duplicates(tmp_path):
    records1 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]
    path = tmp_path / "puzzles.json"
    save_bank(records1, path)

    records2 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]  # same id
    append_puzzles(records2, path)

    loaded = load_bank(path)
    assert len(loaded) == 1  # no duplicate added


def test_append_puzzles_adds_new(tmp_path):
    records1 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]
    path = tmp_path / "puzzles.json"
    save_bank(records1, path)

    records2 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=2)]
    append_puzzles(records2, path)

    loaded = load_bank(path)
    assert len(loaded) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_puzzle_bank.py -v
```

Expected: `ImportError` for `sudoku_bench.puzzle_bank`.

- [ ] **Step 3: Implement `src/sudoku_bench/puzzle_bank.py`**

```python
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sudoku import Sudoku


@dataclass
class PuzzleRecord:
    id: str
    box_rows: int
    box_cols: int
    difficulty: float
    board: list[list[Optional[int]]]   # None = empty cell
    givens: list[tuple[int, int]]       # positions of pre-filled cells
    solution: list[list[int]]


def generate_puzzle(
    box_rows: int, box_cols: int, difficulty: float, seq: int
) -> PuzzleRecord:
    """Generate one puzzle using py-sudoku."""
    size = box_rows * box_cols
    puzzle_id = f"{size}x{size}_d{difficulty:.2f}_{seq:04d}"

    sdk = Sudoku(box_rows, box_cols).difficulty(difficulty)
    board = sdk.board  # list[list[Optional[int]]]

    solution_sdk = sdk.solve()
    solution = solution_sdk.board  # list[list[int]]

    givens: list[tuple[int, int]] = []
    for r in range(size):
        for c in range(size):
            if board[r][c] is not None:
                givens.append((r, c))

    return PuzzleRecord(
        id=puzzle_id,
        box_rows=box_rows,
        box_cols=box_cols,
        difficulty=difficulty,
        board=board,
        givens=givens,
        solution=solution,
    )


def load_bank(path: Path) -> list[PuzzleRecord]:
    """Load puzzle bank from JSON. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    records = []
    for item in data:
        item["givens"] = [tuple(g) for g in item["givens"]]
        records.append(PuzzleRecord(**item))
    return records


def save_bank(records: list[PuzzleRecord], path: Path) -> None:
    """Save puzzle bank to JSON, overwriting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_puzzles(new_records: list[PuzzleRecord], path: Path) -> None:
    """Add new puzzles to the bank, skipping any that already exist (by id)."""
    existing = load_bank(path)
    existing_ids = {r.id for r in existing}
    to_add = [r for r in new_records if r.id not in existing_ids]
    save_bank(existing + to_add, path)


def main() -> None:
    """CLI entry point: generate puzzle bank from config.yaml."""
    import yaml

    if len(sys.argv) < 2:
        print("Usage: sudoku-gen <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bank_path = Path(config.get("benchmark", {}).get("puzzle_bank_file", "puzzles/puzzles.json"))
    puzzle_configs = config.get("puzzles", [])

    for pc in puzzle_configs:
        box_rows = pc["box_rows"]
        box_cols = pc["box_cols"]
        diffs = pc["diffs"]
        tests_per_diff = pc["tests_per_diff"]
        size = box_rows * box_cols

        for diff in diffs:
            existing = load_bank(bank_path)
            existing_ids = {r.id for r in existing}
            new_records = []
            for seq in range(1, tests_per_diff + 1):
                puzzle_id = f"{size}x{size}_d{diff:.2f}_{seq:04d}"
                if puzzle_id not in existing_ids:
                    print(f"Generating {puzzle_id}...")
                    record = generate_puzzle(box_rows, box_cols, diff, seq)
                    new_records.append(record)
            if new_records:
                append_puzzles(new_records, bank_path)

    print(f"Puzzle bank saved to {bank_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_puzzle_bank.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/puzzle_bank.py tests/test_puzzle_bank.py
git commit -m "feat: puzzle bank generator with py-sudoku, JSON storage, dedup"
```

---

## Task 8: Metrics Collector

**Files:**
- Create: `src/sudoku_bench/metrics.py`
- Create: `tests/test_metrics.py`

Accumulates stats for one puzzle run and appends a CSV row. The CSV header is written only if the file doesn't exist yet.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import csv
import pytest
from pathlib import Path
from sudoku_bench.metrics import PuzzleMetrics, append_csv_row, CSV_COLUMNS


def make_metrics(**kwargs):
    defaults = dict(
        model_name="llama3:8b",
        model_params="8B",
        model_quant="Q4_K_M",
        gpu_name="RTX 3090",
        gpu_max_vram_mb=24576,
        board_size="9x9",
        difficulty=0.5,
        puzzle_id="9x9_d0.50_0001",
        solved=True,
        best_pct_correct=100.0,
        final_pct_correct=100.0,
        best_num_errors=0,
        final_num_errors=0,
        total_tokens=4500,
        context_tokens_used=4500,
        context_pct_used=22.5,
        total_turns=3,
        total_seconds=45.2,
        avg_vram_mb=18000.0,
        max_vram_mb=19200,
        spilled_to_ram=False,
        avg_sys_ram_mb=512.0,
        max_sys_ram_mb=600,
        total_ram_mb=19800,
        malformed_submissions=0,
    )
    defaults.update(kwargs)
    return PuzzleMetrics(**defaults)


def test_csv_columns_match_dataclass():
    """Ensure CSV_COLUMNS covers all PuzzleMetrics fields."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PuzzleMetrics)}
    assert fields == set(CSV_COLUMNS)


def test_append_creates_file_with_header(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics()
    append_csv_row(m, path)
    assert path.exists()
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == 1


def test_append_second_row_no_duplicate_header(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics()
    append_csv_row(m, path)
    append_csv_row(m, path)
    with open(path) as f:
        content = f.read()
    header_count = content.count("model_name")
    assert header_count == 1


def test_append_values_correct(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics(puzzle_id="9x9_d0.50_0001", solved=True, total_tokens=9999)
    append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["puzzle_id"] == "9x9_d0.50_0001"
    assert row["solved"] == "True"
    assert row["total_tokens"] == "9999"


def test_append_none_values_as_empty_string(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics(gpu_name=None, gpu_max_vram_mb=None, avg_vram_mb=None,
                     max_vram_mb=None, spilled_to_ram=None,
                     avg_sys_ram_mb=None, max_sys_ram_mb=None, total_ram_mb=None)
    append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["gpu_name"] == ""
    assert row["avg_vram_mb"] == ""


def test_append_multiple_puzzle_ids(tmp_path):
    path = tmp_path / "results.csv"
    for i in range(5):
        m = make_metrics(puzzle_id=f"9x9_d0.50_{i:04d}")
        append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 5
    assert rows[2]["puzzle_id"] == "9x9_d0.50_0002"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError` for `sudoku_bench.metrics`.

- [ ] **Step 3: Implement `src/sudoku_bench/metrics.py`**

```python
from __future__ import annotations
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CSV_COLUMNS = [
    "model_name",
    "model_params",
    "model_quant",
    "gpu_name",
    "gpu_max_vram_mb",
    "board_size",
    "difficulty",
    "puzzle_id",
    "solved",
    "best_pct_correct",
    "final_pct_correct",
    "best_num_errors",
    "final_num_errors",
    "total_tokens",
    "context_tokens_used",
    "context_pct_used",
    "total_turns",
    "total_seconds",
    "avg_vram_mb",
    "max_vram_mb",
    "spilled_to_ram",
    "avg_sys_ram_mb",
    "max_sys_ram_mb",
    "total_ram_mb",
    "malformed_submissions",
]


@dataclass
class PuzzleMetrics:
    model_name: str
    model_params: Optional[str]
    model_quant: Optional[str]
    gpu_name: Optional[str]
    gpu_max_vram_mb: Optional[int]
    board_size: str
    difficulty: float
    puzzle_id: str
    solved: bool
    best_pct_correct: float
    final_pct_correct: float
    best_num_errors: int
    final_num_errors: int
    total_tokens: int
    context_tokens_used: int
    context_pct_used: float
    total_turns: int
    total_seconds: float
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[int]
    spilled_to_ram: Optional[bool]
    avg_sys_ram_mb: Optional[float]
    max_sys_ram_mb: Optional[int]
    total_ram_mb: Optional[int]
    malformed_submissions: int


def append_csv_row(metrics: PuzzleMetrics, path: Path) -> None:
    """Append one row to the CSV. Writes header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    row = {
        col: ("" if val is None else val)
        for col, val in zip(CSV_COLUMNS, dataclasses.astuple(metrics))
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sudoku_bench/metrics.py tests/test_metrics.py
git commit -m "feat: metrics dataclass + CSV append with header auto-write"
```

---

## Task 9: Config Loader

**Files:**
- Create: `src/sudoku_bench/config.py`

No unit tests — config loading is a thin YAML→dataclass layer; integration-tested via the runner.

- [ ] **Step 1: Implement `src/sudoku_bench/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    api_base: str = "http://localhost:11434/v1"
    name: Optional[str] = None


@dataclass
class PuzzleSetConfig:
    box_rows: int
    box_cols: int
    diffs: list[float]
    tests_per_diff: int


@dataclass
class BenchmarkConfig:
    gpu_poll_interval: float = 1.0
    results_file: str = "results/benchmark.csv"
    puzzle_bank_file: str = "puzzles/puzzles.json"
    context_buffer_tokens: int = 500


@dataclass
class Config:
    model: ModelConfig
    puzzles: list[PuzzleSetConfig]
    benchmark: BenchmarkConfig


def load_config(path: Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(**raw.get("model", {}))
    puzzles = [PuzzleSetConfig(**p) for p in raw.get("puzzles", [])]
    benchmark = BenchmarkConfig(**raw.get("benchmark", {}))

    return Config(model=model, puzzles=puzzles, benchmark=benchmark)
```

- [ ] **Step 2: Smoke test config loading**

```bash
python -c "
from pathlib import Path
from sudoku_bench.config import load_config
c = load_config(Path('config.example.yaml'))
print(c.model.api_base)
print(c.puzzles[0].box_rows)
"
```

Expected output:
```
http://localhost:11434/v1
2
```

- [ ] **Step 3: Commit**

```bash
git add src/sudoku_bench/config.py
git commit -m "feat: YAML config loader with dataclasses"
```

---

## Task 10: GPU Monitor

**Files:**
- Create: `src/sudoku_bench/gpu_monitor.py`

Polls `nvidia-smi` in a background thread. If `nvidia-smi` is unavailable, returns `None` stats gracefully (non-GPU machines can still run the benchmark).

- [ ] **Step 1: Implement `src/sudoku_bench/gpu_monitor.py`**

```python
from __future__ import annotations
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional
import psutil


@dataclass
class GPUStats:
    gpu_name: Optional[str]
    gpu_max_vram_mb: Optional[int]
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[int]
    spilled_to_ram: Optional[bool]
    avg_sys_ram_mb: Optional[float]
    max_sys_ram_mb: Optional[int]


class GPUMonitor:
    """
    Background thread that polls nvidia-smi at a configurable interval.
    Call start() before inference and stop() after to collect stats.
    If nvidia-smi is not available, all GPU fields are None.
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._vram_samples: list[int] = []
        self._sys_ram_samples: list[int] = []
        self._gpu_name: Optional[str] = None
        self._gpu_max_vram_mb: Optional[int] = None
        self._nvidia_available = self._check_nvidia()

    def _check_nvidia(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi"], capture_output=True, check=True, timeout=5
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _query_nvidia(self) -> Optional[tuple[int, str, int]]:
        """Returns (used_vram_mb, gpu_name, max_vram_mb) or None on error."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            used_mb = int(parts[1])
            total_mb = int(parts[2])
            return used_mb, name, total_mb
        except Exception:
            return None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._nvidia_available:
                data = self._query_nvidia()
                if data:
                    used_mb, name, total_mb = data
                    if self._gpu_name is None:
                        self._gpu_name = name
                        self._gpu_max_vram_mb = total_mb
                    self._vram_samples.append(used_mb)

            # Always sample system RAM
            ram_mb = int(psutil.virtual_memory().used / 1024 / 1024)
            self._sys_ram_samples.append(ram_mb)

            self._stop_event.wait(timeout=self.poll_interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._vram_samples = []
        self._sys_ram_samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

        if not self._vram_samples:
            avg_vram = None
            max_vram = None
            spilled = None
        else:
            avg_vram = sum(self._vram_samples) / len(self._vram_samples)
            max_vram = max(self._vram_samples)
            spilled = (
                max_vram > self._gpu_max_vram_mb
                if self._gpu_max_vram_mb
                else None
            )

        if not self._sys_ram_samples:
            avg_sys = None
            max_sys = None
        else:
            avg_sys = sum(self._sys_ram_samples) / len(self._sys_ram_samples)
            max_sys = max(self._sys_ram_samples)

        return GPUStats(
            gpu_name=self._gpu_name,
            gpu_max_vram_mb=self._gpu_max_vram_mb,
            avg_vram_mb=round(avg_vram, 1) if avg_vram is not None else None,
            max_vram_mb=max_vram,
            spilled_to_ram=spilled,
            avg_sys_ram_mb=round(avg_sys, 1) if avg_sys is not None else None,
            max_sys_ram_mb=max_sys,
        )
```

- [ ] **Step 2: Smoke test GPU monitor (runs even without nvidia-smi)**

```bash
python -c "
import time
from sudoku_bench.gpu_monitor import GPUMonitor
m = GPUMonitor(poll_interval=0.1)
m.start()
time.sleep(0.3)
stats = m.stop()
print('gpu_name:', stats.gpu_name)
print('avg_sys_ram_mb:', stats.avg_sys_ram_mb)
"
```

Expected: prints gpu_name (or None if no GPU) and a non-None avg_sys_ram_mb value.

- [ ] **Step 3: Commit**

```bash
git add src/sudoku_bench/gpu_monitor.py
git commit -m "feat: background GPU monitor via nvidia-smi + psutil system RAM"
```

---

## Task 11: Model Info Auto-Detection

**Files:**
- Create: `src/sudoku_bench/model_info.py`

Probes the server to detect if it's Ollama or vLLM, then extracts model name, param count, quantization, and context window size.

- [ ] **Step 1: Implement `src/sudoku_bench/model_info.py`**

```python
from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from typing import Optional
import urllib.request
import urllib.error
import json


@dataclass
class ModelInfo:
    name: str
    params: Optional[str]       # e.g. "70B"
    quant: Optional[str]        # e.g. "Q4_K_M"
    context_window: Optional[int]


def _get_json(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _extract_params(name: str) -> Optional[str]:
    """Extract parameter count from model name, e.g. '70b' → '70B'."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    return m.group(0).upper() if m else None


def _extract_quant(name: str) -> Optional[str]:
    """Extract quantization tag from model name, e.g. 'q4_K_M'."""
    m = re.search(r"[qQ]\d+[_\-]?[kK]?[_\-]?[mMsSlL]?", name)
    return m.group(0).upper() if m else None


def detect_model_info(api_base: str, name_override: Optional[str] = None) -> ModelInfo:
    """
    Auto-detect model metadata from a running Ollama or vLLM server.
    Falls back gracefully with warnings for any fields that can't be determined.
    """
    base = api_base.rstrip("/")

    # --- Try Ollama ---
    # List models: GET /api/tags
    tags = _get_json(f"{base.replace('/v1', '')}/api/tags")
    if tags and "models" in tags:
        models = tags["models"]
        model_name = name_override or (models[0]["name"] if models else None)
        if model_name:
            show = _get_json_post(
                f"{base.replace('/v1', '')}/api/show",
                {"name": model_name},
            )
            context_window = None
            if show:
                # context_window may be in modelinfo or parameters
                params_text = show.get("parameters", "")
                m = re.search(r"num_ctx\s+(\d+)", str(params_text))
                if m:
                    context_window = int(m.group(1))
                if context_window is None:
                    mi = show.get("modelinfo", {})
                    for key in mi:
                        if "context" in key.lower():
                            context_window = int(mi[key])
                            break

            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=context_window,
            )

    # --- Try vLLM ---
    # List models: GET /v1/models
    v1_models = _get_json(f"{base}/models")
    if v1_models and "data" in v1_models:
        data = v1_models["data"]
        model_name = name_override or (data[0]["id"] if data else None)
        if model_name:
            # vLLM doesn't expose context window via API; parse from name
            warnings.warn(f"Could not determine context window for {model_name} — set it manually if needed.")
            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=None,
            )

    # Fallback
    name = name_override or "unknown"
    warnings.warn(f"Could not auto-detect model info from {api_base}. Using name='{name}'.")
    return ModelInfo(
        name=name,
        params=_extract_params(name),
        quant=_extract_quant(name),
        context_window=None,
    )


def _get_json_post(url: str, body: dict, timeout: int = 5) -> Optional[dict]:
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None
```

- [ ] **Step 2: Commit**

```bash
git add src/sudoku_bench/model_info.py
git commit -m "feat: model info auto-detection for Ollama and vLLM"
```

---

## Task 12: Test Runner

**Files:**
- Create: `src/sudoku_bench/runner.py`

The conversational loop. Loads config + puzzle bank, runs each puzzle, collects metrics, appends CSV. This is integration-level code — tested manually, not unit tested.

- [ ] **Step 1: Create `src/sudoku_bench/runner.py`**

```python
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from sudoku_bench.board import Board
from sudoku_bench.config import load_config
from sudoku_bench.feedback import generate_feedback
from sudoku_bench.formatter import format_board
from sudoku_bench.gpu_monitor import GPUMonitor
from sudoku_bench.metrics import PuzzleMetrics, append_csv_row
from sudoku_bench.model_info import detect_model_info
from sudoku_bench.parser import parse_board
from sudoku_bench.puzzle_bank import load_bank, PuzzleRecord
from sudoku_bench.validator import validate


# ── System prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are solving a sudoku puzzle. This is a {size}x{size} board with {box_rows}x{box_cols} \
boxes. Valid numbers are 1-{size}.

Rules:
- Each row must contain each number from 1-{size} exactly once.
- Each column must contain each number from 1-{size} exactly once.
- Each {box_rows}x{box_cols} box must contain each number from 1-{size} exactly once.
- Cells marked with * are pre-filled givens and must not be changed.

When you want to check your progress, submit your current grid in the same format shown \
below. I will report any rule violations (duplicate numbers, modified given cells, \
out-of-range values) and how many cells are filled.

Submitting a fully correct, fully filled board will end the challenge.

Do not write code — reason through the puzzle logically.
"""

MALFORMED_RESPONSE = """\
Your board couldn't be parsed. Please resubmit in this exact format (replace . with your \
numbers, keep * on given cells):

{example}
"""


# ── Board helpers ─────────────────────────────────────────────────────────────

def _record_to_board(record: PuzzleRecord) -> Board:
    """Convert a PuzzleRecord to a Board (using givens as frozenset)."""
    return Board(
        cells=[row[:] for row in record.board],
        givens=frozenset(tuple(g) for g in record.givens),
        box_rows=record.box_rows,
        box_cols=record.box_cols,
    )


def _pct_correct(board: Board, solution: list[list[int]]) -> float:
    """Percentage of cells that match the solution."""
    size = board.size
    correct = sum(
        1
        for r in range(size)
        for c in range(size)
        if board.cells[r][c] == solution[r][c]
    )
    return round(correct / (size * size) * 100, 2)


# ── Single puzzle run ─────────────────────────────────────────────────────────

def run_puzzle(
    record: PuzzleRecord,
    client: OpenAI,
    model_name: str,
    context_window: Optional[int],
    context_buffer: int,
) -> dict:
    """
    Run one puzzle to completion (or context exhaustion).
    Returns a dict of per-puzzle stats (not including model/GPU info).
    """
    original_board = _record_to_board(record)
    size = original_board.size
    solution = record.solution

    system_prompt = SYSTEM_PROMPT.format(
        size=size,
        box_rows=record.box_rows,
        box_cols=record.box_cols,
    )
    initial_grid = format_board(original_board)
    first_user_message = f"Here is your puzzle:\n\n{initial_grid}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message},
    ]

    solved = False
    total_turns = 0
    total_tokens = 0
    context_tokens_used = 0
    malformed_submissions = 0
    best_pct = 0.0
    final_pct = 0.0
    best_errors = 9999
    final_errors = 0

    start_time = time.time()

    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        # Include thinking tokens if present (some models return them separately)
        thinking_tokens = getattr(usage, "completion_tokens_details", None)
        if thinking_tokens:
            reasoning = getattr(thinking_tokens, "reasoning_tokens", 0) or 0
        else:
            reasoning = 0

        turn_tokens = prompt_tokens + completion_tokens + reasoning
        total_tokens += completion_tokens + reasoning
        context_tokens_used = prompt_tokens + total_tokens

        assistant_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        # Try to parse a board from the response
        submitted = parse_board(assistant_text, record.box_rows, record.box_cols)

        if submitted is None:
            malformed_submissions += 1
            feedback_text = MALFORMED_RESPONSE.format(example=format_board(original_board))
            messages.append({"role": "user", "content": feedback_text})
            # Don't count malformed as a scored turn
            # Check context before looping
            if context_window and context_tokens_used + context_buffer >= context_window:
                break
            continue

        # Valid parse — validate and score
        total_turns += 1
        violations = validate(submitted, original=original_board)
        pct = _pct_correct(submitted, solution)
        num_errors = len(violations)

        if pct > best_pct:
            best_pct = pct
        if num_errors < best_errors:
            best_errors = num_errors
        final_pct = pct
        final_errors = num_errors

        # Check for completion
        if submitted.cells_filled == submitted.total_cells and not violations:
            solved = True
            break

        # Generate feedback
        feedback_text = generate_feedback(
            violations,
            cells_filled=submitted.cells_filled,
            total_cells=submitted.total_cells,
            board_size=size,
        )
        messages.append({"role": "user", "content": feedback_text})

        # Check context window
        if context_window and context_tokens_used + context_buffer >= context_window:
            break

    total_seconds = round(time.time() - start_time, 2)
    context_pct = (
        round(context_tokens_used / context_window * 100, 2)
        if context_window
        else 0.0
    )

    return {
        "solved": solved,
        "best_pct_correct": best_pct,
        "final_pct_correct": final_pct,
        "best_num_errors": best_errors if best_errors < 9999 else 0,
        "final_num_errors": final_errors,
        "total_tokens": total_tokens,
        "context_tokens_used": context_tokens_used,
        "context_pct_used": context_pct,
        "total_turns": total_turns,
        "total_seconds": total_seconds,
        "malformed_submissions": malformed_submissions,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sudoku-bench <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    print(f"Detecting model info from {config.model.api_base}...")
    model_info = detect_model_info(config.model.api_base, config.model.name)
    print(f"  Model: {model_info.name}")
    print(f"  Params: {model_info.params or 'unknown'}")
    print(f"  Quant: {model_info.quant or 'unknown'}")
    print(f"  Context window: {model_info.context_window or 'unknown'}")

    bank_path = Path(config.benchmark.puzzle_bank_file)
    puzzles = load_bank(bank_path)
    if not puzzles:
        print(f"No puzzles found at {bank_path}. Run: sudoku-gen {config_path}")
        sys.exit(1)

    results_path = Path(config.benchmark.results_file)
    client = OpenAI(base_url=config.model.api_base, api_key="not-needed")
    monitor = GPUMonitor(poll_interval=config.benchmark.gpu_poll_interval)

    print(f"\nRunning {len(puzzles)} puzzles...")

    for i, record in enumerate(puzzles, 1):
        size = record.box_rows * record.box_cols
        print(f"  [{i}/{len(puzzles)}] {record.id} ...", end=" ", flush=True)

        monitor.start()
        try:
            stats = run_puzzle(
                record=record,
                client=client,
                model_name=model_info.name,
                context_window=model_info.context_window,
                context_buffer=config.benchmark.context_buffer_tokens,
            )
        finally:
            gpu_stats = monitor.stop()

        total_ram = None
        if gpu_stats.max_vram_mb is not None and gpu_stats.max_sys_ram_mb is not None:
            total_ram = gpu_stats.max_vram_mb + gpu_stats.max_sys_ram_mb

        metrics = PuzzleMetrics(
            model_name=model_info.name,
            model_params=model_info.params,
            model_quant=model_info.quant,
            gpu_name=gpu_stats.gpu_name,
            gpu_max_vram_mb=gpu_stats.gpu_max_vram_mb,
            board_size=f"{size}x{size}",
            difficulty=record.difficulty,
            puzzle_id=record.id,
            solved=stats["solved"],
            best_pct_correct=stats["best_pct_correct"],
            final_pct_correct=stats["final_pct_correct"],
            best_num_errors=stats["best_num_errors"],
            final_num_errors=stats["final_num_errors"],
            total_tokens=stats["total_tokens"],
            context_tokens_used=stats["context_tokens_used"],
            context_pct_used=stats["context_pct_used"],
            total_turns=stats["total_turns"],
            total_seconds=stats["total_seconds"],
            avg_vram_mb=gpu_stats.avg_vram_mb,
            max_vram_mb=gpu_stats.max_vram_mb,
            spilled_to_ram=gpu_stats.spilled_to_ram,
            avg_sys_ram_mb=gpu_stats.avg_sys_ram_mb,
            max_sys_ram_mb=gpu_stats.max_sys_ram_mb,
            total_ram_mb=total_ram,
            malformed_submissions=stats["malformed_submissions"],
        )

        append_csv_row(metrics, results_path)
        status = "SOLVED" if stats["solved"] else f"{stats['final_pct_correct']:.1f}%"
        print(f"{status} in {stats['total_turns']} turns ({stats['total_seconds']}s)")

    print(f"\nResults appended to {results_path}")
```

- [ ] **Step 2: Verify imports resolve**

```bash
python -c "from sudoku_bench.runner import main; print('OK')"
```

Expected: `OK` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add src/sudoku_bench/runner.py
git commit -m "feat: conversational runner — puzzle loop, feedback, context tracking"
```

---

## Task 13: Setup Script

**Files:**
- Create: `setup.sh`

- [ ] **Step 1: Create `setup.sh`**

```bash
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
uv venv --python 3.11
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
```

- [ ] **Step 2: Make executable and test base setup**

```bash
chmod +x setup.sh
./setup.sh
```

Expected: uv present, virtual env created, all tests pass, no errors.

- [ ] **Step 3: Commit**

```bash
git add setup.sh
git commit -m "feat: setup.sh with uv bootstrap and --nvidia/--rocm/--apple GPU flags"
```

---

## Task 14: Full Test Suite Pass

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests in `test_board.py`, `test_formatter.py`, `test_parser.py`, `test_validator.py`, `test_feedback.py`, `test_puzzle_bank.py`, `test_metrics.py` PASS.

- [ ] **Step 2: Check test coverage**

```bash
pytest tests/ --cov=sudoku_bench --cov-report=term-missing
```

Note any uncovered lines in core logic modules. Add tests for any significant uncovered branches.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: full test suite passing"
```

---

## Notes for Implementer

- **Box dimension terminology:** `box_rows` = height of each box, `box_cols` = width. For a 6×6 board with 2×3 boxes: `box_rows=2, box_cols=3`. `board_size = box_rows * box_cols`.
- **1-indexed feedback:** All row/column references in feedback messages use 1-indexed numbers (human-friendly). Internal code uses 0-indexed.
- **Thinking tokens:** The runner uses `usage.completion_tokens_details.reasoning_tokens` for models that expose them (e.g., o1, QwQ). For others this is 0.
- **context_window=None:** If context window can't be detected, the runner never terminates early — puzzles run until solved. Warn the user to set it manually if needed.
- **py-sudoku API:** `Sudoku(box_rows, box_cols).difficulty(float)` — verify the constructor argument order matches py-sudoku's published API before Task 7.
