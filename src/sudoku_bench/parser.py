from __future__ import annotations
import re
from typing import Optional
from sudoku_bench.board import Board


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (including multiline) from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _is_separator_line(line: str) -> bool:
    """True if the line is a box-row separator (dashes and + signs only)."""
    stripped = line.strip()
    return bool(stripped) and all(c in "-+" for c in stripped.replace(" ", ""))


def _parse_data_line(
    line: str, box_cols: Optional[int] = None
) -> list[Optional[int]] | None:
    """
    Parse one data row into a list of values (int or None for empty cells).
    Returns None if any token is non-numeric (other than '.' for empty).
    If box_cols is given and the line contains '|' separators, each segment
    must have exactly box_cols cells.
    """
    # Validate box column structure when | separators are present
    if box_cols is not None and "|" in line:
        segments = line.split("|")
        for seg in segments:
            parsed_seg = [
                t for t in seg.split()
                if t == "." or t.isdigit()
            ]
            if parsed_seg and len(parsed_seg) != box_cols:
                return None

    # Strip box-column separators
    cleaned = line.replace("|", " ")
    tokens = cleaned.split()
    result: list[Optional[int]] = []
    for token in tokens:
        if token == ".":
            result.append(None)
        elif token.isdigit():
            result.append(int(token))
        else:
            return None
    return result if result else None


def _parse_board_from_text(text: str, box_rows: int, box_cols: int) -> Optional[Board]:
    """Core board extraction from a text string."""
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
        parsed = _parse_data_line(stripped, box_cols=box_cols)
        if parsed is not None and len(parsed) == size:
            data_lines.append(stripped)
        elif parsed is not None and len(parsed) != size:
            # Wrong column count — might be prose that happens to have numbers;
            # collect anyway and fail at validation
            data_lines.append(stripped)

    # Find the last contiguous window of `size` valid rows.
    # Using the last window (not the first) means that when an LLM echoes the
    # original puzzle before presenting its solution, we parse the solution.
    window: list[list[Optional[int]]] = []
    best_window: list[list[Optional[int]]] = []
    for line in data_lines:
        parsed = _parse_data_line(line, box_cols=box_cols)
        if parsed is None:
            window = []
            continue
        if len(parsed) != size:
            window = []
            continue
        window.append(parsed)
        if len(window) == size:
            best_window = window[:]
            window = []

    if not best_window:
        return None

    return Board(
        cells=[row[:] for row in best_window],
        givens=frozenset(),
        box_rows=box_rows,
        box_cols=box_cols,
    )


def parse_board(text: str, box_rows: int, box_cols: int) -> Optional[Board]:
    """
    Extract a Board from free-form LLM text.
    Returns None if no valid board of the expected size can be found.

    <think>...</think> blocks are always stripped before parsing so that
    partial or intermediate boards in the LLM's reasoning chain are ignored.
    A board that appears only inside a think block is not considered a
    submission.
    """
    stripped = _strip_think_blocks(text)
    return _parse_board_from_text(stripped, box_rows, box_cols)
