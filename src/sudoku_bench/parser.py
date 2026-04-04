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
) -> list[tuple[Optional[int], bool]] | None:
    """
    Parse one data row into a list of (value, is_given) pairs.
    Returns None if any token is non-numeric (other than '.' for empty).
    If box_cols is given and the line contains '|' separators, each segment
    must have exactly box_cols cells.
    Tokens may have '*' as prefix, suffix, or both (e.g. '2*', '*2', '*2*').
    """
    # Validate box column structure when | separators are present
    if box_cols is not None and "|" in line:
        segments = line.split("|")
        for seg in segments:
            # Parse segment tokens
            parsed_seg = []
            for token in seg.split():
                raw = token.strip("*")
                if raw == "." or raw.isdigit():
                    parsed_seg.append(raw)
                else:
                    # Non-cell token, skip (shouldn't happen in valid lines)
                    pass
            if parsed_seg and len(parsed_seg) != box_cols:
                return None

    # Strip box-column separators
    cleaned = line.replace("|", " ")
    tokens = cleaned.split()
    result: list[tuple[Optional[int], bool]] = []
    for token in tokens:
        is_given = token.startswith("*") or token.endswith("*")
        raw = token.strip("*")
        if raw == ".":
            result.append((None, is_given))
        elif raw.isdigit():
            result.append((int(raw), is_given))
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

    # Find the first contiguous window of `size` valid rows
    window: list[list[tuple[Optional[int], bool]]] = []
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
