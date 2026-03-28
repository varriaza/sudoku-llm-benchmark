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
        row_seen: dict[int, list[int]] = {}
        for c in range(size):
            val = board.cells[r][c]
            if val is not None:
                row_seen.setdefault(val, []).append(c)
        for val, cols in row_seen.items():
            if len(cols) > 1:
                violations.append(Violation(
                    type=ViolationType.ROW_DUPLICATE,
                    row=r, value=val,
                    positions=[(r, c) for c in cols],
                ))

    # Column duplicate check
    for c in range(size):
        col_seen: dict[int, list[int]] = {}
        for r in range(size):
            val = board.cells[r][c]
            if val is not None:
                col_seen.setdefault(val, []).append(r)
        for val, rows in col_seen.items():
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
            box_seen: dict[int, list[tuple[int, int]]] = {}
            for r in range(br * board.box_rows, (br + 1) * board.box_rows):
                for c in range(bc * board.box_cols, (bc + 1) * board.box_cols):
                    val = board.cells[r][c]
                    if val is not None:
                        box_seen.setdefault(val, []).append((r, c))
            for val, positions in box_seen.items():
                if len(positions) > 1:
                    violations.append(Violation(
                        type=ViolationType.BOX_DUPLICATE,
                        value=val,
                        positions=positions,
                    ))

    return violations
