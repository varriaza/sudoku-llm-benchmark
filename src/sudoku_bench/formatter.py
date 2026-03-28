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
