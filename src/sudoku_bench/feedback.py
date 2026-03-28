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
            if v.positions:
                r, c = v.positions[0]
                lines.append(f"Box containing R{r + 1}C{c + 1} has duplicate {v.value}s.")
            else:
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
