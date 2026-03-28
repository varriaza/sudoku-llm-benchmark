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
    cells9 = [[None] * 9 for _ in range(9)]
    cells9[0][0] = 5
    cells9[4][4] = 3
    board = make_board(cells9, {(0, 0)}, box_rows=3, box_cols=3)
    result = format_board(board)
    # (0,0) is given → "5*" present
    assert "5*" in result
    # (4,4) is not given → "3*" NOT present
    assert "3*" not in result
