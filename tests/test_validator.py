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
