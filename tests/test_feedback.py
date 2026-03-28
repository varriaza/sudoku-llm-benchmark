import pytest
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
