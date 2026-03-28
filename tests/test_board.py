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
    assert board.cells_filled == 8
    assert board.total_cells == 16


def test_board_copy_with_cells():
    cells = [[1, None], [None, 2]]
    board = Board(cells=cells, givens=frozenset({(0, 0)}), box_rows=2, box_cols=2)
    new_cells = [[1, 3], [4, 2]]
    copy = board.copy_with_cells(new_cells)
    assert copy.cells == new_cells
    assert copy.givens == board.givens
    assert copy.box_rows == board.box_rows
