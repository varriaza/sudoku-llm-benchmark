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


def test_parse_all_cells_starred():
    """LLM marks all cells with * — should still parse all 16 values."""
    text = """
2* 3* | 1* 4*
4* 1* | 3* 2*
------ + ------
3* 2* | 4* 1*
1* 4* | 2* 3*
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells_filled == 16
    assert board.cells[1] == [4, 1, 3, 2]
    assert board.cells[2] == [3, 2, 4, 1]


def test_parse_in_code_fence():
    """LLM wraps board in a markdown code fence."""
    text = """
```
2* 3* | 1* 4*
4  1  | 3* 2*
------ + ------
3  2  | 4* 1
1* 4* | 2* 3*
```

This fills:
- R2C1: 4
- R2C2: 1
- R3C1: 3
- R3C4: 1
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells_filled == 16
    assert board.cells[1] == [4, 1, 3, 2]
    assert board.cells[2] == [3, 2, 4, 1]


def test_parse_ignores_think_block_partial_board():
    """Partial board inside <think> should not shadow the final complete board."""
    text = """<think>
Let me work through this step by step.

2* 3* | 1* 4*
4* .  | 3* 2*
------ + ------
.  2* | 4* .
1* 4* | 2* 3*

I still need to fill in R2C2, R3C1, R3C4...
</think>

Here is my completed board:

2* 3* | 1* 4*
4* 1* | 3* 2*
------ + ------
3* 2* | 4* 1*
1* 4* | 2* 3*
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells_filled == 16
    assert board.cells[1] == [4, 1, 3, 2]
    assert board.cells[2] == [3, 2, 4, 1]


def test_parse_falls_back_to_think_block_if_no_board_outside():
    """If the only board is inside <think>, fall back and return it."""
    text = """<think>
2* 3* | 1* 4*
4* 1* | 3* 2*
------ + ------
3* 2* | 4* 1*
1* 4* | 2* 3*
</think>

I have submitted my answer.
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells_filled == 16


def test_parse_star_prefix_numbers():
    """Handle *4* or *4 tokens where * appears before the digit."""
    text = """
*2* *3* | *1* *4*
*4* *1* | *3* *2*
-------- + --------
*3* *2* | *4* *1*
*1* *4* | *2* *3*
"""
    board = parse_board(text, box_rows=2, box_cols=2)
    assert board is not None
    assert board.cells_filled == 16
    assert board.cells[0] == [2, 3, 1, 4]
