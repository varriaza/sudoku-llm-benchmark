import json
import pytest
from pathlib import Path
from sudoku_bench.puzzle_bank import (
    generate_puzzle,
    load_bank,
    save_bank,
    append_puzzles,
    PuzzleRecord,
)


def test_generate_puzzle_returns_record():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    assert isinstance(record, PuzzleRecord)
    assert record.box_rows == 2
    assert record.box_cols == 2
    assert record.difficulty == 0.5
    assert record.id == "4x4_d0.50_0001"


def test_generate_puzzle_board_has_correct_size():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    assert len(record.board) == 4
    assert len(record.board[0]) == 4


def test_generate_puzzle_solution_is_complete():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    # Solution has no None values
    assert all(v is not None for row in record.solution for v in row)


def test_generate_puzzle_solution_valid_range():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    for row in record.solution:
        for v in row:
            assert 1 <= v <= 4


def test_generate_puzzle_givens_match_solution():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)
    for r, c in record.givens:
        assert record.board[r][c] == record.solution[r][c]


def test_generate_puzzle_empty_cells_in_board():
    record = generate_puzzle(box_rows=2, box_cols=2, difficulty=0.75, seq=1)
    empty_count = sum(1 for row in record.board for v in row if v is None)
    # At least 1 cell should be empty with difficulty 0.75
    assert empty_count > 0


def test_puzzle_id_format():
    r = generate_puzzle(box_rows=3, box_cols=3, difficulty=0.25, seq=42)
    assert r.id == "9x9_d0.25_0042"


def test_save_and_load_bank(tmp_path):
    records = [
        generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=i)
        for i in range(1, 4)
    ]
    path = tmp_path / "puzzles.json"
    save_bank(records, path)
    loaded = load_bank(path)
    assert len(loaded) == 3
    assert loaded[0].id == records[0].id


def test_load_bank_missing_file_returns_empty(tmp_path):
    path = tmp_path / "nonexistent.json"
    loaded = load_bank(path)
    assert loaded == []


def test_append_puzzles_no_duplicates(tmp_path):
    records1 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]
    path = tmp_path / "puzzles.json"
    save_bank(records1, path)

    records2 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]  # same id
    append_puzzles(records2, path)

    loaded = load_bank(path)
    assert len(loaded) == 1  # no duplicate added


def test_append_puzzles_adds_new(tmp_path):
    records1 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=1)]
    path = tmp_path / "puzzles.json"
    save_bank(records1, path)

    records2 = [generate_puzzle(box_rows=2, box_cols=2, difficulty=0.5, seq=2)]
    append_puzzles(records2, path)

    loaded = load_bank(path)
    assert len(loaded) == 2
