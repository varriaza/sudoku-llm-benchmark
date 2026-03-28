from __future__ import annotations
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sudoku import Sudoku


@dataclass
class PuzzleRecord:
    id: str
    box_rows: int
    box_cols: int
    difficulty: float
    board: list[list[Optional[int]]]   # None = empty cell
    givens: list[tuple[int, int]]       # positions of pre-filled cells
    solution: list[list[int]]


def generate_puzzle(
    box_rows: int, box_cols: int, difficulty: float, seq: int
) -> PuzzleRecord:
    """Generate one puzzle using py-sudoku."""
    size = box_rows * box_cols
    puzzle_id = f"{size}x{size}_d{difficulty:.2f}_{seq:04d}"

    sdk = Sudoku(box_rows, box_cols).difficulty(difficulty)
    board = sdk.board  # list[list[Optional[int]]]

    solution_sdk = sdk.solve()
    solution = solution_sdk.board  # list[list[int]]

    givens: list[tuple[int, int]] = []
    for r in range(size):
        for c in range(size):
            if board[r][c] is not None:
                givens.append((r, c))

    return PuzzleRecord(
        id=puzzle_id,
        box_rows=box_rows,
        box_cols=box_cols,
        difficulty=difficulty,
        board=board,
        givens=givens,
        solution=solution,
    )


def load_bank(path: Path) -> list[PuzzleRecord]:
    """Load puzzle bank from JSON. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    records = []
    for item in data:
        item["givens"] = [tuple(g) for g in item["givens"]]
        records.append(PuzzleRecord(**item))
    return records


def save_bank(records: list[PuzzleRecord], path: Path) -> None:
    """Save puzzle bank to JSON, overwriting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def append_puzzles(new_records: list[PuzzleRecord], path: Path) -> None:
    """Add new puzzles to the bank, skipping any that already exist (by id)."""
    existing = load_bank(path)
    existing_ids = {r.id for r in existing}
    to_add = [r for r in new_records if r.id not in existing_ids]
    save_bank(existing + to_add, path)


def main() -> None:
    """CLI entry point: generate puzzle bank from config.yaml."""
    import yaml

    if len(sys.argv) < 2:
        print("Usage: sudoku-gen <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bank_path = Path(config.get("benchmark", {}).get("puzzle_bank_file", "puzzles/puzzles.json"))
    puzzle_configs = config.get("puzzles", [])

    for pc in puzzle_configs:
        box_rows = pc["box_rows"]
        box_cols = pc["box_cols"]
        diffs = pc["diffs"]
        tests_per_diff = pc["tests_per_diff"]
        size = box_rows * box_cols

        for diff in diffs:
            existing = load_bank(bank_path)
            existing_ids = {r.id for r in existing}
            new_records = []
            for seq in range(1, tests_per_diff + 1):
                puzzle_id = f"{size}x{size}_d{diff:.2f}_{seq:04d}"
                if puzzle_id not in existing_ids:
                    print(f"Generating {puzzle_id}...")
                    record = generate_puzzle(box_rows, box_cols, diff, seq)
                    new_records.append(record)
            if new_records:
                append_puzzles(new_records, bank_path)

    print(f"Puzzle bank saved to {bank_path}")
