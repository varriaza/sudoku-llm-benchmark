from __future__ import annotations
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CSV_COLUMNS = [
    "model_name",
    "model_params",
    "model_quant",
    "gpu_name",
    "gpu_max_vram_mb",
    "board_size",
    "difficulty",
    "puzzle_id",
    "solved",
    "best_pct_correct",
    "final_pct_correct",
    "best_num_errors",
    "final_num_errors",
    "total_tokens",
    "context_tokens_used",
    "context_pct_used",
    "total_turns",
    "total_seconds",
    "avg_vram_mb",
    "max_vram_mb",
    "spilled_to_ram",
    "avg_sys_ram_mb",
    "max_sys_ram_mb",
    "total_ram_mb",
    "malformed_submissions",
]


@dataclass
class PuzzleMetrics:
    model_name: str
    model_params: Optional[str]
    model_quant: Optional[str]
    gpu_name: Optional[str]
    gpu_max_vram_mb: Optional[int]
    board_size: str
    difficulty: float
    puzzle_id: str
    solved: bool
    best_pct_correct: float
    final_pct_correct: float
    best_num_errors: int
    final_num_errors: int
    total_tokens: int
    context_tokens_used: int
    context_pct_used: float
    total_turns: int
    total_seconds: float
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[int]
    spilled_to_ram: Optional[bool]
    avg_sys_ram_mb: Optional[float]
    max_sys_ram_mb: Optional[int]
    total_ram_mb: Optional[int]
    malformed_submissions: int


def append_csv_row(metrics: PuzzleMetrics, path: Path) -> None:
    """Append one row to the CSV. Writes header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    row = {
        col: ("" if val is None else val)
        for col, val in zip(CSV_COLUMNS, dataclasses.astuple(metrics))
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
