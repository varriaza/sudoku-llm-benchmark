from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    api_base: str = "http://localhost:11434/v1"
    name: Optional[str] = None


@dataclass
class PuzzleSetConfig:
    box_rows: int
    box_cols: int
    diffs: list[float]
    tests_per_diff: int


@dataclass
class BenchmarkConfig:
    gpu_poll_interval: float = 1.0
    results_file: str = "results/benchmark.csv"
    puzzle_bank_file: str = "puzzles/puzzles.json"
    context_buffer_tokens: int = 500


@dataclass
class Config:
    model: ModelConfig
    puzzles: list[PuzzleSetConfig]
    benchmark: BenchmarkConfig


def load_config(path: Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(**raw.get("model", {}))
    puzzles = [PuzzleSetConfig(**p) for p in raw.get("puzzles", [])]
    benchmark = BenchmarkConfig(**raw.get("benchmark", {}))

    return Config(model=model, puzzles=puzzles, benchmark=benchmark)
