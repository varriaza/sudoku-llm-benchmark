import csv
from pathlib import Path
from sudoku_bench.metrics import PuzzleMetrics, append_csv_row, CSV_COLUMNS


def make_metrics(**kwargs):
    defaults = dict(
        model_name="llama3:8b",
        model_params="8B",
        model_quant="Q4_K_M",
        gpu_name="RTX 3090",
        gpu_max_vram_mb=24576,
        board_size="9x9",
        difficulty=0.5,
        puzzle_id="9x9_d0.50_0001",
        solved=True,
        best_pct_correct=100.0,
        final_pct_correct=100.0,
        best_num_errors=0,
        final_num_errors=0,
        total_tokens=4500,
        context_tokens_used=4500,
        context_pct_used=22.5,
        total_turns=3,
        total_seconds=45.2,
        avg_vram_mb=18000.0,
        max_vram_mb=19200,
        spilled_to_ram=False,
        avg_sys_ram_mb=512.0,
        max_sys_ram_mb=600,
        total_ram_mb=19800,
        malformed_submissions=0,
    )
    defaults.update(kwargs)
    return PuzzleMetrics(**defaults)


def test_csv_columns_match_dataclass():
    """Ensure CSV_COLUMNS covers all PuzzleMetrics fields."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PuzzleMetrics)}
    assert fields == set(CSV_COLUMNS)


def test_append_creates_file_with_header(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics()
    append_csv_row(m, path)
    assert path.exists()
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == 1


def test_append_second_row_no_duplicate_header(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics()
    append_csv_row(m, path)
    append_csv_row(m, path)
    with open(path) as f:
        content = f.read()
    header_count = content.count("model_name")
    assert header_count == 1


def test_append_values_correct(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics(puzzle_id="9x9_d0.50_0001", solved=True, total_tokens=9999)
    append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["puzzle_id"] == "9x9_d0.50_0001"
    assert row["solved"] == "True"
    assert row["total_tokens"] == "9999"


def test_append_none_values_as_empty_string(tmp_path):
    path = tmp_path / "results.csv"
    m = make_metrics(gpu_name=None, gpu_max_vram_mb=None, avg_vram_mb=None,
                     max_vram_mb=None, spilled_to_ram=None,
                     avg_sys_ram_mb=None, max_sys_ram_mb=None, total_ram_mb=None)
    append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["gpu_name"] == ""
    assert row["avg_vram_mb"] == ""


def test_append_multiple_puzzle_ids(tmp_path):
    path = tmp_path / "results.csv"
    for i in range(5):
        m = make_metrics(puzzle_id=f"9x9_d0.50_{i:04d}")
        append_csv_row(m, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 5
    assert rows[2]["puzzle_id"] == "9x9_d0.50_0002"
