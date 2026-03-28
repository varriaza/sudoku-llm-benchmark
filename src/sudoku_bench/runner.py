from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from sudoku_bench.board import Board
from sudoku_bench.config import load_config
from sudoku_bench.feedback import generate_feedback
from sudoku_bench.formatter import format_board
from sudoku_bench.gpu_monitor import GPUMonitor
from sudoku_bench.metrics import PuzzleMetrics, append_csv_row
from sudoku_bench.model_info import detect_model_info
from sudoku_bench.parser import parse_board
from sudoku_bench.puzzle_bank import load_bank, PuzzleRecord
from sudoku_bench.validator import validate


# ── System prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are solving a sudoku puzzle. This is a {size}x{size} board with {box_rows}x{box_cols} \
boxes. Valid numbers are 1-{size}.

Rules:
- Each row must contain each number from 1-{size} exactly once.
- Each column must contain each number from 1-{size} exactly once.
- Each {box_rows}x{box_cols} box must contain each number from 1-{size} exactly once.
- Cells marked with * are pre-filled givens and must not be changed.

When you want to check your progress, submit your current grid in the same format shown \
below. I will report any rule violations (duplicate numbers, modified given cells, \
out-of-range values) and how many cells are filled.

Submitting a fully correct, fully filled board will end the challenge.

Do not write code — reason through the puzzle logically.
"""

MALFORMED_RESPONSE = """\
Your board couldn't be parsed. Please resubmit in this exact format (replace . with your \
numbers, keep * on given cells):

{example}
"""


# ── Board helpers ─────────────────────────────────────────────────────────────

def _record_to_board(record: PuzzleRecord) -> Board:
    """Convert a PuzzleRecord to a Board (using givens as frozenset)."""
    return Board(
        cells=[row[:] for row in record.board],
        givens=frozenset(tuple(g) for g in record.givens),
        box_rows=record.box_rows,
        box_cols=record.box_cols,
    )


def _pct_correct(board: Board, solution: list[list[int]]) -> float:
    """Percentage of cells that match the solution."""
    size = board.size
    correct = sum(
        1
        for r in range(size)
        for c in range(size)
        if board.cells[r][c] == solution[r][c]
    )
    return round(correct / (size * size) * 100, 2)


# ── Single puzzle run ─────────────────────────────────────────────────────────

def run_puzzle(
    record: PuzzleRecord,
    client: OpenAI,
    model_name: str,
    context_window: Optional[int],
    context_buffer: int,
) -> dict:
    """
    Run one puzzle to completion (or context exhaustion).
    Returns a dict of per-puzzle stats (not including model/GPU info).
    """
    original_board = _record_to_board(record)
    size = original_board.size
    solution = record.solution

    system_prompt = SYSTEM_PROMPT.format(
        size=size,
        box_rows=record.box_rows,
        box_cols=record.box_cols,
    )
    initial_grid = format_board(original_board)
    first_user_message = f"Here is your puzzle:\n\n{initial_grid}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message},
    ]

    solved = False
    total_turns = 0
    total_tokens = 0
    context_tokens_used = 0
    malformed_submissions = 0
    best_pct = 0.0
    final_pct = 0.0
    best_errors = 9999
    final_errors = 0

    start_time = time.time()

    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        # Include thinking tokens if present (some models return them separately)
        thinking_tokens = getattr(usage, "completion_tokens_details", None)
        if thinking_tokens:
            reasoning = getattr(thinking_tokens, "reasoning_tokens", 0) or 0
        else:
            reasoning = 0

        total_tokens += completion_tokens + reasoning
        context_tokens_used = prompt_tokens + completion_tokens + reasoning

        assistant_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})

        # Try to parse a board from the response
        submitted = parse_board(assistant_text, record.box_rows, record.box_cols)

        if submitted is None:
            malformed_submissions += 1
            feedback_text = MALFORMED_RESPONSE.format(example=format_board(original_board))
            messages.append({"role": "user", "content": feedback_text})
            # Check context before looping
            if context_window and context_tokens_used + context_buffer >= context_window:
                break
            continue

        # Valid parse — validate and score
        total_turns += 1
        violations = validate(submitted, original=original_board)
        pct = _pct_correct(submitted, solution)
        num_errors = len(violations)

        if pct > best_pct:
            best_pct = pct
        if num_errors < best_errors:
            best_errors = num_errors
        final_pct = pct
        final_errors = num_errors

        # Check for completion
        if submitted.cells_filled == submitted.total_cells and not violations:
            solved = True
            break

        # Generate feedback
        feedback_text = generate_feedback(
            violations,
            cells_filled=submitted.cells_filled,
            total_cells=submitted.total_cells,
            board_size=size,
        )
        messages.append({"role": "user", "content": feedback_text})

        # Check context window
        if context_window and context_tokens_used + context_buffer >= context_window:
            break

    total_seconds = round(time.time() - start_time, 2)
    context_pct = (
        round(context_tokens_used / context_window * 100, 2)
        if context_window
        else 0.0
    )

    return {
        "solved": solved,
        "best_pct_correct": best_pct,
        "final_pct_correct": final_pct,
        "best_num_errors": best_errors if best_errors < 9999 else 0,
        "final_num_errors": final_errors,
        "total_tokens": total_tokens,
        "context_tokens_used": context_tokens_used,
        "context_pct_used": context_pct,
        "total_turns": total_turns,
        "total_seconds": total_seconds,
        "malformed_submissions": malformed_submissions,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sudoku-bench <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    print(f"Detecting model info from {config.model.api_base}...")
    model_info = detect_model_info(config.model.api_base, config.model.name)
    print(f"  Model: {model_info.name}")
    print(f"  Params: {model_info.params or 'unknown'}")
    print(f"  Quant: {model_info.quant or 'unknown'}")
    print(f"  Context window: {model_info.context_window or 'unknown'}")

    bank_path = Path(config.benchmark.puzzle_bank_file)
    puzzles = load_bank(bank_path)
    if not puzzles:
        print(f"No puzzles found at {bank_path}. Run: sudoku-gen {config_path}")
        sys.exit(1)

    results_path = Path(config.benchmark.results_file)
    client = OpenAI(base_url=config.model.api_base, api_key="not-needed")
    monitor = GPUMonitor(poll_interval=config.benchmark.gpu_poll_interval)

    print(f"\nRunning {len(puzzles)} puzzles...")

    for i, record in enumerate(puzzles, 1):
        size = record.box_rows * record.box_cols
        print(f"  [{i}/{len(puzzles)}] {record.id} ...", end=" ", flush=True)

        monitor.start()
        try:
            stats = run_puzzle(
                record=record,
                client=client,
                model_name=model_info.name,
                context_window=model_info.context_window,
                context_buffer=config.benchmark.context_buffer_tokens,
            )
        finally:
            gpu_stats = monitor.stop()

        total_ram = None
        if gpu_stats.max_vram_mb is not None and gpu_stats.max_sys_ram_mb is not None:
            total_ram = gpu_stats.max_vram_mb + gpu_stats.max_sys_ram_mb

        metrics = PuzzleMetrics(
            model_name=model_info.name,
            model_params=model_info.params,
            model_quant=model_info.quant,
            gpu_name=gpu_stats.gpu_name,
            gpu_max_vram_mb=gpu_stats.gpu_max_vram_mb,
            board_size=f"{size}x{size}",
            difficulty=record.difficulty,
            puzzle_id=record.id,
            solved=stats["solved"],
            best_pct_correct=stats["best_pct_correct"],
            final_pct_correct=stats["final_pct_correct"],
            best_num_errors=stats["best_num_errors"],
            final_num_errors=stats["final_num_errors"],
            total_tokens=stats["total_tokens"],
            context_tokens_used=stats["context_tokens_used"],
            context_pct_used=stats["context_pct_used"],
            total_turns=stats["total_turns"],
            total_seconds=stats["total_seconds"],
            avg_vram_mb=gpu_stats.avg_vram_mb,
            max_vram_mb=gpu_stats.max_vram_mb,
            spilled_to_ram=gpu_stats.spilled_to_ram,
            avg_sys_ram_mb=gpu_stats.avg_sys_ram_mb,
            max_sys_ram_mb=gpu_stats.max_sys_ram_mb,
            total_ram_mb=total_ram,
            malformed_submissions=stats["malformed_submissions"],
        )

        append_csv_row(metrics, results_path)
        status = "SOLVED" if stats["solved"] else f"{stats['final_pct_correct']:.1f}%"
        print(f"{status} in {stats['total_turns']} turns ({stats['total_seconds']}s)")

    print(f"\nResults appended to {results_path}")
