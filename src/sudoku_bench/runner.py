from __future__ import annotations

import contextlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

from openai import OpenAI

from backends.llamacpp.monitor import LlamaCppMonitor
from backends.vllm.monitor import VLLMMonitor
from sudoku_bench.board import Board
from sudoku_bench.config import load_config
from sudoku_bench.feedback import generate_feedback
from sudoku_bench.formatter import format_board
from sudoku_bench.gpu_monitor import GPUMonitor
from sudoku_bench.metrics import PuzzleMetrics, append_csv_row
from sudoku_bench.model_info import detect_model_info
from sudoku_bench.parser import parse_board
from sudoku_bench.puzzle_bank import PuzzleRecord, load_bank
from sudoku_bench.server import start_server, stop_server
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
below. The computer will report back with any rule violations (duplicate numbers, modified given cells, \
out-of-range values) and how many cells are filled.

Submitting a fully correct, fully filled board will end the challenge.

Important: Submit early and often! Do not wait until you are fully confident. \
You have unlimited attempts, wrong answers are not penalized and partially completed submissions are expected. \
If you are unsure, just submit your best guess and use the feedback to improve.

Do not write code — reason through the puzzle logically.
"""

MALFORMED_RESPONSE = """\
Your board couldn't be parsed. Please resubmit in this exact format (replace . with your \
numbers, keep * on given cells):

{example}
"""


# ── Board helpers ─────────────────────────────────────────────────────────────


def _filter_puzzles(puzzles: list[PuzzleRecord], config) -> list[PuzzleRecord]:
    """Return only puzzles matching the config's puzzle specs, capped at tests_per_diff."""
    result = []
    for pc in config.puzzles:
        diffs = set(pc.diffs)
        seen: dict[float, int] = {}
        for record in puzzles:
            if record.box_rows != pc.box_rows or record.box_cols != pc.box_cols:
                continue
            if record.difficulty not in diffs:
                continue
            if seen.get(record.difficulty, 0) >= pc.tests_per_diff:
                continue
            result.append(record)
            seen[record.difficulty] = seen.get(record.difficulty, 0) + 1
    return result


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


def _write_llm_exchange(
    f: IO[str],
    puzzle_id: str,
    turn: int,
    messages: list[dict],
    response_text: str,
) -> None:
    """Append one LLM input/output exchange to the debug file."""
    f.write(f"\n{'=' * 80}\n")
    f.write(f"PUZZLE: {puzzle_id}  |  TURN: {turn}\n")
    f.write(f"{'=' * 80}\n\n")
    f.write("--- INPUT MESSAGES ---\n\n")
    for msg in messages:
        role = msg["role"].upper()
        f.write(f"[{role}]\n{msg['content']}\n\n")
    f.write("--- LLM RESPONSE ---\n\n")
    f.write(response_text)
    f.write("\n\n")
    f.flush()


def run_puzzle(
    record: PuzzleRecord,
    client: OpenAI,
    model_name: str,
    context_window: Optional[int],
    context_buffer: int,
    max_turns: int = 50,
    temperature: Optional[float] = None,
    llm_output_file: Optional[IO[str]] = None,
) -> dict:
    """
    Run one puzzle to completion (or context exhaustion).
    Returns a dict of per-puzzle stats (not including model/GPU info).
    """
    run_started_at = datetime.now().astimezone().isoformat(timespec="seconds")

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
    num_thinking_tokens = 0
    num_output_tokens = 0
    num_input_tokens = 0
    last_context_tokens = 0
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
            **({"temperature": temperature} if temperature is not None else {}),
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

        num_thinking_tokens += reasoning
        num_output_tokens += completion_tokens - reasoning
        num_input_tokens = prompt_tokens
        last_context_tokens = prompt_tokens + completion_tokens + reasoning

        msg = response.choices[0].message
        visible_text = msg.content or ""
        # Some backends (e.g. llama.cpp with Qwen3) return thinking separately
        reasoning_text = getattr(msg, "reasoning_content", None) or ""
        if reasoning_text:
            assistant_text = f"<think>{reasoning_text}</think>{visible_text}"
        else:
            assistant_text = visible_text

        if llm_output_file is not None:
            _write_llm_exchange(
                llm_output_file,
                puzzle_id=record.id,
                turn=total_turns + malformed_submissions + 1,
                messages=messages,
                response_text=assistant_text,
            )

        messages.append({"role": "assistant", "content": assistant_text})

        # Try to parse a board from the response
        submitted = parse_board(assistant_text, record.box_rows, record.box_cols)

        if submitted is None:
            malformed_submissions += 1
            feedback_text = MALFORMED_RESPONSE.format(
                example=format_board(original_board)
            )
            messages.append({"role": "user", "content": feedback_text})
            # Check context before looping
            if (
                context_window
                and last_context_tokens + context_buffer >= context_window
            ):
                break
            if (
                context_window is None
                and total_turns + malformed_submissions >= max_turns
            ):
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
        if context_window and last_context_tokens + context_buffer >= context_window:
            break
        if context_window is None and total_turns + malformed_submissions >= max_turns:
            break

    run_finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    total_seconds = round(time.time() - start_time, 2)
    total_response_tokens = num_thinking_tokens + num_output_tokens
    total_tokens_used = num_input_tokens + total_response_tokens
    context_pct = (
        round(num_input_tokens / context_window * 100, 2) if context_window else 0.0
    )

    return {
        "solved": solved,
        "best_pct_correct": best_pct,
        "final_pct_correct": final_pct,
        "best_num_errors": best_errors if best_errors < 9999 else 0,
        "final_num_errors": final_errors,
        "num_input_tokens": num_input_tokens,
        "num_thinking_tokens": num_thinking_tokens,
        "num_output_tokens": num_output_tokens,
        "total_response_tokens": total_response_tokens,
        "total_tokens_used": total_tokens_used,
        "context_pct_used": context_pct,
        "total_turns": total_turns,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
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

    server_proc = None
    if config.serve:
        model_name = config.model.name or ""
        # For GGUF models specified as "repo:tag", split into repo and tag parts
        if ":" in model_name:
            model_repo, model_tag = model_name.rsplit(":", 1)
        else:
            model_repo, model_tag = model_name, ""
        command = [
            arg.replace("{model}", model_name)
            .replace("{model_repo}", model_repo)
            .replace("{model_tag}", model_tag)
            for arg in config.serve.command
        ]
        print(f"Starting server: {' '.join(command)}")
        try:
            server_proc = start_server(
                command=command,
                api_base=config.model.api_base,
                startup_timeout=config.serve.startup_timeout,
            )
        except (RuntimeError, TimeoutError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        print("  Server ready.")

    try:
        _run_benchmark(config, config_path)
    finally:
        stop_server(server_proc)


def _model_file_size_gb(model_path: Optional[str]) -> Optional[float]:
    if model_path is None:
        return None
    try:
        size_bytes = Path(model_path).stat().st_size
        return round(size_bytes / 1e9, 3)
    except OSError:
        return None


def _run_benchmark(config, config_path: Path) -> None:
    print(f"Detecting model info from {config.model.api_base}...")
    model_info = detect_model_info(config.model.api_base, config.model.name)

    # Config override takes precedence; error if still unknown after merge
    context_window = config.model.context_window or model_info.context_window
    if context_window is None:
        print("Error: could not detect context window size for this model.")
        print("Set it explicitly in your config: model.context_window: <tokens>")
        sys.exit(1)

    print(f"  Model: {model_info.name}")
    print(f"  Params: {model_info.params or 'unknown'}")
    print(f"  Quant: {model_info.quant or 'unknown'}")
    print(f"  BPW: {model_info.bits_per_weight or 'unknown'}")
    print(f"  Backend: {model_info.backend_type}")
    print(f"  Context window: {context_window}")

    model_file_size = _model_file_size_gb(config.model.model_path)

    bank_path = Path(config.benchmark.puzzle_bank_file)
    puzzles = load_bank(bank_path)
    if config.puzzles:
        puzzles = _filter_puzzles(puzzles, config)
    if not puzzles:
        print(f"No puzzles found at {bank_path}. Run: sudoku-gen {config_path}")
        sys.exit(1)

    results_path = Path(config.benchmark.results_file)
    client = OpenAI(base_url=config.model.api_base, api_key="not-needed")
    if model_info.backend_type == "vllm":
        monitor = VLLMMonitor(
            api_base=config.model.api_base,
            poll_interval=config.benchmark.gpu_poll_interval,
        )
        print("  Monitor: vLLM /metrics")
    elif model_info.backend_type == "llamacpp":
        monitor = LlamaCppMonitor(
            api_base=config.model.api_base,
            poll_interval=config.benchmark.gpu_poll_interval,
        )
        print("  Monitor: llama.cpp /metrics + nvidia-smi")
    else:
        monitor = GPUMonitor(poll_interval=config.benchmark.gpu_poll_interval)
        print("  Monitor: nvidia-smi")

    llm_output_path: Optional[Path] = None
    if config.benchmark.save_llm_output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_output_path = results_path.parent / f"llm_output_{ts}.txt"
        llm_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  LLM output: {llm_output_path}")

    print(f"\nRunning {len(puzzles)} puzzles...")

    for i, record in enumerate(puzzles, 1):
        size = record.box_rows * record.box_cols
        print(f"  [{i}/{len(puzzles)}] {record.id} ...", end=" ", flush=True)

        monitor.start()
        try:
            with (
                open(llm_output_path, "a", encoding="utf-8")
                if llm_output_path
                else contextlib.nullcontext()
            ) as llm_file:
                stats = run_puzzle(
                    record=record,
                    client=client,
                    model_name=model_info.name,
                    context_window=context_window,
                    context_buffer=config.benchmark.context_buffer_tokens,
                    max_turns=config.benchmark.max_turns_per_puzzle,
                    temperature=0.1,
                    llm_output_file=llm_file if llm_output_path else None,
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
            bits_per_weight=model_info.bits_per_weight,
            model_file_size_gb=model_file_size,
            backend=model_info.backend_type,
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
            num_input_tokens=stats["num_input_tokens"],
            num_thinking_tokens=stats["num_thinking_tokens"],
            num_output_tokens=stats["num_output_tokens"],
            total_response_tokens=stats["total_response_tokens"],
            total_tokens_used=stats["total_tokens_used"],
            context_pct_used=stats["context_pct_used"],
            avg_gen_toks_per_sec=gpu_stats.avg_gen_toks_per_sec,
            median_gen_toks_per_sec=gpu_stats.median_gen_toks_per_sec,
            max_gen_toks_per_sec=gpu_stats.max_gen_toks_per_sec,
            avg_prompt_toks_per_sec=gpu_stats.avg_prompt_toks_per_sec,
            median_prompt_toks_per_sec=gpu_stats.median_prompt_toks_per_sec,
            max_prompt_toks_per_sec=gpu_stats.max_prompt_toks_per_sec,
            total_turns=stats["total_turns"],
            run_started_at=stats["run_started_at"],
            run_finished_at=stats["run_finished_at"],
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
