"""Tests for runner behaviour."""
from __future__ import annotations
from unittest.mock import MagicMock, patch

from sudoku_bench.runner import run_puzzle, main, _filter_puzzles
from sudoku_bench.puzzle_bank import PuzzleRecord
from sudoku_bench.config import PuzzleSetConfig


def _make_record() -> PuzzleRecord:
    """A minimal 4x4 puzzle record."""
    board = [
        [None, 2, 3, 4],
        [3, 4, None, 2],
        [2, None, 4, 3],
        [4, 3, 2, None],
    ]
    givens = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)]
    solution = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]
    return PuzzleRecord(
        id="4x4_d0.25_0001",
        box_rows=2,
        box_cols=2,
        difficulty=0.25,
        board=board,
        givens=givens,
        solution=solution,
    )


def _make_client(response_text: str) -> MagicMock:
    """Build a mock OpenAI client that always returns the given text."""
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.completion_tokens_details = None

    choice = MagicMock()
    choice.message.content = response_text

    completion = MagicMock()
    completion.usage = usage
    completion.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


def test_max_turns_stops_loop_when_context_window_none():
    """When context_window is None, max_turns prevents infinite loop."""
    # Model always responds with gibberish (malformed) → never solves
    client = _make_client("I am thinking about this puzzle very carefully.")
    record = _make_record()

    result = run_puzzle(
        record=record,
        client=client,
        model_name="test-model",
        context_window=None,
        context_buffer=0,
        max_turns=3,
    )

    assert not result["solved"]
    # Should have stopped at or before max_turns
    assert result["total_turns"] + result["malformed_submissions"] <= 3


def test_max_turns_not_triggered_when_context_window_known():
    """When context_window is set, the context guard fires before max_turns."""
    # Small context window (101 tokens) ensures context guard triggers first.
    # prompt_tokens=100, completion_tokens=50 → context_tokens_used=150 > 101
    client = _make_client("I am thinking about this puzzle very carefully.")
    record = _make_record()

    result = run_puzzle(
        record=record,
        client=client,
        model_name="test-model",
        context_window=101,
        context_buffer=0,
        max_turns=50,  # high enough to not trigger
    )

    assert not result["solved"]
    # Context guard fires on the first response
    assert client.chat.completions.create.call_count == 1


# ── {model} substitution in serve command ────────────────────────────────────

def test_model_placeholder_substituted_in_serve_command():
    """serve.command {model} is replaced with model.name before starting server."""
    from sudoku_bench.config import Config, ModelConfig, BenchmarkConfig, ServeConfig

    config = Config(
        model=ModelConfig(
            api_base="http://localhost:8000/v1",
            name="meta-llama/Llama-3.1-8B-Instruct",
            context_window=32768,
        ),
        puzzles=[],
        benchmark=BenchmarkConfig(),
        serve=ServeConfig(command=["vllm", "serve", "{model}"], startup_timeout=1),
    )

    captured = {}

    def fake_start_server(command, api_base, startup_timeout):
        captured["command"] = command
        return MagicMock()

    with patch("sudoku_bench.runner.load_config", return_value=config), \
         patch("sudoku_bench.runner.start_server", side_effect=fake_start_server), \
         patch("sudoku_bench.runner.stop_server"), \
         patch("sudoku_bench.runner._run_benchmark"), \
         patch("sys.argv", ["sudoku-bench", "configs/example.yaml"]):
        main()

    assert captured["command"] == ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]


# ── _filter_puzzles ───────────────────────────────────────────────────────────

def _make_records() -> list[PuzzleRecord]:
    """A small bank: 2x2 and 3x3, each at difficulties 0.25 and 0.5, 2 per diff."""
    base = dict(board=[[None]], givens=[], solution=[[1]])
    return [
        PuzzleRecord(id=f"{r}x{c}_d{d:.2f}_{i:04d}", box_rows=r, box_cols=c, difficulty=d, **base)
        for r, c in [(2, 2), (3, 3)]
        for d in [0.25, 0.5]
        for i in [1, 2]
    ]


def _cfg(*specs):
    """Build a minimal config stub with the given PuzzleSetConfig list."""
    cfg = MagicMock()
    cfg.puzzles = list(specs)
    return cfg


def test_filter_by_dimensions():
    """Only puzzles matching box_rows/box_cols are returned."""
    puzzles = _make_records()
    result = _filter_puzzles(puzzles, _cfg(PuzzleSetConfig(box_rows=2, box_cols=2, diffs=[0.25, 0.5], tests_per_diff=2)))
    assert all(r.box_rows == 2 and r.box_cols == 2 for r in result)
    assert len(result) == 4


def test_filter_by_difficulty():
    """Only puzzles whose difficulty is in diffs are returned."""
    puzzles = _make_records()
    result = _filter_puzzles(puzzles, _cfg(PuzzleSetConfig(box_rows=2, box_cols=2, diffs=[0.25], tests_per_diff=2)))
    assert all(r.difficulty == 0.25 for r in result)
    assert len(result) == 2


def test_filter_respects_tests_per_diff():
    """At most tests_per_diff puzzles per difficulty are returned."""
    puzzles = _make_records()
    result = _filter_puzzles(puzzles, _cfg(PuzzleSetConfig(box_rows=2, box_cols=2, diffs=[0.25, 0.5], tests_per_diff=1)))
    assert len(result) == 2  # 1 per difficulty


def test_filter_multiple_specs():
    """Multiple PuzzleSetConfigs are each applied and results combined."""
    puzzles = _make_records()
    result = _filter_puzzles(puzzles, _cfg(
        PuzzleSetConfig(box_rows=2, box_cols=2, diffs=[0.25], tests_per_diff=2),
        PuzzleSetConfig(box_rows=3, box_cols=3, diffs=[0.5], tests_per_diff=1),
    ))
    ids = [r.id for r in result]
    assert sum(1 for r in result if r.box_rows == 2) == 2
    assert sum(1 for r in result if r.box_rows == 3) == 1
    assert len(ids) == 3


def test_filter_empty_bank():
    """Empty bank returns empty list without error."""
    result = _filter_puzzles([], _cfg(PuzzleSetConfig(box_rows=2, box_cols=2, diffs=[0.25], tests_per_diff=3)))
    assert result == []


def test_filter_no_specs():
    """Empty puzzles config returns empty list."""
    result = _filter_puzzles(_make_records(), _cfg())
    assert result == []
