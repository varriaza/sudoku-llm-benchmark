"""Tests for run_puzzle max_turns safeguard."""
from __future__ import annotations
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from sudoku_bench.runner import run_puzzle
from sudoku_bench.puzzle_bank import PuzzleRecord


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
