# Token Breakdown Column Redesign

**Date:** 2026-03-30
**Status:** Approved

## Problem

The current CSV has two token columns that conflate different things:

- `total_tokens` — accumulated output tokens only (completion + reasoning across all turns). Does **not** include input tokens.
- `context_tokens_used` — prompt + completion + reasoning from the **last API call only**, used as a proxy for context window pressure.

Neither column name makes its scope obvious, and there is no way to separate thinking tokens from visible output tokens.

## Design

Replace `total_tokens` and `context_tokens_used` with five new columns:

| Column | Definition | Scope |
|---|---|---|
| `num_input_tokens` | `prompt_tokens` from the last API call | Last call only |
| `num_thinking_tokens` | `sum(reasoning_tokens)` across all calls | Accumulated |
| `num_output_tokens` | `sum(completion_tokens - reasoning_tokens)` across all calls | Accumulated |
| `total_response_tokens` | `num_thinking_tokens + num_output_tokens` | Derived |
| `total_tokens_used` | `num_input_tokens + total_response_tokens` | Derived |

### Why last-call only for `num_input_tokens`

The full conversation history is appended to the message list on every turn, so `prompt_tokens` on the last call represents the maximum context the model ever processed. This is the most useful number for understanding context window pressure. It is also the value already used internally for the context-window guard.

### `context_pct_used`

Kept. Recomputed as `num_input_tokens / context_window * 100` (same calculation, just renamed source).

### Context window guard (internal, not in CSV)

The guard needs `prompt_tokens + completion_tokens + reasoning_tokens` (total tokens on last call) to check whether the next call would exceed the context window. This is computed as a local variable `last_context_tokens` and is not written to CSV.

### Thinking tokens note

`reasoning_tokens` is sourced from `usage.completion_tokens_details.reasoning_tokens`. For non-thinking models this is 0. Thinking text is included in the assistant `content` field and re-sent in subsequent prompts (it counts against `prompt_tokens` in future turns), which is why `num_input_tokens` on the last call will be large for long thinking-model runs.

## Files to change

### `src/sudoku_bench/metrics.py`
- Remove `total_tokens`, `context_tokens_used` from `CSV_COLUMNS` and `PuzzleMetrics`
- Add `num_input_tokens`, `num_thinking_tokens`, `num_output_tokens`, `total_response_tokens`, `total_tokens_used`

### `src/sudoku_bench/runner.py`
- Replace `total_tokens` accumulator with two accumulators: `num_thinking_tokens`, `num_output_tokens`
- Replace `context_tokens_used` (overwritten each call) with `num_input_tokens` (same semantics, new name)
- Derive `total_response_tokens` and `total_tokens_used` before building `PuzzleMetrics`
- Keep `last_context_tokens = prompt_tokens + completion_tokens + reasoning` as a local for the context window guard

### Tests
- Update `tests/test_metrics.py` (if present) for new column names
- Update any runner tests that assert on old column names

## What does NOT change

- `context_pct_used` — kept, same formula
- All GPU/RAM/solve-outcome columns — untouched
- CSV row structure (one row per puzzle run) — untouched
- The context window guard logic — same condition, just renamed variable
