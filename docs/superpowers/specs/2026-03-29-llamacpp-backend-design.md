# llama.cpp Backend Design

**Date:** 2026-03-29
**Branch:** feature/llamacpp-backend
**Status:** Approved

---

## Overview

Add a llama.cpp backend to `src/backends/llamacpp/` that enables the benchmark to run against a
`llama-server` process. The backend follows the same pattern as the existing vLLM backend: a
monitor class with `start()` / `stop() -> GPUStats` interface, and a detection branch in
`model_info.py`.

Ollama detection is removed from `model_info.py` as part of this change.

---

## New Files

```
src/backends/llamacpp/
    __init__.py                       (empty, mirrors src/backends/vllm/)
    monitor.py                        (LlamaCppMonitor)
tests/backends/
    test_llamacpp_monitor.py
```

---

## Modified Files

| File | Change |
|------|--------|
| `src/sudoku_bench/model_info.py` | Remove Ollama branch; add llama.cpp detection before vLLM branch |
| `src/sudoku_bench/runner.py` | Add `elif backend_type == "llamacpp"` monitor selection branch |

No config schema changes. The existing `serve.command` + `{model}` placeholder system is sufficient
for llama-server invocations.

---

## `LlamaCppMonitor` (`src/backends/llamacpp/monitor.py`)

A background-thread monitor that samples three sources on each tick:

### 1. Prometheus `/metrics` (llama.cpp-specific)

Polls `GET <api_base>/metrics` (strips `/v1` suffix, same as `VLLMMonitor`). Parses:

- `llamacpp:kv_cache_usage_ratio` — KV-cache fill fraction (0–1). A value `>= 1.0` signals that
  the cache is full and inference is spilling to CPU RAM → `spilled_to_ram = True`.

Falls back gracefully if `/metrics` is unreachable (llama-server requires `--metrics` flag) or the
metric is absent. When unavailable, `spilled_to_ram = None`.

> **NOTE:** `/metrics` parsing is duplicated from `VLLMMonitor`. Changes to the parsing logic in
> either class must be mirrored in the other.

### 2. nvidia-smi (raw VRAM)

Same query as `GPUMonitor`: `nvidia-smi --query-gpu=name,memory.used,memory.total`. Populates
`gpu_name`, `gpu_max_vram_mb`, `avg_vram_mb`, `max_vram_mb`. Falls back gracefully if nvidia-smi
is absent (all fields `None`).

> **NOTE:** nvidia-smi polling is duplicated from `GPUMonitor`. Changes to the polling logic in
> either class must be mirrored in the other.

### 3. psutil (system RAM)

Always sampled, same as all other monitors. Populates `avg_sys_ram_mb`, `max_sys_ram_mb`.

### Interface

`stop()` returns `GPUStats` — identical interface to `VLLMMonitor` and `GPUMonitor`. The runner
needs no structural changes beyond adding the `elif` branch.

---

## `model_info.py` Changes

### Remove

- Ollama detection branch (`GET /api/tags`, `POST /api/show`)
- `_get_json_post` helper

### Add

A llama.cpp detection branch inserted **before** the vLLM branch (llama-server also answers
`/v1/models`, so llama.cpp must be checked first to avoid misidentification):

1. `GET <base>/props` — llama-server returns JSON with `n_ctx` (context window).
2. If `/props` responds: set `context_window = n_ctx`, `backend_type = "llamacpp"`.
3. Model name: `name_override` if provided, otherwise from `GET /v1/models`.

### Detection order

```
llama.cpp (/props) → vLLM (/v1/models) → fallback
```

---

## `runner.py` Changes

In `_run_benchmark`, the monitor selection block gains one branch:

```python
if model_info.backend_type == "vllm":
    monitor = VLLMMonitor(...)
elif model_info.backend_type == "llamacpp":
    monitor = LlamaCppMonitor(...)
else:
    monitor = GPUMonitor(...)
```

---

## Testing

### `tests/backends/test_llamacpp_monitor.py`

Follows the pattern of `tests/backends/test_vllm_monitor.py`. No real server required —
`_sample` is monkey-patched to inject fixture data.

**Covered scenarios:**

| Test | What it verifies |
|------|-----------------|
| `_metrics_url` strips `/v1` | URL helper correctness |
| `_parse_gauge` for `llamacpp:kv_cache_usage_ratio` | Parses labelled and unlabelled Prometheus lines |
| `_parse_gauge` missing metric | Returns `None` |
| `_parse_gauge` NaN | Returns `None` |
| Spill = False (ratio < 1.0) | `spilled_to_ram is False` |
| Spill = True (ratio >= 1.0) | `spilled_to_ram is True` |
| No `/metrics` response | `spilled_to_ram is None`, VRAM fields from nvidia-smi |
| nvidia-smi data present | `avg_vram_mb`, `max_vram_mb`, `gpu_name` populated |
| nvidia-smi absent | VRAM fields `None` |
| Multiple samples | Averages computed correctly |

### `tests/test_model_info.py` (new or appended)

| Test | What it verifies |
|------|-----------------|
| `/props` returns `n_ctx` | `backend_type == "llamacpp"`, correct `context_window` |
| `/props` with `name_override` | Model name taken from override |
| `/props` absent, `/v1/models` present | Falls through to vLLM detection |
| Both absent | Fallback with warning |

---

## Example Config

```yaml
model:
  api_base: "http://localhost:8080/v1"
  name: "my-model.gguf"
  # context_window: 8192  # optional override; auto-detected from /props

serve:
  command: ["llama-server", "-m", "{model}", "--port", "8080", "--metrics"]
  startup_timeout: 60
```
