# llama.cpp Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LlamaCppMonitor` backend and llama.cpp detection in `model_info.py`, removing the Ollama detection branch.

**Architecture:** Mirror the vLLM backend structure: a new `src/backends/llamacpp/` package with a `LlamaCppMonitor` that polls Prometheus `/metrics` for KV-cache spill detection, nvidia-smi for raw VRAM, and psutil for system RAM. Detection order in `model_info.py` becomes: llama.cpp (`/props`) → vLLM (`/v1/models`) → fallback.

**Tech Stack:** Python stdlib (`subprocess`, `threading`, `urllib.request`, `re`), `psutil`, `pytest`

---

### Task 1: Create `src/backends/llamacpp/__init__.py`

**Files:**
- Create: `src/backends/llamacpp/__init__.py`
- Test: `tests/backends/test_llamacpp_monitor.py` (created in Task 2 — this task has no test)

- [ ] **Step 1: Create the empty `__init__.py`**

```python
```

(Empty file, mirrors `src/backends/vllm/__init__.py`.)

- [ ] **Step 2: Commit**

```bash
git add src/backends/llamacpp/__init__.py
git commit -m "feat: scaffold llamacpp backend package"
```

---

### Task 2: Write failing tests for `LlamaCppMonitor`

**Files:**
- Create: `tests/backends/test_llamacpp_monitor.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for the llama.cpp Prometheus metrics monitor."""
from __future__ import annotations

import pytest
from backends.llamacpp.monitor import LlamaCppMonitor, _metrics_url, _parse_gauge

# ── _metrics_url ──────────────────────────────────────────────────────────────

def test_metrics_url_strips_v1_suffix():
    assert _metrics_url("http://localhost:8080/v1") == "http://localhost:8080/metrics"


def test_metrics_url_no_v1_suffix():
    assert _metrics_url("http://localhost:8080") == "http://localhost:8080/metrics"


def test_metrics_url_strips_trailing_slash():
    assert _metrics_url("http://localhost:8080/v1/") == "http://localhost:8080/metrics"


# ── _parse_gauge ──────────────────────────────────────────────────────────────

SAMPLE_METRICS_WITH_LABEL = """\
# HELP llamacpp:kv_cache_usage_ratio KV-cache usage ratio (0-1).
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio{slot_id="0",model="my-model.gguf"} 0.45
"""

SAMPLE_METRICS_UNLABELLED = """\
llamacpp:kv_cache_usage_ratio 0.45
"""


def test_parse_gauge_labelled():
    assert _parse_gauge(SAMPLE_METRICS_WITH_LABEL, "llamacpp:kv_cache_usage_ratio") == 0.45


def test_parse_gauge_unlabelled():
    assert _parse_gauge(SAMPLE_METRICS_UNLABELLED, "llamacpp:kv_cache_usage_ratio") == 0.45


def test_parse_gauge_missing_metric():
    assert _parse_gauge(SAMPLE_METRICS_WITH_LABEL, "llamacpp:nonexistent") is None


def test_parse_gauge_nan_returns_none():
    text = 'llamacpp:kv_cache_usage_ratio{slot_id="0"} NaN\n'
    assert _parse_gauge(text, "llamacpp:kv_cache_usage_ratio") is None


# ── LlamaCppMonitor ───────────────────────────────────────────────────────────

def _make_monitor(
    prometheus_responses: list[str | None],
    nvidia_responses: list[tuple[int, str, int] | None],
) -> LlamaCppMonitor:
    """
    Build a LlamaCppMonitor whose _sample is monkey-patched to inject fixture data.
    prometheus_responses: sequence of raw metrics text or None (unreachable)
    nvidia_responses: sequence of (used_mb, name, total_mb) tuples or None (nvidia-smi absent)
    """
    monitor = LlamaCppMonitor("http://localhost:8080/v1", poll_interval=0.0)
    monitor._prometheus_iter = iter(prometheus_responses)
    monitor._nvidia_iter = iter(nvidia_responses)

    def fake_sample():
        text = next(monitor._prometheus_iter, None)
        if text:
            kv = _parse_gauge(text, "llamacpp:kv_cache_usage_ratio")
            if kv is not None:
                monitor._kv_cache_samples.append(kv)

        nv = next(monitor._nvidia_iter, None)
        if nv is not None:
            used_mb, name, total_mb = nv
            if monitor._gpu_name is None:
                monitor._gpu_name = name
                monitor._gpu_max_vram_mb = total_mb
            monitor._vram_samples.append(used_mb)

        import psutil
        ram_mb = int(psutil.virtual_memory().used / 1024 / 1024)
        monitor._sys_ram_samples.append(ram_mb)

    monitor._sample = fake_sample
    return monitor


def test_spill_false_when_kv_ratio_below_one():
    monitor = _make_monitor(
        prometheus_responses=[SAMPLE_METRICS_WITH_LABEL],  # kv = 0.45
        nvidia_responses=[(4000, "RTX 4090", 24576)],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is False


def test_spill_true_when_kv_ratio_at_one():
    text = SAMPLE_METRICS_WITH_LABEL.replace("0.45", "1.0")
    monitor = _make_monitor(
        prometheus_responses=[text],
        nvidia_responses=[(24576, "RTX 4090", 24576)],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is True


def test_spill_true_when_kv_ratio_above_one():
    text = SAMPLE_METRICS_WITH_LABEL.replace("0.45", "1.1")
    monitor = _make_monitor(
        prometheus_responses=[text],
        nvidia_responses=[None],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is True


def test_spill_none_when_no_prometheus_response():
    monitor = _make_monitor(
        prometheus_responses=[None],
        nvidia_responses=[(4000, "RTX 4090", 24576)],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is None
    # VRAM still populated from nvidia-smi
    assert stats.avg_vram_mb == 4000.0
    assert stats.max_vram_mb == 4000
    assert stats.gpu_name == "RTX 4090"
    assert stats.gpu_max_vram_mb == 24576


def test_nvidia_smi_data_populates_vram_fields():
    monitor = _make_monitor(
        prometheus_responses=[SAMPLE_METRICS_WITH_LABEL],
        nvidia_responses=[(8192, "RTX 3080", 10240)],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.avg_vram_mb == 8192.0
    assert stats.max_vram_mb == 8192
    assert stats.gpu_name == "RTX 3080"
    assert stats.gpu_max_vram_mb == 10240


def test_nvidia_smi_absent_gives_none_vram():
    monitor = _make_monitor(
        prometheus_responses=[SAMPLE_METRICS_WITH_LABEL],
        nvidia_responses=[None],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.avg_vram_mb is None
    assert stats.max_vram_mb is None
    assert stats.gpu_name is None
    assert stats.gpu_max_vram_mb is None


def test_multiple_samples_compute_averages():
    text_low = SAMPLE_METRICS_WITH_LABEL  # kv = 0.45
    text_high = SAMPLE_METRICS_WITH_LABEL.replace("0.45", "0.85")
    monitor = _make_monitor(
        prometheus_responses=[text_low, text_high],
        nvidia_responses=[(4000, "RTX 4090", 24576), (6000, "RTX 4090", 24576)],
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.avg_vram_mb == 5000.0   # (4000 + 6000) / 2
    assert stats.max_vram_mb == 6000
    assert stats.spilled_to_ram is False  # max kv = 0.85 < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest tests/backends/test_llamacpp_monitor.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'backends.llamacpp'`

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/backends/test_llamacpp_monitor.py
git commit -m "test: add failing tests for LlamaCppMonitor"
```

---

### Task 3: Implement `LlamaCppMonitor`

**Files:**
- Create: `src/backends/llamacpp/monitor.py`
- Test: `tests/backends/test_llamacpp_monitor.py` (created in Task 2)

- [ ] **Step 1: Create the monitor**

```python
"""
llama.cpp metrics monitor.

Polls three sources on each tick:
  1. GET <api_base>/metrics  — Prometheus text; reads llamacpp:kv_cache_usage_ratio
     (requires --metrics flag on llama-server). ratio >= 1.0 means KV cache is full
     and inference is spilling to CPU RAM.
  2. nvidia-smi              — raw VRAM in MB (gpu_name, avg/max_vram_mb, gpu_max_vram_mb)
  3. psutil                  — system RAM (avg/max_sys_ram_mb); always sampled.

NOTE: _metrics_url and _parse_gauge are duplicated from backends.vllm.monitor.
      Any changes to parsing logic must be mirrored in both files.

NOTE: nvidia-smi polling logic is duplicated from sudoku_bench.gpu_monitor.GPUMonitor.
      Any changes to the nvidia-smi query must be mirrored in both files.
"""
from __future__ import annotations

import re
import subprocess
import threading
import urllib.request
from typing import Optional

import psutil

from sudoku_bench.gpu_monitor import GPUStats


# ── Prometheus helpers (mirrors backends.vllm.monitor) ───────────────────────

def _fetch_metrics(url: str, timeout: int = 5) -> Optional[str]:
    """Fetch raw Prometheus text from the /metrics endpoint."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _parse_gauge(text: str, metric_name: str) -> Optional[float]:
    """
    Extract the first numeric value for a Prometheus gauge metric.

    Handles both labelled and unlabelled forms:
      llamacpp:kv_cache_usage_ratio{slot_id="0"} 0.45
      llamacpp:kv_cache_usage_ratio 0.45
    Returns None if the metric is absent or the value is NaN/Inf.
    """
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if value != value:  # NaN check
        return None
    return value


# ── Monitor ───────────────────────────────────────────────────────────────────

def _metrics_url(api_base: str) -> str:
    """Derive the /metrics URL from the OpenAI-compatible api_base."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/metrics"


class LlamaCppMonitor:
    """
    Background poller for llama-server's Prometheus /metrics endpoint plus nvidia-smi.

    Replaces GPUMonitor for llama.cpp backends. Returns GPUStats so the runner
    needs no structural changes.

    Spill detection: llamacpp:kv_cache_usage_ratio >= 1.0 means the KV cache is
    full and inference is spilling to CPU RAM.
    """

    def __init__(self, api_base: str, poll_interval: float = 1.0):
        self._url = _metrics_url(api_base)
        self._poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._kv_cache_samples: list[float] = []
        self._vram_samples: list[int] = []
        self._sys_ram_samples: list[int] = []
        self._gpu_name: Optional[str] = None
        self._gpu_max_vram_mb: Optional[int] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._kv_cache_samples = []
        self._vram_samples = []
        self._sys_ram_samples = []
        self._gpu_name = None
        self._gpu_max_vram_mb = None
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self._compute_stats()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(timeout=self._poll_interval)

    def _sample(self) -> None:
        # 1. Prometheus /metrics
        text = _fetch_metrics(self._url)
        if text:
            kv = _parse_gauge(text, "llamacpp:kv_cache_usage_ratio")
            if kv is not None:
                self._kv_cache_samples.append(kv)

        # 2. nvidia-smi
        nv = self._query_nvidia()
        if nv is not None:
            used_mb, name, total_mb = nv
            if self._gpu_name is None:
                self._gpu_name = name
                self._gpu_max_vram_mb = total_mb
            self._vram_samples.append(used_mb)

        # 3. psutil system RAM
        ram_mb = int(psutil.virtual_memory().used / 1024 / 1024)
        self._sys_ram_samples.append(ram_mb)

    def _query_nvidia(self) -> Optional[tuple[int, str, int]]:
        """Returns (used_vram_mb, gpu_name, max_vram_mb) or None on error."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            used_mb = int(parts[1])
            total_mb = int(parts[2])
            return used_mb, name, total_mb
        except Exception:
            return None

    def _compute_stats(self) -> GPUStats:
        # Spill: max KV-cache ratio >= 1.0
        if self._kv_cache_samples:
            max_kv = max(self._kv_cache_samples)
            spilled: Optional[bool] = max_kv >= 1.0
        else:
            spilled = None

        # VRAM
        if self._vram_samples:
            avg_vram: Optional[float] = round(
                sum(self._vram_samples) / len(self._vram_samples), 1
            )
            max_vram: Optional[int] = max(self._vram_samples)
        else:
            avg_vram = None
            max_vram = None

        # System RAM
        if self._sys_ram_samples:
            avg_sys: Optional[float] = round(
                sum(self._sys_ram_samples) / len(self._sys_ram_samples), 1
            )
            max_sys: Optional[int] = max(self._sys_ram_samples)
        else:
            avg_sys = None
            max_sys = None

        return GPUStats(
            gpu_name=self._gpu_name,
            gpu_max_vram_mb=self._gpu_max_vram_mb,
            avg_vram_mb=avg_vram,
            max_vram_mb=max_vram,
            spilled_to_ram=spilled,
            avg_sys_ram_mb=avg_sys,
            max_sys_ram_mb=max_sys,
        )
```

- [ ] **Step 2: Run monitor tests to verify they pass**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest tests/backends/test_llamacpp_monitor.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/backends/llamacpp/monitor.py
git commit -m "feat: implement LlamaCppMonitor with prometheus, nvidia-smi, and psutil"
```

---

### Task 4: Write failing tests for `model_info.py` changes

**Files:**
- Create: `tests/test_model_info.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for model_info.detect_model_info — llama.cpp detection and vLLM fallback."""
from __future__ import annotations

import warnings
from unittest.mock import patch, MagicMock

from sudoku_bench.model_info import detect_model_info


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_get_json(responses: dict):
    """
    Return a side_effect function for _get_json that maps URL → response.
    URLs not in responses return None.
    """
    def _get_json(url: str, timeout: int = 5):
        return responses.get(url)
    return _get_json


# ── llama.cpp detection ───────────────────────────────────────────────────────

def test_llamacpp_detected_via_props():
    responses = {
        "http://localhost:8080/props": {"n_ctx": 4096},
        "http://localhost:8080/v1/models": {"data": [{"id": "my-model.gguf"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        info = detect_model_info("http://localhost:8080/v1")

    assert info.backend_type == "llamacpp"
    assert info.context_window == 4096
    assert info.name == "my-model.gguf"


def test_llamacpp_name_override():
    responses = {
        "http://localhost:8080/props": {"n_ctx": 8192},
        "http://localhost:8080/v1/models": {"data": [{"id": "ignored-name.gguf"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        info = detect_model_info("http://localhost:8080/v1", name_override="custom-name.gguf")

    assert info.backend_type == "llamacpp"
    assert info.name == "custom-name.gguf"
    assert info.context_window == 8192


def test_llamacpp_props_absent_falls_through_to_vllm():
    responses = {
        "http://localhost:8000/v1/models": {"data": [{"id": "meta-llama/Llama-3-8B"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1")

    assert info.backend_type == "vllm"
    assert info.name == "meta-llama/Llama-3-8B"
    assert len(w) == 1
    assert "context window" in str(w[0].message).lower()


def test_both_absent_gives_fallback_with_warning():
    with patch("sudoku_bench.model_info._get_json", return_value=None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1")

    assert info.backend_type == "unknown"
    assert info.name == "unknown"
    assert len(w) == 1
    assert "Could not auto-detect" in str(w[0].message)


def test_both_absent_uses_name_override_in_fallback():
    with patch("sudoku_bench.model_info._get_json", return_value=None):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1", name_override="my-model")

    assert info.name == "my-model"
    assert info.backend_type == "unknown"
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest tests/test_model_info.py -v
```

Expected: `test_llamacpp_detected_via_props` FAILs (backend_type is "vllm" not "llamacpp") because llama.cpp detection doesn't exist yet. The vLLM and fallback tests may pass or fail depending on current code shape.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_model_info.py
git commit -m "test: add failing tests for llama.cpp model_info detection"
```

---

### Task 5: Update `model_info.py` — remove Ollama, add llama.cpp detection

**Files:**
- Modify: `src/sudoku_bench/model_info.py`

- [ ] **Step 1: Replace the file content**

```python
from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from typing import Optional
import urllib.request
import json


@dataclass
class ModelInfo:
    name: str
    params: Optional[str]       # e.g. "70B"
    quant: Optional[str]        # e.g. "Q4_K_M"
    context_window: Optional[int]
    backend_type: str = "unknown"  # "llamacpp" | "vllm" | "unknown"


def _get_json(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _extract_params(name: str) -> Optional[str]:
    """Extract parameter count from model name, e.g. '70b' → '70B'."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    return m.group(0).upper() if m else None


def _extract_quant(name: str) -> Optional[str]:
    """Extract quantization tag from model name, e.g. 'q4_K_M'."""
    m = re.search(r"[qQ]\d+(?:[_\-][kK])?(?:[_\-][mMsSlL0-9]+)?", name)
    return m.group(0).upper() if m else None


def detect_model_info(api_base: str, name_override: Optional[str] = None) -> ModelInfo:
    """
    Auto-detect model metadata from a running llama-server or vLLM server.
    Detection order: llama.cpp (/props) → vLLM (/v1/models) → fallback.
    Falls back gracefully with warnings for any fields that can't be determined.
    """
    base = api_base.rstrip("/")
    # Strip trailing /v1 to get the server root (both llama-server and vLLM serve
    # /props and /metrics at the root, not under /v1)
    root = base[:-3] if base.endswith("/v1") else base

    # --- Try llama.cpp ---
    # llama-server exposes GET /props with {"n_ctx": <int>, ...}
    # This check must come before vLLM because llama-server also serves /v1/models.
    props = _get_json(f"{root}/props")
    if props is not None and "n_ctx" in props:
        context_window = int(props["n_ctx"])
        # Get model name from /v1/models if no override provided
        if name_override:
            model_name = name_override
        else:
            v1_models = _get_json(f"{base}/models")
            data = (v1_models or {}).get("data", [])
            model_name = data[0]["id"] if data else "unknown"
        return ModelInfo(
            name=model_name,
            params=_extract_params(model_name),
            quant=_extract_quant(model_name),
            context_window=context_window,
            backend_type="llamacpp",
        )

    # --- Try vLLM ---
    # List models: GET /v1/models
    v1_models = _get_json(f"{base}/models")
    if v1_models and "data" in v1_models:
        data = v1_models["data"]
        model_name = name_override or (data[0]["id"] if data else None)
        if model_name:
            warnings.warn(f"Could not determine context window for {model_name} — set it manually if needed.")
            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=None,
                backend_type="vllm",
            )

    # Fallback
    name = name_override or "unknown"
    warnings.warn(f"Could not auto-detect model info from {api_base}. Using name='{name}'.")
    return ModelInfo(
        name=name,
        params=_extract_params(name),
        quant=_extract_quant(name),
        context_window=None,
        backend_type="unknown",
    )
```

- [ ] **Step 2: Run model_info tests**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest tests/test_model_info.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest --tb=short 2>&1 | tail -20
```

Expected: All existing tests still pass (runner tests, etc.).

- [ ] **Step 4: Commit**

```bash
git add src/sudoku_bench/model_info.py
git commit -m "feat: replace ollama detection with llama.cpp /props detection in model_info"
```

---

### Task 6: Update `runner.py` — add `llamacpp` monitor branch

**Files:**
- Modify: `src/sudoku_bench/runner.py`

- [ ] **Step 1: Add the import at the top of runner.py**

In `src/sudoku_bench/runner.py`, add the LlamaCppMonitor import alongside the VLLMMonitor import (line 13):

```python
from backends.vllm.monitor import VLLMMonitor
from backends.llamacpp.monitor import LlamaCppMonitor
```

- [ ] **Step 2: Replace the monitor selection block in `_run_benchmark`**

Find this block (around line 277–284):

```python
    if model_info.backend_type == "vllm":
        monitor = VLLMMonitor(
            api_base=config.model.api_base,
            poll_interval=config.benchmark.gpu_poll_interval,
        )
        print("  Monitor: vLLM /metrics")
    else:
        monitor = GPUMonitor(poll_interval=config.benchmark.gpu_poll_interval)
        print("  Monitor: nvidia-smi")
```

Replace it with:

```python
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
```

- [ ] **Step 3: Run the full test suite**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/sudoku_bench/runner.py
git commit -m "feat: add llamacpp monitor branch to runner"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && python -m pytest -v 2>&1 | tail -40
```

Expected: All tests PASS, including:
- `tests/backends/test_llamacpp_monitor.py` — all 10 monitor tests
- `tests/test_model_info.py` — all 5 detection tests
- All pre-existing tests

- [ ] **Step 2: Verify package structure**

```bash
find src/backends/llamacpp -type f | sort
```

Expected:
```
src/backends/llamacpp/__init__.py
src/backends/llamacpp/monitor.py
```

- [ ] **Step 3: Commit final state (if any uncommitted changes)**

```bash
cd /home/kiwi/repos/sudoku-llm-benchmark && git status
```

If clean: done. If not: commit any outstanding changes.
