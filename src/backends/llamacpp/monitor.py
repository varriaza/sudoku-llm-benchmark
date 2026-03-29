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
    Returns None if the metric is absent or the value cannot be parsed as a float.
    Note: the regex only matches numeric characters, so NaN/Inf literal strings
    are rejected at the regex stage rather than the float-conversion stage.
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
            line = result.stdout.strip().splitlines()[0]  # single-GPU only; multi-GPU machines use GPU 0
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
