"""
vLLM metrics monitor.

Polls the vLLM Prometheus endpoint (GET /metrics) during a puzzle run and
returns a GPUStats-compatible result so the runner needs no structural changes.

What vLLM /metrics provides vs gpu_monitor:
  vllm:gpu_cache_usage_perc  — KV-cache GPU usage fraction (0–1)
  vllm:cpu_cache_usage_perc  — KV-cache CPU usage fraction; >0 means spilled
  avg/max VRAM in MB         — NOT available; fields left None
  system RAM                 — still collected via psutil (same as GPUMonitor)
"""
from __future__ import annotations

import re
import threading
import urllib.request
from typing import Optional

import psutil

from sudoku_bench.gpu_monitor import GPUStats


# ── Prometheus helpers ────────────────────────────────────────────────────────

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
      vllm:gpu_cache_usage_perc{model_name="..."} 0.42
      vllm:gpu_cache_usage_perc 0.42
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
    # Strip /v1 suffix — vLLM serves metrics at the root, not under /v1
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/metrics"


class VLLMMonitor:
    """
    Background poller for vLLM's Prometheus /metrics endpoint.

    Replaces GPUMonitor for vLLM backends. Returns GPUStats so the runner
    needs no changes. Fields that vLLM doesn't expose (raw VRAM in MB,
    gpu_name, gpu_max_vram_mb) are left None.
    """

    def __init__(self, api_base: str, poll_interval: float = 1.0):
        self._url = _metrics_url(api_base)
        self._poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._gpu_cache_samples: list[float] = []
        self._cpu_cache_samples: list[float] = []
        self._sys_ram_samples: list[int] = []

    def start(self) -> None:
        self._stop_event.clear()
        self._gpu_cache_samples = []
        self._cpu_cache_samples = []
        self._sys_ram_samples = []
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
        text = _fetch_metrics(self._url)
        if text:
            gpu = _parse_gauge(text, "vllm:gpu_cache_usage_perc")
            cpu = _parse_gauge(text, "vllm:cpu_cache_usage_perc")
            if gpu is not None:
                self._gpu_cache_samples.append(gpu)
            if cpu is not None:
                self._cpu_cache_samples.append(cpu)

        ram_mb = int(psutil.virtual_memory().used / 1024 / 1024)
        self._sys_ram_samples.append(ram_mb)

    def _compute_stats(self) -> GPUStats:
        max_cpu = max(self._cpu_cache_samples) if self._cpu_cache_samples else None
        spilled = (max_cpu > 0) if max_cpu is not None else None

        if self._sys_ram_samples:
            avg_sys = round(
                sum(self._sys_ram_samples) / len(self._sys_ram_samples), 1
            )
            max_sys = max(self._sys_ram_samples)
        else:
            avg_sys = None
            max_sys = None

        # avg_vram_mb and max_vram_mb are None — vLLM /metrics exposes
        # KV-cache as a fraction (0–1), not raw VRAM in MB, so we cannot
        # populate those columns without knowing total VRAM capacity.
        return GPUStats(
            gpu_name=None,
            gpu_max_vram_mb=None,
            avg_vram_mb=None,
            max_vram_mb=None,
            spilled_to_ram=spilled,
            avg_sys_ram_mb=avg_sys,
            max_sys_ram_mb=max_sys,
        )
