"""Tests for the llama.cpp Prometheus metrics monitor."""
from __future__ import annotations

import psutil
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
