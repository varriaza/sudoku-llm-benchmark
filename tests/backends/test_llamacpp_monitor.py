"""Tests for the llama.cpp Prometheus metrics monitor."""
from __future__ import annotations

from unittest.mock import patch

import psutil
import backends.llamacpp.monitor as llamacpp_module
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

    # VRAM fallback: 4000 < 24576, so not spilled
    assert stats.spilled_to_ram is False
    # VRAM still populated from nvidia-smi
    assert stats.avg_vram_mb == 4000.0
    assert stats.max_vram_mb == 4000
    assert stats.gpu_name == "RTX 4090"
    assert stats.gpu_max_vram_mb == 24576


def test_spill_false_via_vram_fallback_when_kv_metric_absent():
    """When /metrics has no kv_cache ratio, fall back to VRAM < capacity → False."""
    monitor = _make_monitor(
        prometheus_responses=[None],      # no kv_cache metric available
        nvidia_responses=[(4000, "RTX 4090", 24576)],  # well within capacity
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is False


def test_spill_true_via_vram_fallback_when_kv_metric_absent():
    """When /metrics has no kv_cache ratio, fall back to VRAM > capacity → True."""
    monitor = _make_monitor(
        prometheus_responses=[None],
        nvidia_responses=[(25000, "RTX 4090", 24576)],  # exceeds capacity
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is True


def test_spill_none_when_neither_kv_nor_vram_available():
    """No kv_cache metric and no VRAM data → spilled_to_ram remains None."""
    monitor = _make_monitor(
        prometheus_responses=[None],
        nvidia_responses=[None],          # nvidia-smi also absent
    )
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.spilled_to_ram is None


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


# ── speed sampling ────────────────────────────────────────────────────────────

# llama-server exposes cumulative counters, not instant gauges, for token throughput.
SAMPLE_METRICS_COUNTERS_T1 = """\
llamacpp:kv_cache_usage_ratio{slot_id="0"} 0.45
llamacpp:tokens_predicted_total 100.0
llamacpp:tokens_predicted_seconds_total 2.0
llamacpp:prompt_tokens_total 500.0
llamacpp:prompt_seconds_total 0.5
"""

SAMPLE_METRICS_COUNTERS_T2 = """\
llamacpp:kv_cache_usage_ratio{slot_id="0"} 0.50
llamacpp:tokens_predicted_total 200.0
llamacpp:tokens_predicted_seconds_total 4.0
llamacpp:prompt_tokens_total 1000.0
llamacpp:prompt_seconds_total 1.5
"""


def test_counter_metrics_parsed_from_prometheus_text():
    """_parse_gauge correctly extracts counter metrics from llama.cpp metrics text."""
    assert _parse_gauge(SAMPLE_METRICS_COUNTERS_T1, "llamacpp:tokens_predicted_total") == 100.0
    assert _parse_gauge(SAMPLE_METRICS_COUNTERS_T1, "llamacpp:tokens_predicted_seconds_total") == 2.0
    assert _parse_gauge(SAMPLE_METRICS_COUNTERS_T1, "llamacpp:prompt_tokens_total") == 500.0
    assert _parse_gauge(SAMPLE_METRICS_COUNTERS_T1, "llamacpp:prompt_seconds_total") == 0.5


def _make_speed_monitor() -> LlamaCppMonitor:
    """Fresh monitor with all state cleared, _query_nvidia disabled."""
    monitor = LlamaCppMonitor("http://localhost:8080/v1", poll_interval=0.0)
    monitor._gen_toks_samples = []
    monitor._prompt_toks_samples = []
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._prev_gen_total = None
    monitor._prev_gen_secs = None
    monitor._prev_prompt_total = None
    monitor._prev_prompt_secs = None
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._query_nvidia = lambda: None
    return monitor


def test_speed_fields_populated_in_compute_stats():
    monitor = LlamaCppMonitor("http://localhost:8080/v1", poll_interval=0.0)
    monitor._gen_toks_samples = [38.5]
    monitor._prompt_toks_samples = [950.0]
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec == 38.5
    assert stats.median_gen_toks_per_sec == 38.5
    assert stats.max_gen_toks_per_sec == 38.5
    assert stats.avg_prompt_toks_per_sec == 950.0
    assert stats.median_prompt_toks_per_sec == 950.0
    assert stats.max_prompt_toks_per_sec == 950.0


def test_gen_speed_computed_from_counter_deltas():
    """Rate = delta_tokens / delta_seconds across two consecutive polls."""
    texts = iter([SAMPLE_METRICS_COUNTERS_T1, SAMPLE_METRICS_COUNTERS_T2])
    monitor = _make_speed_monitor()

    with patch.object(llamacpp_module, "_fetch_metrics", side_effect=lambda *a, **kw: next(texts, None)):
        monitor._sample()  # first poll: primes prev counters, no rate yet
        monitor._sample()  # second poll: delta = 100 tok / 2 sec = 50 t/s gen, 500 tok / 1 sec = 500 t/s prompt

    assert monitor._gen_toks_samples == [50.0]      # (200-100) / (4.0-2.0)
    assert monitor._prompt_toks_samples == [500.0]  # (1000-500) / (1.5-0.5)


def test_no_rate_on_first_poll():
    """First poll only primes state; no sample is added until a second poll."""
    texts = iter([SAMPLE_METRICS_COUNTERS_T1])
    monitor = _make_speed_monitor()

    with patch.object(llamacpp_module, "_fetch_metrics", side_effect=lambda *a, **kw: next(texts, None)):
        monitor._sample()

    assert monitor._gen_toks_samples == []
    assert monitor._prompt_toks_samples == []


def test_idle_poll_produces_no_sample():
    """When counters don't advance (server idle), no rate sample is added."""
    # Both polls report the same counter values
    texts = iter([SAMPLE_METRICS_COUNTERS_T1, SAMPLE_METRICS_COUNTERS_T1])
    monitor = _make_speed_monitor()

    with patch.object(llamacpp_module, "_fetch_metrics", side_effect=lambda *a, **kw: next(texts, None)):
        monitor._sample()
        monitor._sample()

    assert monitor._gen_toks_samples == []
    assert monitor._prompt_toks_samples == []


def test_stop_takes_final_sample_capturing_completion_delta():
    """
    llama-server updates token counters only at request completion, so all in-flight
    polls see unchanged counters. stop() takes a final sample after the thread exits
    to capture the completion delta.
    """
    # Simulate: poll loop sees counter=0 (generation in progress),
    # then stop() is called and the final sample sees counters at their completed values.
    idle_text = """\
llamacpp:tokens_predicted_total 0.0
llamacpp:tokens_predicted_seconds_total 0.0
llamacpp:prompt_tokens_total 500.0
llamacpp:prompt_seconds_total 0.5
"""
    completed_text = """\
llamacpp:tokens_predicted_total 32500.0
llamacpp:tokens_predicted_seconds_total 500.0
llamacpp:prompt_tokens_total 500.0
llamacpp:prompt_seconds_total 0.5
"""
    call_count = [0]

    def fake_fetch(*a, **kw):
        call_count[0] += 1
        # First N calls (poll loop) return idle; last call (stop's final sample) returns completed
        return idle_text if call_count[0] < 3 else completed_text

    monitor = _make_speed_monitor()
    monitor._stop_event.set()  # prevent actual thread from running

    with patch.object(llamacpp_module, "_fetch_metrics", side_effect=fake_fetch):
        # Simulate two in-flight polls (counter stuck at 0)
        monitor._sample()  # primes prev state with gen counter=0
        monitor._sample()  # sees same gen counter; no sample added
        # stop() takes final sample, captures the completion jump
        stats = monitor.stop()

    # 32500 tokens in 500 seconds = 65 t/s
    assert monitor._gen_toks_samples == [65.0]
    assert stats.avg_gen_toks_per_sec == 65.0
    assert stats.median_gen_toks_per_sec == 65.0
    assert stats.max_gen_toks_per_sec == 65.0


def test_speed_fields_none_when_no_samples():
    monitor = LlamaCppMonitor("http://localhost:8080/v1", poll_interval=0.0)
    monitor._gen_toks_samples = []
    monitor._prompt_toks_samples = []
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec is None
    assert stats.median_gen_toks_per_sec is None
    assert stats.max_gen_toks_per_sec is None
    assert stats.avg_prompt_toks_per_sec is None
    assert stats.median_prompt_toks_per_sec is None
    assert stats.max_prompt_toks_per_sec is None


def test_speed_stats_avg_median_max_across_samples():
    monitor = LlamaCppMonitor("http://localhost:8080/v1", poll_interval=0.0)
    monitor._gen_toks_samples = [20.0, 40.0, 60.0]
    monitor._prompt_toks_samples = [800.0, 1000.0, 1200.0]
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec == 40.0
    assert stats.median_gen_toks_per_sec == 40.0
    assert stats.max_gen_toks_per_sec == 60.0
    assert stats.avg_prompt_toks_per_sec == 1000.0
    assert stats.median_prompt_toks_per_sec == 1000.0
    assert stats.max_prompt_toks_per_sec == 1200.0
