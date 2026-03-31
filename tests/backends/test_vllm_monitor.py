"""Tests for the vLLM Prometheus metrics monitor."""
from __future__ import annotations

from backends.vllm.monitor import VLLMMonitor, _metrics_url, _parse_gauge, _speed_stats

# ── _metrics_url ──────────────────────────────────────────────────────────────

def test_metrics_url_strips_v1_suffix():
    assert _metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"


def test_metrics_url_no_v1_suffix():
    assert _metrics_url("http://localhost:8000") == "http://localhost:8000/metrics"


def test_metrics_url_strips_trailing_slash():
    assert _metrics_url("http://localhost:8000/v1/") == "http://localhost:8000/metrics"


def test_metrics_url_does_not_strip_v1_mid_path():
    # A host that contains /v1 elsewhere should not be corrupted
    assert _metrics_url("http://v1.example.com/v1") == "http://v1.example.com/metrics"


# ── _parse_gauge ──────────────────────────────────────────────────────────────

SAMPLE_METRICS = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.42
# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
"""


def test_parse_gauge_labelled():
    assert _parse_gauge(SAMPLE_METRICS, "vllm:gpu_cache_usage_perc") == 0.42


def test_parse_gauge_zero():
    assert _parse_gauge(SAMPLE_METRICS, "vllm:cpu_cache_usage_perc") == 0.0


def test_parse_gauge_integer_value():
    assert _parse_gauge(SAMPLE_METRICS, "vllm:num_requests_running") == 1.0


def test_parse_gauge_missing_metric():
    assert _parse_gauge(SAMPLE_METRICS, "vllm:nonexistent_metric") is None


def test_parse_gauge_unlabelled():
    text = "some_metric 3.14\n"
    assert _parse_gauge(text, "some_metric") == 3.14


def test_parse_gauge_nan_returns_none():
    text = "vllm:gpu_cache_usage_perc{model_name=\"x\"} NaN\n"
    assert _parse_gauge(text, "vllm:gpu_cache_usage_perc") is None


# ── VLLMMonitor ───────────────────────────────────────────────────────────────

def _make_monitor(responses: list[str | None]) -> VLLMMonitor:
    """Build a VLLMMonitor whose _fetch_metrics returns each response in turn."""
    monitor = VLLMMonitor("http://localhost:8000/v1", poll_interval=0.0)
    monitor._responses = iter(responses)

    def fake_sample():
        text = next(monitor._responses, None)
        if text:
            gpu = _parse_gauge(text, "vllm:gpu_cache_usage_perc")
            cpu = _parse_gauge(text, "vllm:cpu_cache_usage_perc")
            if gpu is not None:
                monitor._gpu_cache_samples.append(gpu)
            if cpu is not None:
                monitor._cpu_cache_samples.append(cpu)
        monitor._sys_ram_samples.append(1024)  # fixed value for testing

    monitor._sample = fake_sample
    return monitor


def test_stop_returns_gpu_stats_with_correct_spill_false():
    monitor = _make_monitor([SAMPLE_METRICS])
    monitor.start()
    monitor._stop_event.set()
    monitor._thread.join()
    stats = monitor.stop()

    assert stats.spilled_to_ram is False   # cpu_cache == 0.0
    assert stats.avg_sys_ram_mb == 1024.0
    assert stats.max_sys_ram_mb == 1024


def test_stop_spilled_when_cpu_cache_nonzero():
    spilled_metrics = SAMPLE_METRICS.replace(
        "cpu_cache_usage_perc{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 0.0",
        "cpu_cache_usage_perc{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 0.15",
    )
    monitor = _make_monitor([spilled_metrics])
    monitor.start()
    monitor._stop_event.set()
    monitor._thread.join()
    stats = monitor.stop()

    assert stats.spilled_to_ram is True


def test_stop_all_none_when_no_samples():
    monitor = _make_monitor([None])  # fetch always fails
    monitor.start()
    monitor._stop_event.set()
    monitor._thread.join()
    stats = monitor.stop()

    assert stats.spilled_to_ram is None
    assert stats.avg_vram_mb is None
    assert stats.max_vram_mb is None
    assert stats.gpu_name is None
    assert stats.gpu_max_vram_mb is None
    # sys RAM is always sampled via psutil regardless of fetch success
    assert stats.avg_sys_ram_mb == 1024.0


def test_stop_averages_multiple_samples():
    metrics_low = SAMPLE_METRICS  # gpu_cache = 0.42
    metrics_high = SAMPLE_METRICS.replace(
        "gpu_cache_usage_perc{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 0.42",
        "gpu_cache_usage_perc{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 0.58",
    )
    monitor = _make_monitor([metrics_low, metrics_high])
    # Manually drive two samples
    monitor._gpu_cache_samples = []
    monitor._cpu_cache_samples = []
    monitor._sys_ram_samples = []
    monitor._sample()
    monitor._sample()
    stats = monitor._compute_stats()

    assert stats.avg_vram_mb is None  # raw VRAM not available from /metrics


# ── _speed_stats ──────────────────────────────────────────────────────────────

def test_speed_stats_empty_returns_none_triple():
    assert _speed_stats([]) == (None, None, None)


def test_speed_stats_single_value():
    assert _speed_stats([42.0]) == (42.0, 42.0, 42.0)


def test_speed_stats_multiple_values():
    avg, median, maximum = _speed_stats([10.0, 20.0, 30.0])
    assert avg == 20.0
    assert median == 20.0
    assert maximum == 30.0


def test_speed_stats_rounds_to_two_dp():
    avg, median, maximum = _speed_stats([10.0, 20.0, 21.0])
    assert avg == round((10 + 20 + 21) / 3, 2)


# ── VLLMMonitor speed sampling ────────────────────────────────────────────────

SAMPLE_METRICS_WITH_SPEED = SAMPLE_METRICS + """\
# HELP vllm:avg_generation_throughput_toks_per_s Rolling avg generation throughput.
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s{model_name="meta-llama/Llama-3.1-8B-Instruct"} 45.5
# HELP vllm:avg_prompt_throughput_toks_per_s Rolling avg prompt throughput.
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1200.0
"""


def test_speed_metrics_parsed_from_prometheus_text():
    """_parse_gauge correctly extracts throughput metrics from vLLM metrics text."""
    assert _parse_gauge(SAMPLE_METRICS_WITH_SPEED, "vllm:avg_generation_throughput_toks_per_s") == 45.5
    assert _parse_gauge(SAMPLE_METRICS_WITH_SPEED, "vllm:avg_prompt_throughput_toks_per_s") == 1200.0


def test_speed_fields_populated_in_compute_stats():
    monitor = VLLMMonitor("http://localhost:8000/v1", poll_interval=0.0)
    monitor._gen_toks_samples = [45.5]
    monitor._prompt_toks_samples = [1200.0]
    monitor._gpu_cache_samples = []
    monitor._cpu_cache_samples = []
    monitor._sys_ram_samples = [1024]
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec == 45.5
    assert stats.median_gen_toks_per_sec == 45.5
    assert stats.max_gen_toks_per_sec == 45.5
    assert stats.avg_prompt_toks_per_sec == 1200.0
    assert stats.median_prompt_toks_per_sec == 1200.0
    assert stats.max_prompt_toks_per_sec == 1200.0


def test_speed_fields_none_when_no_samples():
    monitor = VLLMMonitor("http://localhost:8000/v1", poll_interval=0.0)
    monitor._gen_toks_samples = []
    monitor._prompt_toks_samples = []
    monitor._gpu_cache_samples = []
    monitor._cpu_cache_samples = []
    monitor._sys_ram_samples = [1024]
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec is None
    assert stats.median_gen_toks_per_sec is None
    assert stats.max_gen_toks_per_sec is None
    assert stats.avg_prompt_toks_per_sec is None
    assert stats.median_prompt_toks_per_sec is None
    assert stats.max_prompt_toks_per_sec is None


def test_speed_stats_avg_median_max_across_samples():
    monitor = VLLMMonitor("http://localhost:8000/v1", poll_interval=0.0)
    monitor._gen_toks_samples = [30.0, 40.0, 50.0]
    monitor._prompt_toks_samples = [1000.0, 1100.0, 1200.0]
    monitor._gpu_cache_samples = []
    monitor._cpu_cache_samples = []
    monitor._sys_ram_samples = [1024]
    stats = monitor._compute_stats()

    assert stats.avg_gen_toks_per_sec == 40.0
    assert stats.median_gen_toks_per_sec == 40.0
    assert stats.max_gen_toks_per_sec == 50.0
    assert stats.avg_prompt_toks_per_sec == 1100.0
    assert stats.median_prompt_toks_per_sec == 1100.0
    assert stats.max_prompt_toks_per_sec == 1200.0
