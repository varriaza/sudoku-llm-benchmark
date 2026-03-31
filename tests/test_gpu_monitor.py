"""Tests for the nvidia-smi GPU monitor."""
from __future__ import annotations

import psutil
from sudoku_bench.gpu_monitor import GPUMonitor


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_monitor(
    nvidia_responses: list[tuple[int, str, int] | None],
    nvidia_available: bool = True,
) -> GPUMonitor:
    """
    Build a GPUMonitor whose _sample is monkey-patched to inject fixture data.
    nvidia_responses: sequence of (used_mb, name, total_mb) tuples or None
                      (simulates nvidia-smi failure / unavailable GPU).
    nvidia_available: overrides the init-time nvidia-smi check result.
    """
    monitor = GPUMonitor(poll_interval=0.0)
    monitor._nvidia_available = nvidia_available
    monitor._nvidia_iter = iter(nvidia_responses)

    def fake_sample():
        if monitor._nvidia_available:
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


# ── VRAM collection ───────────────────────────────────────────────────────────

def test_vram_fields_populated_when_nvidia_available():
    monitor = _make_monitor([(4000, "RTX 4080", 16376)])
    monitor._kv_cache_samples = []
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor.stop()

    assert stats.gpu_name == "RTX 4080"
    assert stats.gpu_max_vram_mb == 16376
    assert stats.avg_vram_mb == 4000.0
    assert stats.max_vram_mb == 4000


def test_vram_fields_none_when_nvidia_unavailable():
    """When nvidia-smi is not available, all GPU fields must be None."""
    monitor = _make_monitor([], nvidia_available=False)
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor.stop()

    assert stats.gpu_name is None
    assert stats.gpu_max_vram_mb is None
    assert stats.avg_vram_mb is None
    assert stats.max_vram_mb is None
    assert stats.spilled_to_ram is None


def test_vram_fields_none_when_nvidia_query_always_fails():
    """nvidia-smi available but every query returns None → GPU fields None."""
    monitor = _make_monitor([None, None, None])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    monitor._sample()
    monitor._sample()
    stats = monitor.stop()

    assert stats.avg_vram_mb is None
    assert stats.max_vram_mb is None
    assert stats.gpu_name is None


def test_sys_ram_always_collected_even_without_nvidia():
    """psutil system RAM is collected regardless of nvidia-smi availability."""
    monitor = _make_monitor([], nvidia_available=False)
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._sample()
    stats = monitor.stop()

    assert stats.avg_sys_ram_mb is not None
    assert stats.max_sys_ram_mb is not None
    assert stats.avg_sys_ram_mb > 0


def test_sys_ram_always_collected_even_when_nvidia_query_fails():
    monitor = _make_monitor([None])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._sample()
    stats = monitor.stop()

    assert stats.avg_sys_ram_mb is not None
    assert stats.max_sys_ram_mb is not None


# ── spill detection ───────────────────────────────────────────────────────────

def test_spill_false_when_vram_within_capacity():
    monitor = _make_monitor([(4000, "RTX 4080", 16376)])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor.stop()

    assert stats.spilled_to_ram is False


def test_spill_true_when_vram_exceeds_capacity():
    monitor = _make_monitor([(20000, "RTX 4080", 16376)])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    stats = monitor.stop()

    assert stats.spilled_to_ram is True


def test_spill_none_when_no_vram_samples():
    monitor = _make_monitor([None])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._sample()
    stats = monitor.stop()

    assert stats.spilled_to_ram is None


# ── averages and max ──────────────────────────────────────────────────────────

def test_multiple_samples_avg_and_max():
    monitor = _make_monitor([
        (4000, "RTX 4080", 16376),
        (6000, "RTX 4080", 16376),
        (8000, "RTX 4080", 16376),
    ])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    monitor._sample()
    monitor._sample()
    stats = monitor.stop()

    assert stats.avg_vram_mb == round((4000 + 6000 + 8000) / 3, 1)
    assert stats.max_vram_mb == 8000


def test_gpu_name_set_from_first_sample_only():
    """gpu_name and gpu_max_vram_mb are only set on the first sample."""
    monitor = _make_monitor([
        (4000, "RTX 4080", 16376),
        (5000, "RTX 4090", 24576),  # second sample — should be ignored for name/capacity
    ])
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    monitor._gpu_name = None
    monitor._gpu_max_vram_mb = None
    monitor._sample()
    monitor._sample()
    stats = monitor.stop()

    assert stats.gpu_name == "RTX 4080"
    assert stats.gpu_max_vram_mb == 16376


def test_all_fields_none_when_no_samples_collected():
    """stop() with zero samples returns all-None GPU stats."""
    monitor = GPUMonitor(poll_interval=0.0)
    monitor._nvidia_available = False
    monitor._vram_samples = []
    monitor._sys_ram_samples = []
    stats = monitor.stop()

    assert stats.gpu_name is None
    assert stats.gpu_max_vram_mb is None
    assert stats.avg_vram_mb is None
    assert stats.max_vram_mb is None
    assert stats.spilled_to_ram is None
    assert stats.avg_sys_ram_mb is None
    assert stats.max_sys_ram_mb is None
