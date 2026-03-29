# Stub — full implementation in Task 3.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _metrics_url(base_url: str) -> str:
    raise NotImplementedError


def _parse_gauge(text: str, metric_name: str) -> Optional[float]:
    raise NotImplementedError


@dataclass
class MonitorStats:
    spilled_to_ram: Optional[bool]
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[int]
    gpu_name: Optional[str]
    gpu_max_vram_mb: Optional[int]
    avg_sys_ram_mb: Optional[float]
    max_sys_ram_mb: Optional[int]


class LlamaCppMonitor:
    def __init__(self, base_url: str, poll_interval: float = 1.0) -> None:
        raise NotImplementedError

    def _sample(self) -> None:
        raise NotImplementedError

    def _compute_stats(self) -> MonitorStats:
        raise NotImplementedError
