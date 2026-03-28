from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class GPUStats:
    gpu_name: Optional[str]
    gpu_max_vram_mb: Optional[int]
    avg_vram_mb: Optional[float]
    max_vram_mb: Optional[int]
    spilled_to_ram: Optional[bool]
    avg_sys_ram_mb: Optional[float]
    max_sys_ram_mb: Optional[int]


class GPUMonitor:
    """
    Background thread that polls nvidia-smi at a configurable interval.
    Call start() before inference and stop() after to collect stats.
    If nvidia-smi is not available, all GPU fields are None.
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._vram_samples: list[int] = []
        self._sys_ram_samples: list[int] = []
        self._gpu_name: Optional[str] = None
        self._gpu_max_vram_mb: Optional[int] = None
        self._nvidia_available = self._check_nvidia()

    def _check_nvidia(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi"], capture_output=True, check=True, timeout=5
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

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

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._nvidia_available:
                data = self._query_nvidia()
                if data:
                    used_mb, name, total_mb = data
                    if self._gpu_name is None:
                        self._gpu_name = name
                        self._gpu_max_vram_mb = total_mb
                    self._vram_samples.append(used_mb)

            # Always sample system RAM
            ram_mb = int(psutil.virtual_memory().used / 1024 / 1024)
            self._sys_ram_samples.append(ram_mb)

            self._stop_event.wait(timeout=self.poll_interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._vram_samples = []
        self._sys_ram_samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUStats:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

        if not self._vram_samples:
            avg_vram = None
            max_vram = None
            spilled = None
        else:
            avg_vram = sum(self._vram_samples) / len(self._vram_samples)
            max_vram = max(self._vram_samples)
            spilled = (
                max_vram > self._gpu_max_vram_mb
                if self._gpu_max_vram_mb
                else None
            )

        if not self._sys_ram_samples:
            avg_sys = None
            max_sys = None
        else:
            avg_sys = sum(self._sys_ram_samples) / len(self._sys_ram_samples)
            max_sys = max(self._sys_ram_samples)

        return GPUStats(
            gpu_name=self._gpu_name,
            gpu_max_vram_mb=self._gpu_max_vram_mb,
            avg_vram_mb=round(avg_vram, 1) if avg_vram is not None else None,
            max_vram_mb=max_vram,
            spilled_to_ram=spilled,
            avg_sys_ram_mb=round(avg_sys, 1) if avg_sys is not None else None,
            max_sys_ram_mb=max_sys,
        )
