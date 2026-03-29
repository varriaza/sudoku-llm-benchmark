"""
Model server lifecycle management.

Starts an external model server (e.g. vLLM) as a subprocess, waits for
its API to become ready, and provides a clean shutdown path.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
from typing import Optional


def _api_ready(api_base: str, timeout: int = 2) -> bool:
    """Return True if the OpenAI-compatible /models endpoint responds."""
    url = f"{api_base.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def start_server(
    command: list[str],
    api_base: str,
    startup_timeout: int = 120,
) -> subprocess.Popen:
    """
    Launch a model server and block until its API is ready.

    Args:
        command: Full command to run, e.g. ["vllm", "serve", "meta-llama/..."]
        api_base: OpenAI-compatible base URL to poll for readiness.
        startup_timeout: Seconds to wait before giving up.

    Returns:
        The running Popen process. Caller is responsible for terminating it.

    Raises:
        RuntimeError: If the process exits before the API becomes ready.
        TimeoutError: If the API is not ready within startup_timeout seconds.
    """
    proc = subprocess.Popen(command)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with return code {proc.returncode}.\n"
                f"Command: {' '.join(command)}"
            )
        if _api_ready(api_base):
            return proc
        time.sleep(2)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise TimeoutError(
        f"Server did not become ready within {startup_timeout}s.\n"
        f"Command: {' '.join(command)}"
    )


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    """Terminate a server process started by start_server."""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
