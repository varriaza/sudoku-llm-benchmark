"""Tests for model server lifecycle management."""
from __future__ import annotations
import subprocess
from unittest.mock import MagicMock, patch

from sudoku_bench.server import _api_ready, start_server, stop_server


# ── _api_ready ────────────────────────────────────────────────────────────────

def test_api_ready_true_on_success():
    with patch("urllib.request.urlopen"):
        assert _api_ready("http://localhost:8000/v1") is True


def test_api_ready_false_on_connection_error():
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert _api_ready("http://localhost:8000/v1") is False


def test_api_ready_strips_trailing_slash():
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        _api_ready("http://localhost:8000/v1/")
        url_called = mock_open.call_args[0][0]
        assert not url_called.endswith("//models")


# ── start_server ──────────────────────────────────────────────────────────────

def test_start_server_returns_proc_when_api_ready():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("sudoku_bench.server._api_ready", return_value=True):
        proc = start_server(["vllm", "serve", "model"], "http://localhost:8000/v1", startup_timeout=10)

    assert proc is mock_proc


def test_start_server_raises_runtime_error_if_proc_exits_early():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited with error

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("sudoku_bench.server._api_ready", return_value=False):
        try:
            start_server(["vllm", "serve", "model"], "http://localhost:8000/v1", startup_timeout=5)
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "exited early" in str(e)


def test_start_server_raises_timeout_if_never_ready():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running but never ready

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("sudoku_bench.server._api_ready", return_value=False), \
         patch("time.sleep"), \
         patch("time.time", side_effect=[0, 0, 999]):  # instant timeout
        try:
            start_server(["vllm", "serve", "model"], "http://localhost:8000/v1", startup_timeout=1)
            assert False, "Expected TimeoutError"
        except TimeoutError as e:
            assert "not become ready" in str(e)
        mock_proc.terminate.assert_called_once()


# ── stop_server ───────────────────────────────────────────────────────────────

def test_stop_server_terminates_proc():
    mock_proc = MagicMock()
    stop_server(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=10)


def test_stop_server_kills_if_wait_times_out():
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
    stop_server(mock_proc)
    mock_proc.kill.assert_called_once()


def test_stop_server_none_is_noop():
    stop_server(None)  # should not raise
