"""Tests for model_info.detect_model_info — llama.cpp detection and vLLM fallback."""
from __future__ import annotations

import warnings
from unittest.mock import patch

from sudoku_bench.model_info import detect_model_info


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_get_json(responses: dict):
    """
    Return a side_effect function for _get_json that maps URL → response.
    URLs not in responses return None.
    """
    def _get_json(url: str, timeout: int = 5):
        return responses.get(url)
    return _get_json


# ── llama.cpp detection ───────────────────────────────────────────────────────

def test_llamacpp_detected_via_props():
    responses = {
        "http://localhost:8080/props": {"n_ctx": 4096},
        "http://localhost:8080/v1/models": {"data": [{"id": "my-model.gguf"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        info = detect_model_info("http://localhost:8080/v1")

    assert info.backend_type == "llamacpp"
    assert info.context_window == 4096
    assert info.name == "my-model.gguf"


def test_llamacpp_name_override():
    responses = {
        "http://localhost:8080/props": {"n_ctx": 8192},
        "http://localhost:8080/v1/models": {"data": [{"id": "ignored-name.gguf"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        info = detect_model_info("http://localhost:8080/v1", name_override="custom-name.gguf")

    assert info.backend_type == "llamacpp"
    assert info.name == "custom-name.gguf"
    assert info.context_window == 8192


def test_llamacpp_props_absent_falls_through_to_vllm():
    responses = {
        "http://localhost:8000/v1/models": {"data": [{"id": "meta-llama/Llama-3-8B"}]},
    }
    with patch("sudoku_bench.model_info._get_json", side_effect=_mock_get_json(responses)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1")

    assert info.backend_type == "vllm"
    assert info.name == "meta-llama/Llama-3-8B"
    assert len(w) == 1
    assert "context window" in str(w[0].message).lower()


def test_both_absent_gives_fallback_with_warning():
    with patch("sudoku_bench.model_info._get_json", return_value=None):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1")

    assert info.backend_type == "unknown"
    assert info.name == "unknown"
    assert len(w) == 1
    assert "Could not auto-detect" in str(w[0].message)


def test_both_absent_uses_name_override_in_fallback():
    with patch("sudoku_bench.model_info._get_json", return_value=None):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            info = detect_model_info("http://localhost:8000/v1", name_override="my-model")

    assert info.name == "my-model"
    assert info.backend_type == "unknown"
