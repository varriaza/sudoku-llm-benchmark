from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from typing import Optional
import urllib.request
import json


@dataclass
class ModelInfo:
    name: str
    params: Optional[str]       # e.g. "70B"
    quant: Optional[str]        # e.g. "Q4_K_M"
    context_window: Optional[int]
    backend_type: str = "unknown"  # "llamacpp" | "vllm" | "unknown"


def _get_json(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _extract_params(name: str) -> Optional[str]:
    """Extract parameter count from model name, e.g. '70b' → '70B'."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    return m.group(0).upper() if m else None


def _extract_quant(name: str) -> Optional[str]:
    """Extract quantization tag from model name, e.g. 'q4_K_M'."""
    m = re.search(r"[qQ]\d+(?:[_\-][kK])?(?:[_\-][mMsSlL0-9]+)?", name)
    return m.group(0).upper() if m else None


def detect_model_info(api_base: str, name_override: Optional[str] = None) -> ModelInfo:
    """
    Auto-detect model metadata from a running llama-server or vLLM server.
    Detection order: llama.cpp (/props) → vLLM (/v1/models) → fallback.
    Falls back gracefully with warnings for any fields that can't be determined.
    """
    base = api_base.rstrip("/")
    # Strip trailing /v1 to get the server root (both llama-server and vLLM serve
    # /props and /metrics at the root, not under /v1)
    root = base[:-3] if base.endswith("/v1") else base

    # --- Try llama.cpp ---
    # llama-server exposes GET /props with {"n_ctx": <int>, ...}
    # This check must come before vLLM because llama-server also serves /v1/models.
    props = _get_json(f"{root}/props")
    if props is not None and "n_ctx" in props:
        context_window = int(props["n_ctx"])
        # Get model name from /v1/models if no override provided
        if name_override:
            model_name = name_override
        else:
            v1_models = _get_json(f"{base}/models")
            data = (v1_models or {}).get("data", [])
            model_name = data[0]["id"] if data else "unknown"
        return ModelInfo(
            name=model_name,
            params=_extract_params(model_name),
            quant=_extract_quant(model_name),
            context_window=context_window,
            backend_type="llamacpp",
        )

    # --- Try vLLM ---
    # List models: GET /v1/models
    v1_models = _get_json(f"{base}/models")
    if v1_models and "data" in v1_models:
        data = v1_models["data"]
        model_name = name_override or (data[0]["id"] if data else None)
        if model_name:
            warnings.warn(f"Could not determine context window for {model_name} — set it manually if needed.")
            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=None,
                backend_type="vllm",
            )

    # Fallback
    name = name_override or "unknown"
    warnings.warn(f"Could not auto-detect model info from {api_base}. Using name='{name}'.")
    return ModelInfo(
        name=name,
        params=_extract_params(name),
        quant=_extract_quant(name),
        context_window=None,
        backend_type="unknown",
    )
