from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from typing import Optional
import urllib.request
import json


_QUANT_BPW: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5625,
    "Q5_K_M": 5.75,
    "Q5_K_S": 5.5,
    "Q5_1": 6.0,
    "Q5_0": 5.5,
    "Q4_K_M": 4.85,
    "Q4_K_S": 4.5,
    "Q4_K": 4.85,
    "Q4_1": 5.0,
    "Q4_0": 4.5,
    "Q3_K_L": 3.6,
    "Q3_K_M": 3.35,
    "Q3_K_S": 3.0,
    "Q3_K": 3.35,
    "Q2_K": 2.625,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "IQ3_S": 3.4375,
    "IQ3_M": 3.6625,
    "IQ3_XXS": 3.0625,
    "IQ2_XS": 2.3125,
    "IQ2_XXS": 2.0625,
    "IQ2_S": 2.5,
    "IQ1_S": 1.5625,
    "IQ1_M": 1.75,
}


def _quant_to_bpw(quant: Optional[str]) -> Optional[float]:
    if quant is None:
        return None
    return _QUANT_BPW.get(quant.upper())


@dataclass
class ModelInfo:
    name: str
    params: Optional[str]           # e.g. "70B"
    quant: Optional[str]            # e.g. "Q4_K_M"
    bits_per_weight: Optional[float]  # e.g. 4.85
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
        quant = _extract_quant(model_name)
        return ModelInfo(
            name=model_name,
            params=_extract_params(model_name),
            quant=quant,
            bits_per_weight=_quant_to_bpw(quant),
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
            quant = _extract_quant(model_name)
            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=quant,
                bits_per_weight=_quant_to_bpw(quant),
                context_window=None,
                backend_type="vllm",
            )

    # Fallback
    name = name_override or "unknown"
    warnings.warn(f"Could not auto-detect model info from {api_base}. Using name='{name}'.")
    quant = _extract_quant(name)
    return ModelInfo(
        name=name,
        params=_extract_params(name),
        quant=quant,
        bits_per_weight=_quant_to_bpw(quant),
        context_window=None,
        backend_type="unknown",
    )
