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


def _get_json_post(url: str, body: dict, timeout: int = 5) -> Optional[dict]:
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def detect_model_info(api_base: str, name_override: Optional[str] = None) -> ModelInfo:
    """
    Auto-detect model metadata from a running Ollama or vLLM server.
    Falls back gracefully with warnings for any fields that can't be determined.
    """
    base = api_base.rstrip("/")
    # Strip trailing /v1 for Ollama endpoints (safe for URLs that contain /v1 elsewhere)
    ollama_base = base[:-3] if base.endswith("/v1") else base

    # --- Try Ollama ---
    # List models: GET /api/tags
    tags = _get_json(f"{ollama_base}/api/tags")
    if tags and "models" in tags:
        models = tags["models"]
        model_name = name_override or (models[0]["name"] if models else None)
        if model_name:
            show = _get_json_post(
                f"{ollama_base}/api/show",
                {"name": model_name},
            )
            context_window = None
            if show:
                # context_window may be in modelinfo or parameters
                params_text = show.get("parameters", "")
                m = re.search(r"num_ctx\s+(\d+)", str(params_text))
                if m:
                    context_window = int(m.group(1))
                if context_window is None:
                    mi = show.get("modelinfo", {})
                    for key in mi:
                        if "context" in key.lower():
                            context_window = int(mi[key])
                            break

            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=context_window,
            )

    # --- Try vLLM ---
    # List models: GET /v1/models
    v1_models = _get_json(f"{base}/models")
    if v1_models and "data" in v1_models:
        data = v1_models["data"]
        model_name = name_override or (data[0]["id"] if data else None)
        if model_name:
            # vLLM doesn't expose context window via API; parse from name
            warnings.warn(f"Could not determine context window for {model_name} — set it manually if needed.")
            return ModelInfo(
                name=model_name,
                params=_extract_params(model_name),
                quant=_extract_quant(model_name),
                context_window=None,
            )

    # Fallback
    name = name_override or "unknown"
    warnings.warn(f"Could not auto-detect model info from {api_base}. Using name='{name}'.")
    return ModelInfo(
        name=name,
        params=_extract_params(name),
        quant=_extract_quant(name),
        context_window=None,
    )
