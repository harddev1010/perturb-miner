from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _normalize(text: str) -> str:
    return text.strip().lower().replace("_", " ")


class VerifyRequest(BaseModel):
    prediction: str = Field(..., min_length=1)
    target_label: str = Field(..., min_length=1)
    llm_model: str | None = None


class VerifyResponse(BaseModel):
    is_match: bool
    reason: str
    method: str


@dataclass
class Metrics:
    started_at: float
    total_requests: int = 0
    llm_requests: int = 0
    llm_failures: int = 0


app = FastAPI(title="Perturb LLM Endpoint", version="0.1.0")
_metrics = Metrics(started_at=time.time())
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("PERTURB_LLM_ENDPOINT_MODEL", os.getenv("PERTURB_LLM_VERIFY_MODEL", "qwen2.5:1.5b-instruct"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "8"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.1"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "20"))


def _resolve_model_name(raw: str) -> str:
    value = raw.strip()
    lowered = value.lower()
    aliases = {
        "qwen2.5-1.5b-instruct": "qwen2.5:1.5b-instruct",
        "qwen2.5:1.5b-instruct": "qwen2.5:1.5b-instruct",
    }
    return aliases.get(lowered, value)


def _prompt(prediction: str, target: str) -> str:
    return (
        "You are a strict taxonomy membership judge for image classification labels.\n"
        "Task: decide whether prediction BELONGS TO target_label.\n"
        "Interpretation: prediction is usually specific; target_label is usually broader.\n"
        "CRITICAL POLICY:\n"
        "- If prediction is a subtype/breed/species/member/instance inside target_label, set is_match=true.\n"
        "- Do NOT reject because prediction is more specific than target_label.\n"
        "- Specific->general membership MUST be true.\n"
        "- Only set is_match=false when prediction is outside target_label taxonomy.\n"
        "- Use common biological/lexical taxonomy knowledge.\n"
        "- If uncertain between true/false, prefer true only when membership is plausible; otherwise false.\n"
        "Self-check before final answer:\n"
        "1) Is prediction inside target category? if yes => true.\n"
        "2) Is prediction unrelated to target category? if yes => false.\n"
        "Return ONLY valid JSON (no markdown, no extra text):\n"
        "{\"is_match\": <true|false>, \"reason\": \"one short sentence\"}\n"
        f"prediction={prediction}\n"
        f"target_label={target}\n"
    )


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _ollama_match(prediction: str, target_label: str, model: str) -> tuple[bool, str]:
    resolved_model = _resolve_model_name(model)
    payload = {
        "model": resolved_model,
        "prompt": _prompt(prediction=prediction, target=target_label),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    raw = body.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Invalid Ollama response payload")
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, dict) or "is_match" not in parsed:
        raise ValueError("LLM JSON did not include is_match")
    parsed_bool = _coerce_bool(parsed["is_match"])
    if parsed_bool is None:
        raise ValueError("LLM JSON is_match is not a boolean")
    is_match = parsed_bool
    reason = str(parsed.get("reason", "llm semantic decision"))
    return is_match, reason


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - _metrics.started_at),
        "default_model": DEFAULT_MODEL,
        "ollama_url": OLLAMA_URL,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "uptime_seconds": int(time.time() - _metrics.started_at),
        "total_requests": _metrics.total_requests,
        "llm_requests": _metrics.llm_requests,
        "llm_failures": _metrics.llm_failures,
    }


@app.post("/verify-label", response_model=VerifyResponse)
def verify_label(req: VerifyRequest) -> VerifyResponse:
    _metrics.total_requests += 1
    prediction = _normalize(req.prediction)
    target = _normalize(req.target_label)
    model = _resolve_model_name((req.llm_model or DEFAULT_MODEL).strip())
    if not prediction or not target:
        raise HTTPException(status_code=400, detail="prediction and target_label are required")

    _metrics.llm_requests += 1
    try:
        is_match, reason = _ollama_match(prediction=prediction, target_label=target, model=model)
        return VerifyResponse(is_match=is_match, reason=reason, method="ollama")
    except Exception as exc:
        _metrics.llm_failures += 1
        raise HTTPException(status_code=502, detail=f"llm endpoint failed: {exc}") from exc


# Backward-compatible alias for prior name.
@app.post("/match-label", response_model=VerifyResponse)
def match_label_alias(req: VerifyRequest) -> VerifyResponse:
    return verify_label(req)
