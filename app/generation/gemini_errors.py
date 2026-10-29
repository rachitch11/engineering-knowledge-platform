"""Parse Google Generative Language API errors for retry and user messaging."""

from __future__ import annotations

import json

import httpx

from app.generation.exceptions import LLMProviderError, LLMRateLimitError

# Official model id: https://ai.google.dev/gemini-api/docs/models (Gemini 3.5 Flash-Lite)
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

_DAILY_QUOTA_MARKERS = (
    "per day",
    "perday",
    "generatecontentrequestsperday",
    "generaterequestsperday",
    "requests per day",
    "daily limit",
)

_RPM_MARKERS = (
    "per minute",
    "requests per minute",
    "generatecontentrequestsperminute",
)


def extract_google_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return (response.text or "").strip()
    if not isinstance(payload, dict):
        return str(payload)
    error = payload.get("error")
    if isinstance(error, dict):
        parts: list[str] = []
        for key in ("message", "status", "code"):
            value = error.get(key)
            if value is not None:
                parts.append(str(value))
        details = error.get("details")
        if isinstance(details, list):
            for item in details:
                parts.append(json.dumps(item, default=str))
        return " ".join(parts)
    return json.dumps(payload, default=str)


def _normalized_error_text(response: httpx.Response) -> str:
    raw = extract_google_error_text(response)
    return raw.lower().replace("_", "").replace("-", "").replace(" ", "")


def classify_rate_limit(response: httpx.Response) -> tuple[str, bool]:
    """Return a user-safe message and whether a bounded retry may help."""
    detail = extract_google_error_text(response)
    normalized = _normalized_error_text(response)

    if any(marker.replace("_", "").replace(" ", "") in normalized for marker in _DAILY_QUOTA_MARKERS):
        return (
            "Gemini daily request quota is exhausted for this model on the free tier "
            f"(HTTP 429). Switch GEMINI_MODEL to {DEFAULT_GEMINI_MODEL}, wait until the "
            "daily reset, or enable billing in Google AI Studio. "
            "AI Studio rate limits show Requests per day (RPD) separately from RPM/TPM.",
            False,
        )

    if any(marker.replace("_", "").replace(" ", "") in normalized for marker in _RPM_MARKERS):
        return (
            "Gemini requests-per-minute limit reached (HTTP 429). Wait a few seconds and "
            "retry, reduce traffic, or enable billing for higher RPM.",
            True,
        )

    if "resourceexhausted" in normalized or response.status_code == 429:
        return (
            "Gemini rate limit reached (HTTP 429). Check AI Studio → Rate limit for RPM, "
            "TPM, and RPD on your active GEMINI_MODEL. If RPD is full, change model or "
            "wait for reset; do not retry in a loop.",
            False,
        )

    return (
        f"Gemini API returned HTTP {response.status_code}. Check model id and quotas.",
        False,
    )


def raise_for_llm_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    if response.status_code == 429:
        message, _ = classify_rate_limit(response)
        raise LLMRateLimitError(message)
    detail = extract_google_error_text(response)
    raise LLMProviderError(f"Hosted LLM request failed: HTTP {response.status_code} {detail}")


def should_retry_response(response: httpx.Response) -> bool:
    if response.status_code in (502, 503):
        return True
    if response.status_code != 429:
        return False
    _, retryable = classify_rate_limit(response)
    return retryable
