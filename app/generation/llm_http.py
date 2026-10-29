"""HTTP timeouts and bounded retries for hosted LLM providers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.generation.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from app.generation.gemini_errors import raise_for_llm_response, should_retry_response

_TIMEOUT_USER_MESSAGE = (
    "The language model did not respond in time. Try again, use gemini-3.5-flash-lite, "
    "or increase LLM_TIMEOUT_SECONDS."
)


@dataclass(frozen=True)
class LLMHttpConfig:
    connect_timeout_seconds: float
    read_timeout_seconds: float
    write_timeout_seconds: float
    pool_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float

    def httpx_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout_seconds,
            read=self.read_timeout_seconds,
            write=self.write_timeout_seconds,
            pool=self.pool_timeout_seconds,
        )


def post_with_retries(
    client: httpx.Client,
    url: str,
    *,
    config: LLMHttpConfig,
    **kwargs: object,
) -> httpx.Response:
    """POST with bounded retries for transient provider/network failures."""
    attempts = config.max_retries + 1

    for attempt in range(attempts):
        try:
            response = client.post(url, **kwargs)  # type: ignore[arg-type]
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(_TIMEOUT_USER_MESSAGE) from exc
        except httpx.HTTPError as exc:
            if attempt >= attempts - 1:
                raise LLMProviderError(f"Hosted LLM request failed: {exc}") from exc
            if isinstance(exc, httpx.ConnectError | httpx.ReadError):
                time.sleep(config.retry_backoff_seconds * (2**attempt))
                continue
            raise LLMProviderError(f"Hosted LLM request failed: {exc}") from exc

        if not response.is_success:
            if attempt < attempts - 1 and should_retry_response(response):
                time.sleep(config.retry_backoff_seconds * (2**attempt))
                continue
            raise_for_llm_response(response)

        return response

    raise LLMProviderError("Hosted LLM request failed after retries.")
