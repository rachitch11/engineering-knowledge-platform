"""Errors raised during answer generation."""


class GenerationError(Exception):
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InsufficientEvidenceError(GenerationError):
    status_code = 200  # handled as normal answer response with uncertainty flag


class LLMProviderError(GenerationError):
    status_code = 502


class LLMTimeoutError(LLMProviderError):
    status_code = 504


class LLMRateLimitError(LLMProviderError):
    status_code = 429
