"""Lightweight RSS memory logging (no document content or secrets)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def rss_mb() -> float | None:
    """Best-effort process RSS in megabytes (Unix/macOS)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        # Linux reports kilobytes; macOS reports bytes.
        if rss > 10_000_000:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def log_memory_event(event: str, **fields: Any) -> None:
    """Structured info log for ingestion/startup milestones."""
    parts = [f"event={event}"]
    rss = rss_mb()
    if rss is not None:
        parts.append(f"rss_mb={rss:.1f}")
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info("memory %s", " ".join(parts))


def memory_constrained_default() -> bool:
    """True when running on Render without an explicit override."""
    import os

    explicit = os.getenv("MEMORY_CONSTRAINED")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return is_render_host()


def is_render_host() -> bool:
    import os

    return os.getenv("RENDER", "").strip().lower() in {"true", "1", "yes"}
