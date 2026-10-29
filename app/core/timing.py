"""Structured pipeline timing without logging sensitive content."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PipelineTimer:
    """Collect stage durations in milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stages[name] = self.stages.get(name, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def as_dict(self) -> dict[str, float]:
        payload = dict(self.stages)
        payload["total"] = self.total_ms
        return payload
