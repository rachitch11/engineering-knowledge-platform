#!/usr/bin/env python3
"""Run the reproducible mock evaluation harness and print measured metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.evaluation.evaluate import build_harness
from app.generation.providers import MockLLMProvider
from app.retrieval.embeddings import MockEmbedding
from app.retrieval.reranker import MockReranker


def main() -> None:
    base = Path("data/evaluation/run")
    base.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        chroma_path=str(base / "chroma"),
        bm25_index_path=str(base / "bm25_index.json"),
        llm_provider="mock",
        environment="test",
    )
    harness = build_harness(
        settings,
        embedding_model=MockEmbedding(),
        reranker=MockReranker(),
        llm_provider=MockLLMProvider(),
        mode="mock_deterministic",
    )
    harness.seed_dataset()
    report = harness.run(k=5)
    payload = {
        "mode": report.mode,
        "k": report.k,
        "notes": report.notes,
        "retrieval": [item.__dict__ for item in report.retrieval],
        "answers": [item.__dict__ for item in report.answers],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
