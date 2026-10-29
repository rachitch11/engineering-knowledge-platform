# Evaluation Methodology (Phase 5)

## Dataset

The fixture dataset lives in `app/evaluation/dataset.py`:

- **Documents:** flange manual, vessel design note, and a prompt-injection sample
- **Answerable questions:** map to expected filename substrings and answer facts
- **Unanswerable question:** outside indexed content (`q_unanswerable`)

## How to run

```bash
python scripts/run_evaluation.py
```

This uses **MockEmbedding**, **MockReranker**, and **MockLLMProvider** with a temporary index under `data/evaluation/run/`. Results are deterministic and suitable for CI. They **do not** represent live SentenceTransformer, cross-encoder, or Ollama quality.

Automated tests in `tests/test_evaluation.py` run the same harness inside temporary directories.

## Metrics

| Metric | Definition |
| --- | --- |
| Recall@k | 1 if any relevant chunk appears in top-k, else 0 (averaged per strategy) |
| MRR | Reciprocal rank of the first relevant chunk |
| nDCG@k | Binary relevance nDCG at k |
| Citation accuracy | Share of cited chunks containing an expected fact substring |
| Answer fact recall | Share of expected facts found in the answer text |
| Latency | Wall time per query (milliseconds) |
| Failure rate | Share of queries returning zero hits |

## Strategies compared

1. `dense`
2. `bm25`
3. `hybrid` (RRF)
4. `hybrid_rerank` (hybrid retrieval + reranker)

## Measured mock fixture results (k=5, local run)

These numbers were produced by `python scripts/run_evaluation.py` in this repository (not fabricated):

| Strategy | Recall@5 | MRR | nDCG@5 | Mean latency (ms) | Failure rate |
| --- | --- | --- | --- | --- | --- |
| dense | 1.0 | 0.667 | 0.75 | ~1.1 | 0.0 |
| bm25 | 1.0 | 1.0 | 1.0 | ~0.24 | 0.0 |
| hybrid | 1.0 | 0.75 | 0.815 | ~1.7 | 0.0 |
| hybrid_rerank | 1.0 | 1.0 | 1.0 | ~1.4 | 0.0 |

Answer metrics on the same run:

| Question | Sufficient evidence | Citation accuracy |
| --- | --- | --- |
| q_flange_torque | true | 1.0 |
| q_vessel_pressure | true | 1.0 |
| q_unanswerable | **false** | 0.0 |

## Limitations

- Two answerable questions are too small for production benchmarking.
- Mock embeddings do not reflect semantic similarity of real models.
- Mock LLM answers cite evidence but do not fully reproduce expected facts (fact recall < 1 on some rows).
- Live-model evaluation requires manual runs with downloaded weights and Ollama/hosted LLM configured.
