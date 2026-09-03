# Changelog

All notable changes to Embedding API are documented here.

## [2.2.1] - 2026-09-04

- Updated FlagEmbedding to 1.4.2, Uvicorn to 0.52.4, and Gunicorn to 26.2.0.
- Kept the canonical repository and the workspace deployment mirror on the
  same tested dependency set.

## [2.2.0] - 2026-09-03

- Added `/v1/embeddings/dense-sparse`, returning the exact ordered dense and
  sparse contract from a single BGE-M3 model pass.
- Added bounded per-worker microbatching and serialized access to each model
  instance to prevent overlapping in-process inference.
- Aligned runtime dependencies with the deployed Gunicorn/Uvicorn-worker stack.

## [2.1.0] - 2026-04-19

- Added the warmup endpoint and Prometheus-compatible request metrics.
- Added input-length validation with a 24,000-character default.
- Kept dense, sparse, ColBERT, and hybrid embedding modes compatible with the
  existing Tailscale-only service.

## [2.0.0] - 2026-04-19

- Published the initial BGE-M3 API with OpenAI-compatible dense embeddings,
  sparse lexical weights, ColBERT vectors, hybrid output, and health reporting.
