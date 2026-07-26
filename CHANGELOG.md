# Changelog

All notable changes to Embedding API are documented here.

## [2.1.0] - 2026-04-19

- Added the warmup endpoint and Prometheus-compatible request metrics.
- Added input-length validation with a 24,000-character default.
- Kept dense, sparse, ColBERT, and hybrid embedding modes compatible with the
  existing Tailscale-only service.

## [2.0.0] - 2026-04-19

- Published the initial BGE-M3 API with OpenAI-compatible dense embeddings,
  sparse lexical weights, ColBERT vectors, hybrid output, and health reporting.
