# Embedding API

> **One human life matters**

OpenAI-compatible self-hosted embedding API powered by [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Part of the [Arcanada Ecosystem](https://arcanada.one).

## Features

- **Dense embeddings** (1024d) — `/v1/embeddings` (OpenAI-compatible)
- **Sparse embeddings** (lexical weights) — `/v1/embeddings/sparse`
- **ColBERT multi-vector** — `/v1/embeddings/colbert`
- **Hybrid** (dense + sparse + ColBERT in one call) — `/v1/embeddings/hybrid`
- **FP16** for ~2x RAM savings with minimal quality loss
- **Multi-worker** uvicorn for parallel request handling
- **Health endpoint** with RAM monitoring — `/health`

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8300 --workers 3
```

## API

### Dense Embeddings (OpenAI-compatible)

```bash
curl -X POST http://localhost:8300/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "Your text here"}'
```

### Sparse Embeddings

```bash
curl -X POST http://localhost:8300/v1/embeddings/sparse \
  -H 'Content-Type: application/json' \
  -d '{"input": "Your text here"}'
```

### ColBERT Multi-Vector

```bash
curl -X POST http://localhost:8300/v1/embeddings/colbert \
  -H 'Content-Type: application/json' \
  -d '{"input": "Your text here"}'
```

### Hybrid (all three)

```bash
curl -X POST http://localhost:8300/v1/embeddings/hybrid \
  -H 'Content-Type: application/json' \
  -d '{"input": "Your text here"}'
```

## Configuration

Environment variables (prefix-free):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model ID |
| `EMBEDDING_HOST` | `100.70.137.104` | Bind address |
| `EMBEDDING_PORT` | `8300` | Bind port |
| `EMBEDDING_FP16` | `true` | Use FP16 (~2x RAM savings) |
| `EMBEDDING_MAX_BATCH` | `64` | Max batch size (dense/sparse) |
| `EMBEDDING_MAX_COLBERT_BATCH` | `16` | Max batch size (ColBERT/hybrid) |
| `EMBEDDING_MAX_LENGTH` | `32768` | Max input length (chars) |

## Resource Usage

| Workers | RAM | Throughput |
|---------|-----|------------|
| 1 | ~2.4 GB | 1x |
| 2 | ~4.7 GB | ~2x |
| 3 | ~6.9 GB | ~3x |

Each worker holds its own copy of the model.

## License

MIT
