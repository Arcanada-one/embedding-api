# CLAUDE.md

## What This Project Is

**Embedding API** — OpenAI-compatible self-hosted embedding server powered by BAAI/bge-m3. Supports dense (1024d), sparse (lexical), ColBERT (multi-vector), and hybrid modes. Part of the Arcanada Ecosystem.

## Tech Stack

- Python 3.12, FastAPI, uvicorn (multi-worker)
- BAAI/bge-m3 via FlagEmbedding library
- No database — stateless service

## Conventions

- `ruff check` + `ruff format` (line-length=120, py312)
- No hardcoded secrets
- Tailscale-only binding (no public exposure)

## Infrastructure

- Server: arcana-db (135.181.222.38), Tailscale IP 100.70.137.104
- Port: 8300
- systemd: `embedding-api.service`
- Code: `/opt/embedding-api/`

## Key Commands

```bash
ruff check *.py
ruff format *.py
uvicorn main:app --host 0.0.0.0 --port 8300 --workers 3
```

## Task Prefix

`SRCH` (shared with Scrutator — Embedding API is a Scrutator component).
