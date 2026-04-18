"""Embedding API configuration."""

import os

MODEL_ID = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BIND_HOST = os.getenv("EMBEDDING_HOST", "100.70.137.104")  # Tailscale IP only
BIND_PORT = int(os.getenv("EMBEDDING_PORT", "8300"))
MAX_BATCH_SIZE = int(os.getenv("EMBEDDING_MAX_BATCH", "64"))
MAX_INPUT_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "32768"))
TRUST_REMOTE_CODE = os.getenv("EMBEDDING_TRUST_REMOTE", "false").lower() == "true"
USE_FP16 = os.getenv("EMBEDDING_FP16", "true").lower() == "true"
MAX_COLBERT_BATCH = int(os.getenv("EMBEDDING_MAX_COLBERT_BATCH", "16"))
