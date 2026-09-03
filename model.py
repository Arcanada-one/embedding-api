"""Model loader — BGE-M3 with dense + sparse + ColBERT support."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from config import INFERENCE_BATCH_SIZE, MODEL_ID, USE_FP16

_model: BGEM3FlagModel | None = None
_inference_lock = Lock()


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel(MODEL_ID, use_fp16=USE_FP16)
    return _model


def _microbatch(texts: list[str], encode_batch: Callable[[list[str]], list[Any]]) -> list[Any]:
    """Serialize singleton-model access and cap each inference allocation."""
    results: list[Any] = []
    # FastAPI executes sync handlers in a thread pool even with one Uvicorn worker.
    # Lock the full request so BGE-M3 never receives overlapping encode calls.
    with _inference_lock:
        for offset in range(0, len(texts), INFERENCE_BATCH_SIZE):
            results.extend(encode_batch(texts[offset : offset + INFERENCE_BATCH_SIZE]))
    return results


def encode_dense(texts: list[str]) -> list[list[float]]:
    model = get_model()

    def encode_batch(batch: list[str]) -> list[list[float]]:
        output = model.encode(batch, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return output["dense_vecs"].tolist()

    return _microbatch(texts, encode_batch)


def encode_dense_sparse(texts: list[str]) -> dict[str, list[Any]]:
    """Return paired dense and sparse vectors from one model pass."""
    model = get_model()

    def encode_batch(batch: list[str]) -> list[dict[str, Any]]:
        output = model.encode(
            batch,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            {
                "dense": output["dense_vecs"][index].tolist(),
                "sparse": output["lexical_weights"][index],
            }
            for index in range(len(batch))
        ]

    items = _microbatch(texts, encode_batch)
    return {
        "dense": [item["dense"] for item in items],
        "sparse": [item["sparse"] for item in items],
    }


def encode_sparse(texts: list[str]) -> list[dict]:
    model = get_model()

    def encode_batch(batch: list[str]) -> list[dict]:
        output = model.encode(batch, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        return output["lexical_weights"]

    return _microbatch(texts, encode_batch)


def encode_colbert(texts: list[str]) -> list[list[list[float]]]:
    model = get_model()

    def encode_batch(batch: list[str]) -> list[list[list[float]]]:
        output = model.encode(batch, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        return [v.tolist() for v in output["colbert_vecs"]]

    return _microbatch(texts, encode_batch)


def encode_hybrid(texts: list[str]) -> dict:
    model = get_model()

    def encode_batch(batch: list[str]) -> list[dict[str, Any]]:
        output = model.encode(batch, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        return [
            {
                "dense": output["dense_vecs"][i].tolist(),
                "sparse": output["lexical_weights"][i],
                "colbert": output["colbert_vecs"][i].tolist(),
            }
            for i in range(len(batch))
        ]

    items = _microbatch(texts, encode_batch)
    return {
        "dense": [item["dense"] for item in items],
        "sparse": [item["sparse"] for item in items],
        "colbert": [item["colbert"] for item in items],
    }


def get_dimension() -> int:
    return 1024  # BGE-M3 fixed
