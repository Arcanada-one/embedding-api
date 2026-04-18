"""OpenAI-compatible embedding API server with sparse + ColBERT support."""

import os
import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import MAX_BATCH_SIZE, MAX_COLBERT_BATCH, MAX_INPUT_LENGTH, MODEL_ID, USE_FP16
from model import encode_dense, encode_sparse, encode_colbert, encode_hybrid, get_dimension, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-api")

app = FastAPI(title="Embedding API", version="2.0.0")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "local-embedding"
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class SparseData(BaseModel):
    sparse_weights: dict[str, float]
    index: int


class HybridData(BaseModel):
    dense: list[float]
    sparse_weights: dict[str, float]
    colbert: list[list[float]]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class SparseResponse(BaseModel):
    object: str = "list"
    data: list[SparseData]
    model: str
    usage: Usage


class HybridResponse(BaseModel):
    object: str = "list"
    data: list[HybridData]
    model: str
    usage: Usage


def _parse_input(req: EmbeddingRequest, max_batch: int = MAX_BATCH_SIZE) -> list[str]:
    texts = [req.input] if isinstance(req.input, str) else req.input
    if len(texts) == 0:
        raise HTTPException(400, "input must not be empty")
    if len(texts) > max_batch:
        raise HTTPException(400, f"batch size {len(texts)} exceeds max {max_batch}")
    for i, t in enumerate(texts):
        if len(t) > MAX_INPUT_LENGTH:
            raise HTTPException(400, f"input[{i}] length {len(t)} exceeds max {MAX_INPUT_LENGTH}")
        if len(t) == 0:
            raise HTTPException(400, f"input[{i}] must not be empty")
    return texts


def _usage(texts: list[str]) -> Usage:
    total_tokens = sum(len(t.split()) for t in texts)
    return Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)


@app.on_event("startup")
def startup():
    logger.info(f"Loading model: {MODEL_ID} (fp16={USE_FP16})")
    t0 = time.time()
    get_model()
    logger.info(f"Model loaded in {time.time()-t0:.1f}s, dim={get_dimension()}")


@app.get("/health")
def health():
    import psutil
    proc = psutil.Process(os.getpid())
    ram_mb = round(proc.memory_info().rss / 1024 / 1024)
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dimension": get_dimension(),
        "fp16": USE_FP16,
        "ram_mb": ram_mb,
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(req: EmbeddingRequest):
    texts = _parse_input(req)
    t0 = time.time()
    embeddings = encode_dense(texts)
    logger.info(f"Dense: {len(texts)} texts in {time.time()-t0:.3f}s")
    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)],
        model=MODEL_ID,
        usage=_usage(texts),
    )


@app.post("/v1/embeddings/sparse", response_model=SparseResponse)
def create_sparse_embeddings(req: EmbeddingRequest):
    texts = _parse_input(req)
    t0 = time.time()
    sparse = encode_sparse(texts)
    logger.info(f"Sparse: {len(texts)} texts in {time.time()-t0:.3f}s")
    return SparseResponse(
        data=[SparseData(sparse_weights={str(k): float(v) for k, v in s.items()}, index=i) for i, s in enumerate(sparse)],
        model=MODEL_ID,
        usage=_usage(texts),
    )


@app.post("/v1/embeddings/colbert")
def create_colbert_embeddings(req: EmbeddingRequest):
    texts = _parse_input(req, max_batch=MAX_COLBERT_BATCH)
    t0 = time.time()
    colbert = encode_colbert(texts)
    logger.info(f"ColBERT: {len(texts)} texts in {time.time()-t0:.3f}s")
    return {
        "object": "list",
        "data": [{"colbert_vecs": vecs, "index": i} for i, vecs in enumerate(colbert)],
        "model": MODEL_ID,
        "usage": _usage(texts).model_dump(),
    }


@app.post("/v1/embeddings/hybrid", response_model=HybridResponse)
def create_hybrid_embeddings(req: EmbeddingRequest):
    texts = _parse_input(req, max_batch=MAX_COLBERT_BATCH)
    t0 = time.time()
    result = encode_hybrid(texts)
    logger.info(f"Hybrid: {len(texts)} texts in {time.time()-t0:.3f}s")
    return HybridResponse(
        data=[
            HybridData(
                dense=result["dense"][i],
                sparse_weights={str(k): float(v) for k, v in result["sparse"][i].items()},
                colbert=result["colbert"][i],
                index=i,
            )
            for i in range(len(texts))
        ],
        model=MODEL_ID,
        usage=_usage(texts),
    )


if __name__ == "__main__":
    import uvicorn
    from config import BIND_HOST, BIND_PORT
    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
