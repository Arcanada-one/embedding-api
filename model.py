"""Model loader — BGE-M3 with dense + sparse + ColBERT support."""

from FlagEmbedding import BGEM3FlagModel

from config import MODEL_ID, USE_FP16

_model: BGEM3FlagModel | None = None


def get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        _model = BGEM3FlagModel(MODEL_ID, use_fp16=USE_FP16)
    return _model


def encode_dense(texts: list[str]) -> list[list[float]]:
    model = get_model()
    output = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return output["dense_vecs"].tolist()


def encode_sparse(texts: list[str]) -> list[dict]:
    model = get_model()
    output = model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    return output["lexical_weights"]


def encode_colbert(texts: list[str]) -> list[list[list[float]]]:
    model = get_model()
    output = model.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
    return [v.tolist() for v in output["colbert_vecs"]]


def encode_hybrid(texts: list[str]) -> dict:
    model = get_model()
    output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    return {
        "dense": output["dense_vecs"].tolist(),
        "sparse": output["lexical_weights"],
        "colbert": [v.tolist() for v in output["colbert_vecs"]],
    }


def get_dimension() -> int:
    return 1024  # BGE-M3 fixed
