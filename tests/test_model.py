"""Regression tests for bounded embedding-worker concurrency (SRCH-0051)."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class _Array(list):
    def tolist(self):
        return list(self)


class _InstrumentedModel:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.batch_sizes = []
        self._guard = threading.Lock()

    def encode(self, texts, **kwargs):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.batch_sizes.append(len(texts))
        time.sleep(0.05)
        with self._guard:
            self.active -= 1
        output = {}
        if kwargs.get("return_dense"):
            output["dense_vecs"] = _Array([_Array([float(index)]) for index, _ in enumerate(texts)])
        if kwargs.get("return_sparse"):
            output["lexical_weights"] = [{str(index): float(index)} for index, _ in enumerate(texts)]
        return output


class ModelConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        source_dir = Path(__file__).parent
        sys.path.insert(0, str(source_dir))
        fake_package = types.ModuleType("FlagEmbedding")
        fake_package.BGEM3FlagModel = object
        sys.modules["FlagEmbedding"] = fake_package
        sys.modules.pop("model", None)
        self.model_module = importlib.import_module("model")
        self.fake_model = _InstrumentedModel()
        self.model_module._model = self.fake_model

    def tearDown(self) -> None:
        sys.path.pop(0)
        sys.modules.pop("model", None)
        sys.modules.pop("FlagEmbedding", None)

    def test_dense_encodes_are_single_flight(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.model_module.encode_dense, [f"doc-{i}"]) for i in range(2)]
            outputs = [future.result() for future in futures]

        self.assertEqual(outputs, [[[0.0]], [[0.0]]])
        self.assertEqual(self.fake_model.max_active, 1)

    def test_dense_encode_splits_large_requests_into_microbatches(self) -> None:
        self.model_module.INFERENCE_BATCH_SIZE = 2

        output = self.model_module.encode_dense(["a", "b", "c", "d", "e"])

        self.assertEqual(len(output), 5)
        self.assertEqual(self.fake_model.batch_sizes, [2, 2, 1])

    def test_dense_sparse_encodes_each_microbatch_once(self) -> None:
        self.model_module.INFERENCE_BATCH_SIZE = 2

        output = self.model_module.encode_dense_sparse(["a", "b", "c"])

        self.assertEqual(self.fake_model.batch_sizes, [2, 1])
        self.assertEqual(output["dense"], [[0.0], [1.0], [0.0]])
        self.assertEqual(
            output["sparse"],
            [{"0": 0.0}, {"1": 1.0}, {"0": 0.0}],
        )


if __name__ == "__main__":
    unittest.main()
