#!/usr/bin/env python3
"""Embedding model benchmark for INFRA-0020.

Compares 5 multilingual embedding models on CPU:
  1. BAAI/bge-m3
  2. Alibaba-NLP/gte-multilingual-base
  3. nomic-ai/nomic-embed-text-v1.5
  4. nomic-ai/nomic-embed-text-v2-moe
  5. ai-sage/Giga-Embeddings-instruct

Metrics: throughput (chunks/sec), p50/p95 latency, RAM, cosine similarity on known pairs.
"""

import json
import os
import sys
import time
import gc
import statistics
from dataclasses import dataclass, asdict

# Test corpus: 50 RU + 50 EN chunks (representative of LTM-0002 content)
CHUNKS_RU = [
    "Hindsight использует четырёхсторонний гибридный поиск: векторный, графовый, темпоральный и ментальные модели.",
    "Cognee строит граф знаний через LLM-извлечение сущностей и связей из текстовых документов.",
    "Graphiti обрабатывает сообщения асинхронно, извлекая факты и обновляя темпоральный граф.",
    "Оператор должен проверить транспортный слой через curl перед тем как обвинять фреймворк в ошибке.",
    "Бенчмарк проводился на бесплатных моделях OpenRouter без затрат на LLM-вызовы.",
    "Документы предобрабатываются: таблицы разворачиваются, code-fence экранируются, текст чанкируется.",
    "Recall@5 показывает долю релевантных документов в топ-5 результатов поиска.",
    "Эмбеддинги текста преобразуют семантическое содержание в числовые векторы фиксированной размерности.",
    "Self-hosted embedding API устраняет зависимость от облачных провайдеров и их ограничений.",
    "FastAPI с uvicorn обеспечивает асинхронную обработку HTTP-запросов на Python.",
    "Sentence-transformers — библиотека для получения dense-эмбеддингов предложений и параграфов.",
    "Tailscale создаёт mesh-сеть поверх WireGuard для безопасного доступа между серверами.",
    "Systemd управляет жизненным циклом сервисов в Linux: запуск, перезапуск, логирование.",
    "AMD Ryzen 5 3600 — шестиядерный процессор с базовой частотой 3.6 ГГц.",
    "Мультиязычные embedding-модели обучены на корпусах из 100+ языков включая русский.",
    "BGE-M3 поддерживает три типа поиска: dense, sparse и multi-vector retrieval.",
    "GTE использует архитектуру encoder-only с bidirectional attention для контекстуализации.",
    "Nomic Embed использует Matryoshka representation для гибкой размерности эмбеддингов.",
    "Cosine similarity измеряет угол между двумя векторами, нормализованный от -1 до 1.",
    "CPU-inference медленнее GPU в 10-50 раз, но не требует специализированного оборудования.",
    "Batch-обработка группирует несколько текстов для параллельного вычисления на CPU.",
    "ONNX Runtime позволяет ускорить inference через оптимизации графа вычислений.",
    "Квантизация INT8 уменьшает размер модели в 2 раза с минимальной потерей качества.",
    "HuggingFace Hub хранит предобученные модели и обеспечивает автоматическую загрузку.",
    "Токенизатор разбивает текст на subword-токены для входа в transformer-модель.",
    "Attention mechanism позволяет модели фокусироваться на релевантных частях входного текста.",
    "Pooling strategy (CLS, mean, max) определяет как токенные эмбеддинги агрегируются в один вектор.",
    "Нормализация L2 приводит все векторы к единичной длине для корректного cosine similarity.",
    "Prompt engineering для embedding-моделей использует instruction prefix для улучшения качества.",
    "Dimension reduction через PCA или Matryoshka позволяет уменьшить размер хранения без потери качества.",
    "Русский язык — один из 10 самых сложных для NLP из-за морфологии и свободного порядка слов.",
    "Мультиязычные модели обычно уступают монолингвальным на конкретном языке.",
    "MTEB benchmark оценивает embedding-модели на 7 задачах: retrieval, classification, clustering и др.",
    "ruMTEB — русскоязычная версия MTEB с 23 задачами для оценки качества эмбеддингов.",
    "Сбер выпустил GigaEmbeddings — модель с SOTA результатами на ruMTEB benchmark.",
    "Apache-2.0 и MIT — permissive лицензии, разрешающие коммерческое использование.",
    "Sentence-transformers автоматически выбирает оптимальный batch size для доступной памяти.",
    "PyTorch CPU backend использует MKL/OpenBLAS для ускорения матричных операций.",
    "Максимальная длина контекста 8192 токенов достаточна для чанков 800-1500 символов.",
    "OpenAI-compatible API формат стал де-факто стандартом для embedding-сервисов.",
    "Latency < 5 секунд на чанк — приемлемый порог для batch-инференса на CPU.",
    "RAM footprint модели зависит от количества параметров и precision (FP32, FP16, INT8).",
    "Warmup run необходим для корректного измерения — первый inference включает JIT-компиляцию.",
    "Детерминированный seed обеспечивает воспроизводимость результатов бенчмарка.",
    "Garbage collection между запусками моделей предотвращает утечки памяти.",
    "JSON-формат отчёта позволяет автоматически сравнивать результаты разных прогонов.",
    "p95 latency показывает время, в которое укладывается 95% запросов.",
    "Throughput в chunks/sec — ключевая метрика для оценки production-пригодности.",
    "Resource isolation через systemd cgroups предотвращает влияние на другие сервисы.",
    "Health check endpoint /health обеспечивает мониторинг доступности сервиса.",
]

CHUNKS_EN = [
    "Hindsight uses four-way hybrid retrieval: vector, graph, temporal, and mental models.",
    "Cognee builds a knowledge graph through LLM-based entity and relation extraction from text.",
    "Graphiti processes messages asynchronously, extracting facts and updating a temporal graph.",
    "Operators must verify the transport layer via curl before blaming a framework for errors.",
    "The benchmark was conducted using free OpenRouter models with zero LLM cost.",
    "Documents are preprocessed: tables flattened, code fences escaped, text chunked.",
    "Recall@5 measures the fraction of relevant documents in the top-5 search results.",
    "Text embeddings transform semantic content into fixed-dimension numerical vectors.",
    "A self-hosted embedding API eliminates dependency on cloud providers and their limitations.",
    "FastAPI with uvicorn provides asynchronous HTTP request handling in Python.",
    "Sentence-transformers is a library for computing dense embeddings of sentences and paragraphs.",
    "Tailscale creates a mesh network over WireGuard for secure access between servers.",
    "Systemd manages service lifecycle in Linux: start, restart, and logging.",
    "The AMD Ryzen 5 3600 is a six-core processor with a base frequency of 3.6 GHz.",
    "Multilingual embedding models are trained on corpora from 100+ languages including Russian.",
    "BGE-M3 supports three retrieval modes: dense, sparse, and multi-vector.",
    "GTE uses encoder-only architecture with bidirectional attention for contextualization.",
    "Nomic Embed uses Matryoshka representation learning for flexible embedding dimensions.",
    "Cosine similarity measures the angle between two vectors, normalized from -1 to 1.",
    "CPU inference is 10-50x slower than GPU but requires no specialized hardware.",
    "Batch processing groups multiple texts for parallel computation on CPU cores.",
    "ONNX Runtime accelerates inference through computation graph optimizations.",
    "INT8 quantization reduces model size by 2x with minimal quality degradation.",
    "HuggingFace Hub stores pretrained models and provides automatic downloading.",
    "A tokenizer splits text into subword tokens for input to the transformer model.",
    "The attention mechanism allows the model to focus on relevant parts of the input.",
    "Pooling strategy (CLS, mean, max) determines how token embeddings are aggregated.",
    "L2 normalization brings all vectors to unit length for correct cosine similarity.",
    "Prompt engineering for embedding models uses instruction prefixes for quality improvement.",
    "Dimension reduction via PCA or Matryoshka reduces storage without quality loss.",
    "Russian is one of the 10 hardest languages for NLP due to morphology and free word order.",
    "Multilingual models typically underperform monolingual ones on specific languages.",
    "The MTEB benchmark evaluates embedding models on 7 tasks: retrieval, classification, etc.",
    "ruMTEB is the Russian version of MTEB with 23 tasks for embedding quality evaluation.",
    "Sber released GigaEmbeddings with SOTA results on the ruMTEB benchmark.",
    "Apache-2.0 and MIT are permissive licenses allowing commercial use.",
    "Sentence-transformers automatically selects optimal batch size for available memory.",
    "The PyTorch CPU backend uses MKL/OpenBLAS for matrix operation acceleration.",
    "A maximum context length of 8192 tokens is sufficient for 800-1500 character chunks.",
    "The OpenAI-compatible API format has become the de facto standard for embedding services.",
    "Latency under 5 seconds per chunk is an acceptable threshold for CPU batch inference.",
    "Model RAM footprint depends on parameter count and precision (FP32, FP16, INT8).",
    "A warmup run is needed for correct measurement — first inference includes JIT compilation.",
    "A deterministic seed ensures reproducibility of benchmark results.",
    "Garbage collection between model runs prevents memory leaks.",
    "JSON report format enables automatic comparison of results across different runs.",
    "p95 latency shows the time within which 95% of requests complete.",
    "Throughput in chunks/sec is the key metric for production readiness assessment.",
    "Resource isolation via systemd cgroups prevents impact on other services.",
    "A health check endpoint /health provides service availability monitoring.",
]

# Known similar pairs for cosine similarity quality check
# (index_ru, index_en) — same semantic content in different languages
SIMILAR_PAIRS = [
    (0, 0),   # Hindsight hybrid retrieval
    (1, 1),   # Cognee knowledge graph
    (3, 3),   # Operator-First Attribution
    (7, 7),   # Text embeddings definition
    (8, 8),   # Self-hosted API
]

MODELS = [
    {
        "id": "BAAI/bge-m3",
        "short": "bge-m3",
        "trust_remote_code": False,
        "prompt_name": None,
    },
    {
        "id": "Alibaba-NLP/gte-multilingual-base",
        "short": "gte-multi",
        "trust_remote_code": True,
        "prompt_name": None,
    },
    {
        "id": "nomic-ai/nomic-embed-text-v1.5",
        "short": "nomic-v1.5",
        "trust_remote_code": True,
        "prompt_name": "search_document",
    },
    {
        "id": "nomic-ai/nomic-embed-text-v2-moe",
        "short": "nomic-v2-moe",
        "trust_remote_code": True,
        "prompt_name": "search_document",
    },
    {
        "id": "ai-sage/Giga-Embeddings-instruct",
        "short": "giga-emb",
        "trust_remote_code": True,
        "prompt_name": None,
    },
]


@dataclass
class BenchResult:
    model_id: str
    short: str
    dimension: int
    latency_single_p50_ms: float
    latency_single_p95_ms: float
    latency_batch32_ms: float
    throughput_single: float  # chunks/sec (batch=1)
    throughput_batch32: float  # chunks/sec (batch=32)
    ram_mb: float
    avg_cross_lingual_sim: float  # avg cosine sim on known pairs
    status: str  # "ok" | "error: ..."


def get_rss_mb():
    """Current process RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: bytes, Linux: KB
    # On Linux ru_maxrss is in KB


def cosine_sim(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bench_model(model_cfg: dict) -> BenchResult:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model_id = model_cfg["id"]
    short = model_cfg["short"]
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id}")
    print(f"{'='*60}")

    try:
        gc.collect()
        rss_before = get_rss_mb()

        load_start = time.time()
        kwargs = {"trust_remote_code": True} if model_cfg["trust_remote_code"] else {}
        model = SentenceTransformer(model_id, **kwargs)
        load_time = time.time() - load_start
        print(f"  Model loaded in {load_time:.1f}s")

        rss_after = get_rss_mb()
        ram_delta = rss_after - rss_before

        dim = model.get_sentence_embedding_dimension()
        print(f"  Dimension: {dim}")
        print(f"  RAM delta: {ram_delta:.0f} MB")

        # Warmup (3 texts)
        encode_kwargs = {}
        if model_cfg.get("prompt_name"):
            encode_kwargs["prompt_name"] = model_cfg["prompt_name"]
        model.encode(CHUNKS_EN[:3], **encode_kwargs)
        print("  Warmup done")

        # --- Single-text latency (all 100 chunks, one by one) ---
        all_chunks = CHUNKS_RU + CHUNKS_EN
        single_times = []
        for chunk in all_chunks:
            t0 = time.time()
            model.encode([chunk], **encode_kwargs)
            single_times.append((time.time() - t0) * 1000)  # ms

        p50 = statistics.median(single_times)
        p95 = sorted(single_times)[int(len(single_times) * 0.95)]
        throughput_single = 1000.0 / p50  # chunks/sec at median

        print(f"  Single: p50={p50:.0f}ms, p95={p95:.0f}ms, throughput={throughput_single:.1f} ch/s")

        # --- Batch-32 latency ---
        batch_times = []
        for i in range(0, len(all_chunks), 32):
            batch = all_chunks[i:i+32]
            t0 = time.time()
            model.encode(batch, batch_size=32, **encode_kwargs)
            batch_times.append((time.time() - t0) * 1000)

        avg_batch_ms = statistics.mean(batch_times)
        avg_batch_size = len(all_chunks) / len(batch_times)
        throughput_batch = avg_batch_size / (avg_batch_ms / 1000.0)

        print(f"  Batch32: avg={avg_batch_ms:.0f}ms, throughput={throughput_batch:.1f} ch/s")

        # --- Cross-lingual cosine similarity ---
        emb_ru = model.encode(CHUNKS_RU, **encode_kwargs)
        emb_en = model.encode(CHUNKS_EN, **encode_kwargs)

        sims = []
        for ru_idx, en_idx in SIMILAR_PAIRS:
            sim = cosine_sim(emb_ru[ru_idx], emb_en[en_idx])
            sims.append(sim)
            print(f"  Pair ({ru_idx},{en_idx}): cosine={sim:.4f}")

        avg_sim = statistics.mean(sims)
        print(f"  Avg cross-lingual similarity: {avg_sim:.4f}")

        # Cleanup
        del model
        gc.collect()

        return BenchResult(
            model_id=model_id,
            short=short,
            dimension=dim,
            latency_single_p50_ms=round(p50, 1),
            latency_single_p95_ms=round(p95, 1),
            latency_batch32_ms=round(avg_batch_ms, 1),
            throughput_single=round(throughput_single, 2),
            throughput_batch32=round(throughput_batch, 2),
            ram_mb=round(ram_delta, 0),
            avg_cross_lingual_sim=round(avg_sim, 4),
            status="ok",
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        gc.collect()
        return BenchResult(
            model_id=model_id, short=short, dimension=0,
            latency_single_p50_ms=0, latency_single_p95_ms=0,
            latency_batch32_ms=0, throughput_single=0, throughput_batch32=0,
            ram_mb=0, avg_cross_lingual_sim=0,
            status=f"error: {e}",
        )


def main():
    print("INFRA-0020 Embedding Model Benchmark")
    print(f"Corpus: {len(CHUNKS_RU)} RU + {len(CHUNKS_EN)} EN = {len(CHUNKS_RU)+len(CHUNKS_EN)} chunks")
    print(f"Similar pairs: {len(SIMILAR_PAIRS)}")
    print(f"Models: {len(MODELS)}")

    results = []
    for m in MODELS:
        result = bench_model(m)
        results.append(asdict(result))

    # Save results
    out_path = "/opt/embedding-api/benchmark/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<20} {'Dim':>4} {'p50ms':>6} {'p95ms':>6} {'B32ms':>6} {'ch/s':>6} {'B32ch/s':>7} {'RAM MB':>7} {'XL-sim':>7} {'Status':<10}")
    print(f"{'='*90}")
    for r in results:
        print(f"{r['short']:<20} {r['dimension']:>4} {r['latency_single_p50_ms']:>6.0f} {r['latency_single_p95_ms']:>6.0f} "
              f"{r['latency_batch32_ms']:>6.0f} {r['throughput_single']:>6.1f} {r['throughput_batch32']:>7.1f} "
              f"{r['ram_mb']:>7.0f} {r['avg_cross_lingual_sim']:>7.4f} {r['status']:<10}")


if __name__ == "__main__":
    main()
