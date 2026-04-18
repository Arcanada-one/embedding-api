#!/usr/bin/env python3
"""Fix and retry failed models with corrected configurations."""

import json
import time
import gc
import statistics
import resource

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def cosine_sim(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Same corpus as main benchmark
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
]

SIMILAR_PAIRS = [(0, 0), (1, 1), (3, 3), (7, 7), (8, 8)]

def bench_with_st(model_id, short, trust_remote=False, prompt_name=None):
    """Benchmark using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id} (sentence-transformers)")
    print(f"{'='*60}")

    gc.collect()
    rss_before = get_rss_mb()

    kwargs = {"trust_remote_code": True} if trust_remote else {}
    t0 = time.time()
    model = SentenceTransformer(model_id, **kwargs)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    rss_after = get_rss_mb()
    ram = rss_after - rss_before

    dim = model.get_sentence_embedding_dimension()
    print(f"  Dim: {dim}, RAM: {ram:.0f} MB")

    encode_kw = {}
    if prompt_name:
        encode_kw["prompt_name"] = prompt_name

    # Warmup
    model.encode(CHUNKS_EN[:3], **encode_kw)

    # Single latency (18 chunks)
    all_chunks = CHUNKS_RU + CHUNKS_EN
    times = []
    for c in all_chunks:
        t = time.time()
        model.encode([c], **encode_kw)
        times.append((time.time() - t) * 1000)

    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    tp_single = 1000.0 / p50

    # Batch
    t = time.time()
    model.encode(all_chunks, batch_size=32, **encode_kw)
    batch_ms = (time.time() - t) * 1000
    tp_batch = len(all_chunks) / (batch_ms / 1000)

    # Cross-lingual sim
    emb_ru = model.encode(CHUNKS_RU, **encode_kw)
    emb_en = model.encode(CHUNKS_EN, **encode_kw)
    sims = [cosine_sim(emb_ru[i], emb_en[j]) for i, j in SIMILAR_PAIRS]
    avg_sim = statistics.mean(sims)

    print(f"  p50={p50:.0f}ms p95={p95:.0f}ms batch={batch_ms:.0f}ms")
    print(f"  single={tp_single:.1f}ch/s batch={tp_batch:.1f}ch/s")
    print(f"  XL-sim={avg_sim:.4f}")

    del model; gc.collect()

    return {
        "model_id": model_id, "short": short, "dimension": dim,
        "latency_single_p50_ms": round(p50, 1),
        "latency_single_p95_ms": round(p95, 1),
        "latency_batch32_ms": round(batch_ms, 1),
        "throughput_single": round(tp_single, 2),
        "throughput_batch32": round(tp_batch, 2),
        "ram_mb": round(ram, 0),
        "avg_cross_lingual_sim": round(avg_sim, 4),
        "status": "ok",
    }


def bench_gte_raw(model_id="Alibaba-NLP/gte-multilingual-base", short="gte-multi"):
    """Benchmark GTE using transformers directly (sentence-transformers has index bug)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id} (raw transformers)")
    print(f"{'='*60}")

    gc.collect()
    rss_before = get_rss_mb()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    rss_after = get_rss_mb()
    ram = rss_after - rss_before

    def encode_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :]  # CLS token
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.numpy()

    dim = encode_texts(["test"]).shape[1]
    print(f"  Dim: {dim}, RAM: {ram:.0f} MB")

    # Warmup
    encode_texts(CHUNKS_EN[:3])

    all_chunks = CHUNKS_RU + CHUNKS_EN
    times = []
    for c in all_chunks:
        t = time.time()
        encode_texts([c])
        times.append((time.time() - t) * 1000)

    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    tp_single = 1000.0 / p50

    t = time.time()
    encode_texts(all_chunks)
    batch_ms = (time.time() - t) * 1000
    tp_batch = len(all_chunks) / (batch_ms / 1000)

    emb_ru = encode_texts(CHUNKS_RU)
    emb_en = encode_texts(CHUNKS_EN)
    sims = [cosine_sim(emb_ru[i], emb_en[j]) for i, j in SIMILAR_PAIRS]
    avg_sim = statistics.mean(sims)

    print(f"  p50={p50:.0f}ms p95={p95:.0f}ms batch={batch_ms:.0f}ms")
    print(f"  single={tp_single:.1f}ch/s batch={tp_batch:.1f}ch/s")
    print(f"  XL-sim={avg_sim:.4f}")

    del model, tokenizer; gc.collect()

    return {
        "model_id": model_id, "short": short, "dimension": dim,
        "latency_single_p50_ms": round(p50, 1),
        "latency_single_p95_ms": round(p95, 1),
        "latency_batch32_ms": round(batch_ms, 1),
        "throughput_single": round(tp_single, 2),
        "throughput_batch32": round(tp_batch, 2),
        "ram_mb": round(ram, 0),
        "avg_cross_lingual_sim": round(avg_sim, 4),
        "status": "ok",
    }


def bench_giga_raw(model_id="ai-sage/Giga-Embeddings-instruct", short="giga-emb"):
    """Benchmark GigaEmbeddings using transformers directly."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id} (raw transformers)")
    print(f"{'='*60}")

    gc.collect()
    rss_before = get_rss_mb()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    rss_after = get_rss_mb()
    ram = rss_after - rss_before

    def encode_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.numpy()

    dim = encode_texts(["test"]).shape[1]
    print(f"  Dim: {dim}, RAM: {ram:.0f} MB")

    encode_texts(CHUNKS_EN[:3])

    all_chunks = CHUNKS_RU + CHUNKS_EN
    times = []
    for c in all_chunks:
        t = time.time()
        encode_texts([c])
        times.append((time.time() - t) * 1000)

    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    tp_single = 1000.0 / p50

    t = time.time()
    encode_texts(all_chunks)
    batch_ms = (time.time() - t) * 1000
    tp_batch = len(all_chunks) / (batch_ms / 1000)

    emb_ru = encode_texts(CHUNKS_RU)
    emb_en = encode_texts(CHUNKS_EN)
    sims = [cosine_sim(emb_ru[i], emb_en[j]) for i, j in SIMILAR_PAIRS]
    avg_sim = statistics.mean(sims)

    print(f"  p50={p50:.0f}ms p95={p95:.0f}ms batch={batch_ms:.0f}ms")
    print(f"  single={tp_single:.1f}ch/s batch={tp_batch:.1f}ch/s")
    print(f"  XL-sim={avg_sim:.4f}")

    del model, tokenizer; gc.collect()

    return {
        "model_id": model_id, "short": short, "dimension": dim,
        "latency_single_p50_ms": round(p50, 1),
        "latency_single_p95_ms": round(p95, 1),
        "latency_batch32_ms": round(batch_ms, 1),
        "throughput_single": round(tp_single, 2),
        "throughput_batch32": round(tp_batch, 2),
        "ram_mb": round(ram, 0),
        "avg_cross_lingual_sim": round(avg_sim, 4),
        "status": "ok",
    }


def main():
    # Load bge-m3 result from previous run
    with open("/opt/embedding-api/benchmark/results.json") as f:
        existing = json.load(f)
    bge = [r for r in existing if r["short"] == "bge-m3"][0]

    results = [bge]

    # GTE — raw transformers (sentence-transformers has index bug)
    try:
        results.append(bench_gte_raw())
    except Exception as e:
        print(f"  GTE FAILED: {e}")
        results.append({"model_id": "Alibaba-NLP/gte-multilingual-base", "short": "gte-multi",
                        "dimension": 0, "latency_single_p50_ms": 0, "latency_single_p95_ms": 0,
                        "latency_batch32_ms": 0, "throughput_single": 0, "throughput_batch32": 0,
                        "ram_mb": 0, "avg_cross_lingual_sim": 0, "status": f"error: {e}"})

    # Nomic v1.5 — fixed prompt_name
    try:
        results.append(bench_with_st("nomic-ai/nomic-embed-text-v1.5", "nomic-v1.5",
                                     trust_remote=True, prompt_name="document"))
    except Exception as e:
        print(f"  Nomic v1.5 FAILED: {e}")
        results.append({"model_id": "nomic-ai/nomic-embed-text-v1.5", "short": "nomic-v1.5",
                        "dimension": 0, "latency_single_p50_ms": 0, "latency_single_p95_ms": 0,
                        "latency_batch32_ms": 0, "throughput_single": 0, "throughput_batch32": 0,
                        "ram_mb": 0, "avg_cross_lingual_sim": 0, "status": f"error: {e}"})

    # Nomic v2 MoE — fixed prompt_name
    try:
        results.append(bench_with_st("nomic-ai/nomic-embed-text-v2-moe", "nomic-v2-moe",
                                     trust_remote=True, prompt_name="document"))
    except Exception as e:
        print(f"  Nomic v2-moe FAILED: {e}")
        results.append({"model_id": "nomic-ai/nomic-embed-text-v2-moe", "short": "nomic-v2-moe",
                        "dimension": 0, "latency_single_p50_ms": 0, "latency_single_p95_ms": 0,
                        "latency_batch32_ms": 0, "throughput_single": 0, "throughput_batch32": 0,
                        "ram_mb": 0, "avg_cross_lingual_sim": 0, "status": f"error: {e}"})

    # GigaEmbeddings — raw transformers
    try:
        results.append(bench_giga_raw())
    except Exception as e:
        print(f"  GigaEmb FAILED: {e}")
        results.append({"model_id": "ai-sage/Giga-Embeddings-instruct", "short": "giga-emb",
                        "dimension": 0, "latency_single_p50_ms": 0, "latency_single_p95_ms": 0,
                        "latency_batch32_ms": 0, "throughput_single": 0, "throughput_batch32": 0,
                        "ram_mb": 0, "avg_cross_lingual_sim": 0, "status": f"error: {e}"})

    with open("/opt/embedding-api/benchmark/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*90}")
    print(f"{'Model':<20} {'Dim':>4} {'p50ms':>6} {'p95ms':>6} {'B32ms':>6} {'ch/s':>6} {'B32ch/s':>7} {'RAM MB':>7} {'XL-sim':>7} {'Status':<10}")
    print(f"{'='*90}")
    for r in results:
        print(f"{r['short']:<20} {r['dimension']:>4} {r['latency_single_p50_ms']:>6.0f} {r['latency_single_p95_ms']:>6.0f} "
              f"{r['latency_batch32_ms']:>6.0f} {r['throughput_single']:>6.1f} {r['throughput_batch32']:>7.1f} "
              f"{r['ram_mb']:>7.0f} {r['avg_cross_lingual_sim']:>7.4f} {r['status']:<10}")


if __name__ == "__main__":
    main()
