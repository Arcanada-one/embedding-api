#!/usr/bin/env python3
"""Retry failed models from embedding-bench.py."""

import json
import sys
sys.path.insert(0, "/opt/embedding-api/benchmark")

from importlib import import_module
bench = import_module("embedding-bench")

# Only retry models that failed
RETRY_MODELS = [
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

# Load existing results
with open("/opt/embedding-api/benchmark/results.json") as f:
    existing = json.load(f)

# Keep bge-m3 result, replace others
bge_result = [r for r in existing if r["short"] == "bge-m3"][0]
all_results = [bge_result]

for m in RETRY_MODELS:
    from dataclasses import asdict
    result = bench.bench_model(m)
    all_results.append(asdict(result))

# Save
with open("/opt/embedding-api/benchmark/results.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*90}")
print(f"{'Model':<20} {'Dim':>4} {'p50ms':>6} {'p95ms':>6} {'B32ms':>6} {'ch/s':>6} {'B32ch/s':>7} {'RAM MB':>7} {'XL-sim':>7} {'Status':<10}")
print(f"{'='*90}")
for r in all_results:
    print(f"{r['short']:<20} {r['dimension']:>4} {r['latency_single_p50_ms']:>6.0f} {r['latency_single_p95_ms']:>6.0f} "
          f"{r['latency_batch32_ms']:>6.0f} {r['throughput_single']:>6.1f} {r['throughput_batch32']:>7.1f} "
          f"{r['ram_mb']:>7.0f} {r['avg_cross_lingual_sim']:>7.4f} {r['status']:<10}")
