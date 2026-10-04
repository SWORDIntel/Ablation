> Historical study/design notes for HIGH-GRAVITY/Khoj, a separate system. Predictions and reported measurements here are not acceptance evidence for the model surgery kit.

# Khoj Semantic Search Ablation Study
**HIGH-GRAVITY Integration Performance Analysis**

## Executive Summary

This document presents an ablation study of the Khoj semantic search integration within the HIGH-GRAVITY proxy system, measuring the impact of various components on context injection quality, search performance, and overall system throughput.

## Experimental Setup

### Baseline Configuration
- **Proxy**: HIGH-GRAVITY v3.6 with PegasusKhojBridge
- **Khoj Version**: Latest (khoj-ai/khoj main branch)
- **Index Size**: ~8 workspaces (HIGH-GRAVITY + 7 Windsurf projects)
- **File Types**: 15+ (.py, .js, .ts, .md, .json, .yaml, .sh, .c, .cpp, .rs, .go)
- **Hardware**: Standard development workstation

### Ablation Conditions

| Condition | Khoj Enabled | Auto-Reindex | Windsurf Detection | Context Injection | Top-K |
|-----------|--------------|--------------|-------------------|-------------------|-------|
| **A0** (Baseline) | ❌ | ❌ | ❌ | ❌ | - |
| **A1** (Khoj Only) | ✅ | ❌ | ❌ | ❌ | 4 |
| **A2** (+ Auto-Reindex) | ✅ | ✅ (5min) | ❌ | ❌ | 4 |
| **A3** (+ Windsurf) | ✅ | ✅ (5min) | ✅ | ❌ | 4 |
| **A4** (+ Injection) | ✅ | ✅ (5min) | ✅ | ✅ | 4 |
| **A5** (Top-K=8) | ✅ | ✅ (5min) | ✅ | ✅ | 8 |
| **A6** (Top-K=2) | ✅ | ✅ (5min) | ✅ | ✅ | 2 |

## Metrics

### Primary Metrics
1. **Search Latency** (ms) - Time to retrieve results from Khoj
2. **Injection Overhead** (ms) - Additional latency from context injection
3. **Context Relevance** (0-1) - Semantic similarity of injected snippets
4. **Token Overhead** (tokens) - Additional tokens from injected context
5. **Cache Hit Rate** (%) - HilbertCache effectiveness with/without Khoj

### Secondary Metrics
6. **Index Freshness** (min) - Time since last reindex
7. **Workspace Coverage** (%) - Percentage of active workspaces indexed
8. **Memory Usage** (MB) - Khoj server memory footprint
9. **Disk I/O** (MB/s) - Indexing throughput

## Hypotheses

### H1: Khoj Improves Context Relevance
**Prediction**: Conditions with Khoj (A1-A6) will show >30% improvement in context relevance vs baseline (A0).

**Rationale**: Semantic search provides more relevant code snippets than keyword-based retrieval.

### H2: Auto-Reindex Maintains Freshness
**Prediction**: A2-A6 will maintain <5min index staleness vs >60min for A1.

**Rationale**: 5-minute reindex interval keeps index synchronized with active development.

### H3: Windsurf Detection Increases Coverage
**Prediction**: A3-A6 will index 3-5x more files than A1-A2.

**Rationale**: Windsurf workspace detection adds multiple active projects to the index.

### H4: Context Injection Improves Response Quality
**Prediction**: A4-A6 will show >40% improvement in LLM response accuracy on code-related queries.

**Rationale**: Injected context provides grounding for LLM responses.

### H5: Top-K Tuning Affects Performance
**Prediction**: A5 (K=8) will have 2x latency of A6 (K=2) but 1.5x better relevance.

**Rationale**: More snippets increase latency but improve context coverage.

## Experimental Protocol

### Phase 1: Baseline Measurement (A0)
```bash
# Disable Khoj
export HG_KHOJ_ENABLED=false

# Run 100 test queries
python3 tests/ablation/khoj_baseline.py \
  --queries data/ablation-smoke/test_queries.json \
  --output results/A0_baseline.json
```

### Phase 2: Khoj-Only (A1)
```bash
# Enable Khoj, disable auto-reindex
export HG_KHOJ_ENABLED=true
export HG_KHOJ_REINDEX_INTERVAL=0

# Start Khoj
bash bin/khoj_launcher.sh

# Manual index once
curl -X POST http://127.0.0.1:9999/hg/khoj/reindex

# Run test suite
python3 tests/ablation/khoj_search_only.py \
  --queries data/ablation-smoke/test_queries.json \
  --output results/A1_khoj_only.json
```

### Phase 3: Auto-Reindex (A2)
```bash
# Enable auto-reindex
export HG_KHOJ_REINDEX_INTERVAL=300  # 5 minutes

# Restart proxy
bash hg_stop.sh && bash hg_start.sh --no-dashboard

# Run long-duration test (30 min)
python3 tests/ablation/khoj_auto_reindex.py \
  --duration 1800 \
  --output results/A2_auto_reindex.json
```

### Phase 4: Windsurf Detection (A3)
```bash
# Windsurf detection is automatic in PegasusKhojBridge
# Verify workspace count
curl -s http://127.0.0.1:9999/hg/khoj/status | jq '.indexed_workspaces'

# Run test suite
python3 tests/ablation/khoj_windsurf.py \
  --queries data/ablation-smoke/test_queries.json \
  --output results/A3_windsurf.json
```

### Phase 5: Context Injection (A4)
```bash
# Context injection is automatic when Khoj is enabled
# Measure injection overhead
python3 tests/ablation/khoj_injection.py \
  --queries data/ablation-smoke/test_queries.json \
  --measure-overhead \
  --output results/A4_injection.json
```

### Phase 6: Top-K Tuning (A5, A6)
```bash
# Test K=8
export HG_KHOJ_TOP_K=8
python3 tests/ablation/khoj_topk.py \
  --queries data/ablation-smoke/test_queries.json \
  --output results/A5_topk8.json

# Test K=2
export HG_KHOJ_TOP_K=2
python3 tests/ablation/khoj_topk.py \
  --queries data/ablation-smoke/test_queries.json \
  --output results/A6_topk2.json
```

## Expected Results

### Search Latency (ms)
| Condition | Mean | p50 | p95 | p99 |
|-----------|------|-----|-----|-----|
| A0 | - | - | - | - |
| A1 | 120 | 100 | 250 | 400 |
| A2 | 125 | 105 | 260 | 420 |
| A3 | 140 | 115 | 280 | 450 |
| A4 | 145 | 120 | 290 | 460 |
| A5 | 180 | 150 | 350 | 550 |
| A6 | 90 | 75 | 180 | 300 |

### Context Relevance (0-1 scale)
| Condition | Mean | Std Dev |
|-----------|------|---------|
| A0 | 0.45 | 0.15 |
| A1 | 0.62 | 0.12 |
| A2 | 0.63 | 0.11 |
| A3 | 0.71 | 0.10 |
| A4 | 0.78 | 0.09 |
| A5 | 0.82 | 0.08 |
| A6 | 0.68 | 0.11 |

### Token Overhead (tokens/query)
| Condition | Mean | Max |
|-----------|------|-----|
| A0 | 0 | 0 |
| A1 | 0 | 0 |
| A2 | 0 | 0 |
| A3 | 0 | 0 |
| A4 | 320 | 800 |
| A5 | 640 | 1600 |
| A6 | 160 | 400 |

## Analysis Framework

### Statistical Tests
1. **Paired t-test**: Compare A0 vs A4 for context relevance
2. **ANOVA**: Compare A4, A5, A6 for Top-K effects
3. **Regression**: Model latency as f(Top-K, workspace_count)

### Qualitative Analysis
1. **Error Analysis**: Categorize failed searches by query type
2. **Snippet Quality**: Manual review of top-10 injected snippets
3. **False Positives**: Identify irrelevant context injections

## Implementation Notes

### Test Query Categories
```python
# data/ablation-smoke/test_queries.json
{
  "code_search": [
    "How does the proxy handle API key rotation?",
    "Where is the HilbertCache implemented?",
    "Show me the Khoj integration code"
  ],
  "cross_project": [
    "Find authentication logic across all workspaces",
    "Where are environment variables configured?",
    "Show me all TypeScript interfaces"
  ],
  "semantic": [
    "How do I add a new worker type?",
    "What's the flow for model editing?",
    "Explain the VPU discovery process"
  ],
  "negative": [
    "What is the weather today?",
    "How do I cook pasta?",
    "Tell me a joke"
  ]
}
```

### Measurement Harness
```python
# tests/ablation/khoj_baseline.py
import time
import requests
import json
from pathlib import Path

class KhojAblationHarness:
    def __init__(self, proxy_url="http://127.0.0.1:9999"):
        self.proxy_url = proxy_url
        self.results = []
    
    def measure_search(self, query: str):
        start = time.time()
        resp = requests.post(
            f"{self.proxy_url}/hg/search",
            json={"query": query, "n": 4}
        )
        latency = (time.time() - start) * 1000
        
        return {
            "query": query,
            "latency_ms": latency,
            "results": resp.json() if resp.ok else [],
            "status": resp.status_code
        }
    
    def run_suite(self, queries: list):
        for q in queries:
            result = self.measure_search(q)
            self.results.append(result)
        
        return self.compute_stats()
    
    def compute_stats(self):
        latencies = [r["latency_ms"] for r in self.results]
        return {
            "mean_latency": sum(latencies) / len(latencies),
            "p50": sorted(latencies)[len(latencies)//2],
            "p95": sorted(latencies)[int(len(latencies)*0.95)],
            "p99": sorted(latencies)[int(len(latencies)*0.99)],
        }
```

## Validation Criteria

### Success Criteria
- ✅ H1: Context relevance improvement >30% (A4 vs A0)
- ✅ H2: Index staleness <5min for A2-A6
- ✅ H3: Workspace coverage 3-5x increase (A3 vs A1)
- ✅ H4: Response accuracy improvement >40% (A4 vs A0)
- ✅ H5: Latency/relevance tradeoff confirmed

### Failure Modes
- ❌ Search latency >500ms p95
- ❌ Context relevance <0.6 mean
- ❌ Token overhead >1000 tokens/query
- ❌ Memory usage >2GB for Khoj server

## Future Work

### Optimization Opportunities
1. **Caching Layer**: Cache Khoj results in HilbertCache
2. **Async Indexing**: Non-blocking reindex in background
3. **Selective Indexing**: Index only changed files
4. **Embedding Compression**: Reduce index size with quantization

### Extended Ablations
1. **Embedding Models**: Compare different embedding backends
2. **Chunking Strategies**: Test various snippet sizes
3. **Rerank Integration**: Add reranking layer for Top-K
4. **Multi-Modal**: Include images/diagrams in index

## References

- Khoj Documentation: https://docs.khoj.dev
- HIGH-GRAVITY Proxy: `/path/to/HIGH-GRAVITY/src/proxy.py`
- PegasusKhojBridge: `/path/to/HIGH-GRAVITY/src/pegasus/khoj_integration.py`
- AEGIS-LAB Framework: `/tank/btrfs-recovery/Ablation/README.md`

---

**Status**: Experimental framework defined, awaiting implementation and data collection.

**Last Updated**: 2026-04-20
