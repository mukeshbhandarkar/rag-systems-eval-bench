---
title: RAG Systems Evaluation Bench
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: BM25, dense, and hybrid RAG evaluation with failure analysis.
---

# RAG Systems Evaluation Bench

Zero-compute static dashboard for the locally generated `ragbench-v1` artifacts. It uses only HTML, CSS, JavaScript, and checked-in JSON. No model, API key, inference endpoint, analytics, or backend is required.

Benchmark dataset: [Max00035/rag-systems-eval-benchmark](https://huggingface.co/datasets/Max00035/rag-systems-eval-benchmark)

Source code: [mukeshbhandarkar/rag-systems-eval-bench](https://github.com/mukeshbhandarkar/rag-systems-eval-bench)

The static dashboard code is licensed under the MIT License.

Preview locally from the project root:

```bash
uv run python -m http.server 8000 --directory hf_space
```
