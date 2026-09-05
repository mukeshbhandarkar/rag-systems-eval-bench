<div align="center">

# RAG Systems Evaluation Bench

### Measure where RAG systems succeed — and where they fail.

CPU-only evaluation of lexical, dense, and hybrid retrieval with transparent retrieval and generation failure analysis.

[**Live Demo · Hugging Face Space**](https://huggingface.co/spaces/Max00035/rag-systems-eval-bench) · [**Dataset · Hugging Face**](https://huggingface.co/datasets/Max00035/rag-systems-eval-benchmark) · [**Source · GitHub**](https://github.com/mukeshbhandarkar/rag-systems-eval-bench)

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![CPU only](https://img.shields.io/badge/runtime-CPU--only-3949AB) ![$0 API cost](https://img.shields.io/badge/API%20cost-%240-087A55) ![MIT](https://img.shields.io/badge/code-MIT-17202A) ![CC BY 4.0](https://img.shields.io/badge/dataset-CC%20BY%204.0-B54708)

</div>

## Why This Project Exists

Aggregate RAG answer quality hides distinct engineering failures. Retrieval can fail to supply relevant evidence, or generation can fail even after receiving it. Those cases require different fixes.

**A bad RAG answer is not one failure mode.** This project evaluates retrieval independently, persists the evidence shown to generation, then classifies answer outcomes with explicit local metrics rather than an opaque LLM judge.

## Architecture

<!-- Add docs/assets/rag-systems-architecture.png here -->

```text
Benchmark
→ Deterministic chunking
→ BM25 / Dense MiniLM / Hybrid RRF
→ Retrieval evaluation
→ Optional FLAN-T5 generation
→ Answer evaluation
→ Failure decomposition
→ Static dashboard
```

Retrieval-only experiments remain independent of generation. Answer evaluation operates on persisted generation artifacts, so MiniLM and FLAN-T5 do not need to share memory in one process.

## Benchmark

`ragbench-v1` is an original, synthetic diagnostic benchmark for inspecting retrieval behavior—not a leaderboard or statistically authoritative dataset.

| Property | Value |
|---|---:|
| Technical documents | 12 |
| Deterministic chunks | 36 |
| Questions | 24 |
| Answerable / unanswerable | 22 / 2 |
| Diagnostic categories | 7 |
| Chunk size | 45 words |
| Chunk overlap | 5 words |

The categories cover lexical matches, semantic wording, acronyms, distractors, paraphrases, controls, and unanswerable questions. Each category has only two to four examples, so category differences are descriptive rather than statistically significant.

## Retrieval Systems

- **BM25** — lexical baseline using `rank-bm25`, transparent lowercase regex tokenization, `k1=1.5`, and `b=0.75`.
- **Dense MiniLM** — `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional L2-normalized embeddings, and cosine similarity through a NumPy dot product.
- **Hybrid RRF** — Reciprocal Rank Fusion with `rrf_k=60` and `candidate_k=20`. It combines ranks instead of averaging incompatible BM25 and cosine scores.

All three systems use the same corpus, questions, relevance labels, and evaluation cutoffs.

## Results

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7955 | 0.8636 | 0.9545 | 0.8818 | 0.43 s | 31.3 MB |
| Dense MiniLM | 0.8409 | 1.0000 | 1.0000 | 0.9545 | 15.30 s | 543.3 MB |
| Hybrid RRF | 0.7500 | 0.9545 | 1.0000 | 0.8977 | 15.15 s | 537.5 MB |

Dense MiniLM performed best overall on this small benchmark. Hybrid achieved perfect Recall@5 but did not outperform Dense at shallow ranking metrics. BM25 was dramatically cheaper in runtime and memory. These are observations from one CPU-only development machine, not universal performance claims; the results do not establish that Hybrid is superior.

One useful failure case is `ragbench-q006`, which asks why hybrid retrieval merges rank positions instead of averaging lexical and cosine values:

- BM25 misses the precise evidence in its top five.
- Dense ranks the evidence first.
- Hybrid recovers it at rank four.

The aggregate table alone cannot show this behavior. The checked-in [failure export](benchmark_results/ragbench-v1/failures.json) and static dashboard make the rankings inspectable.

## Retrieval vs Generation Failure

The reference generation configuration uses BM25 retrieval and `google/flan-t5-small` with greedy CPU decoding.

| Metric | Result |
|---|---:|
| Retrieval success | 0.9545 |
| Exact Match | 0.2273 |
| Token F1 | 0.5465 |
| Semantic similarity | 0.6071 |
| Exact successes | 5 |
| Generation failures | 16 |
| Retrieval failures | 1 |
| Answer without labeled retrieval | 0 |
| Abstention accuracy | 0.0 |

High retrieval success did **not** imply high answer correctness under the benchmark's explicit exact-match rule. This is the central reason retrieval and generation are evaluated separately.

Here, `generation_failure` means labeled evidence was retrieved but the output did not match an accepted answer after documented normalization. It is a reproducible benchmark classification—not a universal claim that every non-exact-match response is semantically wrong. Token F1 and semantic similarity remain visible alongside the stricter rule, and semantic similarity is not treated as faithfulness.

## Static Hugging Face Dashboard

The [live Space](https://huggingface.co/spaces/Max00035/rag-systems-eval-bench) is a static engineering dashboard built with HTML, CSS, and vanilla JavaScript. It has no backend, model runtime, API key, or analytics. The browser consumes versioned `results.json` and `failures.json`; it does not recompute benchmark results.

<!-- Add docs/assets/rag-evaluation-release-flow.png here -->

## Reproduce Locally

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required. Do not use direct `pip` installation.

```bash
uv sync
uv run pytest -q
uv run python scripts/validate_benchmark.py
```

Build the public corpus and run the common top-5 comparison:

```bash
uv run python scripts/build_corpus.py --raw-dir data/public/raw --output-dir data/public/processed --chunk-size 45 --overlap 5 --corpus-version ragbench-v1

uv run python scripts/run_benchmark.py --retriever bm25 --chunks data/public/processed/chunks.jsonl --questions data/public/benchmark/questions.jsonl --top-k 5 --run-id ragbench-v1-bm25-k5 --corpus-version ragbench-v1 --benchmark-version ragbench-v1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_benchmark.py --retriever dense --chunks data/public/processed/chunks.jsonl --questions data/public/benchmark/questions.jsonl --top-k 5 --run-id ragbench-v1-dense-k5 --corpus-version ragbench-v1 --benchmark-version ragbench-v1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_benchmark.py --retriever hybrid --candidate-k 20 --rrf-k 60 --chunks data/public/processed/chunks.jsonl --questions data/public/benchmark/questions.jsonl --top-k 5 --run-id ragbench-v1-hybrid-k5 --corpus-version ragbench-v1 --benchmark-version ragbench-v1
```

Evaluate persisted answers and rebuild dashboard data:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/evaluate_answers.py --run-dir runs/ragbench-v1-bm25-flan-t5-small --questions data/public/benchmark/questions.jsonl --semantic

uv run python scripts/build_dashboard_data.py --bm25-run runs/ragbench-v1-bm25-k5 --dense-run runs/ragbench-v1-dense-k5 --hybrid-run runs/ragbench-v1-hybrid-k5 --generation-run runs/ragbench-v1-bm25-flan-t5-small
```

Models download through the normal Hugging Face cache on first use. The offline flags apply after those model files have been cached. Detailed release guidance is in [docs/PUBLISHING.md](docs/PUBLISHING.md).

## Repository Structure

```text
src/ragbench/        ingestion, retrieval, generation, and evaluation code
data/public/         ragbench-v1 source documents, chunks, and annotations
benchmark_results/   versioned comparison summaries and failure cases
hf_dataset/          Hugging Face Dataset repository package
hf_space/            zero-compute static Space package
scripts/             corpus, benchmark, evaluation, and release CLIs
tests/               fast deterministic offline tests
docs/                publication guidance and architecture assets
```

## Design Constraints

- CPU-only and low-memory-conscious
- $0 hosted API cost
- no vector database or hosted LLM API
- deterministic chunk IDs, ranking tie rules, and versioned artifacts
- model-heavy stages can run in separate processes
- static presentation layer with no inference runtime

## Limitations

- Twenty-four questions are too few for statistical significance.
- The corpus is synthetic and covers a narrow technical domain.
- Category slices contain only two to four questions.
- Runtime and peak RSS are machine-specific observations.
- Exact Match can underrate valid verbose answers.
- MiniLM semantic similarity is neither correctness nor faithfulness.
- FLAN-T5-small is a lightweight reference generator, not a state-of-the-art model.
- Abstention currently performs poorly on the two unanswerable questions.
- Relevance labels and accepted answers may omit valid alternatives.

## Licenses

- Source code and static Space code: [MIT](LICENSE)
- Original synthetic benchmark dataset: [CC BY 4.0](hf_dataset/LICENSE)

---

<p align="center">
  Built by <strong>Mukesh Bhandarkar</strong><br>
  <a href="https://github.com/mukeshbhandarkar">GitHub</a> · <a href="https://huggingface.co/Max00035">Hugging Face</a>
</p>
