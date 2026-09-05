---
pretty_name: RAG Systems Evaluation Bench
language:
- en
license: cc-by-4.0
tags:
- retrieval
- rag
- information-retrieval
- synthetic
- evaluation
---

# RAG Systems Evaluation Bench

## Dataset Summary

RAG Systems Evaluation Bench `ragbench-v1` is a small, curated diagnostic dataset for comparing lexical BM25, dense MiniLM, and hybrid Reciprocal Rank Fusion retrieval. It contains original synthetic English technical passages about RAG and ML systems, explicit chunk-level relevance labels, and reference answers.

Dataset repository: [Max00035/rag-systems-eval-benchmark](https://huggingface.co/datasets/Max00035/rag-systems-eval-benchmark)  
Static results dashboard: [Max00035/rag-systems-eval-bench](https://huggingface.co/spaces/Max00035/rag-systems-eval-bench)  
Source code: [mukeshbhandarkar/rag-systems-eval-bench](https://github.com/mukeshbhandarkar/rag-systems-eval-bench)

It is intended for engineering evaluation, education, and reproducibility demonstrations. It is not an academically validated or industry-standard benchmark, and included scores are not state-of-the-art claims.

## Motivation

The benchmark makes retrieval behavior inspectable across exact lexical matches, paraphrases, acronyms, opaque identifiers, distractors, and straightforward controls. Persisted reference results show how retrieval failures can be distinguished from generation failures without an opaque LLM judge.

## Dataset Structure

- `corpus.jsonl`: 12 original source documents
- `chunks.jsonl`: 36 deterministic chunks used for retrieval
- `benchmark.jsonl`: 24 questions and annotations
- `retrieval_results.jsonl`: precomputed BM25, Dense MiniLM, and Hybrid RRF rankings
- `benchmark_summary.json`: aggregate and per-category metrics plus observed resource measurements

## Fields

Benchmark records contain:

- `question_id`: stable question identifier
- `question`: English question text
- `relevant_chunk_ids`: explicitly curated relevant chunks
- `reference_answers`: one or more acceptable short answers
- `answerable`: whether the synthetic corpus supports an answer
- `category`: one of `lexical`, `semantic`, `acronym`, `distractor`, `paraphrase`, `control`, or `unanswerable`
- `metadata`: reserved lightweight metadata

## Benchmark Categories

Categories are diagnostic slices, not statistically powered sub-benchmarks. Each contains only two to four questions. They expose exact identifiers, semantic rewording, ambiguous vocabulary, acronyms, and basic controls.

## Corpus Construction

Every corpus passage and benchmark question was created specifically for this benchmark as original synthetic, generic technical content. They were not scraped or copied from vendor documentation. Documents are deterministically divided into 45-word windows with 5-word overlap. Reference answers and relevance annotations were manually and explicitly selected only after stable chunk IDs were generated.

## Retrieval Systems

- BM25 with lowercase regex tokenization, `k1=1.5`, and `b=0.75`
- `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings and cosine-equivalent dot product
- Hybrid Reciprocal Rank Fusion with `rrf_k=60` and `candidate_k=20`

All reference runs use identical corpus, questions, labels, and cutoffs.

## Evaluation Metrics

The summary includes Recall@1, Recall@3, Recall@5, and MRR at each cutoff. Reference generation uses BM25 with `google/flan-t5-small`; answer evaluation reports normalized exact match, token F1, optional semantic similarity, abstention, and failure-decomposition counts.

Exact match can underrate valid verbose answers. Token F1 measures surface overlap. Semantic similarity is not a faithfulness metric, and a semantically similar answer may be unsupported by retrieved context.

## Reproduction

See the source project reproduction documentation. In short: sync with `uv`, rebuild and validate the corpus, run all three retrievers at top 5, then build the dashboard and dataset data from persisted artifacts. Generated results are presented in the [static Space](https://huggingface.co/spaces/Max00035/rag-systems-eval-bench).

## Limitations

This is a small synthetic diagnostic benchmark with manually curated labels and tiny category groups. It does not establish statistical significance, production performance, factual coverage of the broader ML field, or answer faithfulness. Labels may omit equivalent evidence. Runtime measurements describe one CPU-only development environment and are not universal performance claims.

## Licensing

The original synthetic benchmark dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` in this dataset repository for the license reference and attribution notice. Source code and the static dashboard are separately licensed under MIT.

## Versioning

Both the corpus and annotations use `ragbench-v1`. Any change to source text, chunk settings, questions, references, or relevance labels requires a reviewed version update.
