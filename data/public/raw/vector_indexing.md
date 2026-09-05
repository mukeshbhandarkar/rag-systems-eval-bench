# Vector index choices

Exact vector search compares a query against every stored embedding. For a small evaluation corpus, a NumPy matrix multiplication is simple, deterministic, and fast enough. It also avoids adding a persistent vector database.

Approximate nearest neighbor indexes trade some recall for lower search cost at large scale. Parameters controlling graph construction and query exploration can materially change both latency and retrieval quality, so they belong in experiment metadata.

The code name HNSW refers to Hierarchical Navigable Small World graphs. This public benchmark does not use HNSW, FAISS, or a hosted vector service. Its focus is controlled comparison rather than production-scale indexing.
