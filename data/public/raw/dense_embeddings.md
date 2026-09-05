# Dense semantic retrieval

An embedding encoder converts text into a fixed-width numeric vector. The reference dense retriever uses all-MiniLM-L6-v2, which produces 384-dimensional sentence embeddings and runs locally on a CPU. Corpus vectors are computed once for each benchmark process.

Both query and corpus vectors are L2-normalized. Their dot product is therefore equivalent to cosine similarity. This avoids adding a separate similarity library and keeps the ranking calculation visible in NumPy.

Semantic vectors can connect paraphrases such as service slowdown and increased response latency. They are less dependable for opaque strings such as ZXQ-417 or build tag r2025.04, because those identifiers carry little reusable linguistic meaning.
