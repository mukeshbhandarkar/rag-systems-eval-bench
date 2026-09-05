# Retrieval observability

Useful RAG telemetry separates retrieval latency, context construction time, and generation latency. A single end-to-end duration cannot identify which stage caused a slowdown. Percentiles such as p95 describe tail behavior better than an average alone.

Trace correlation identifier TRACE-RAG-8841 links a user request to its retrieval and generation spans. Logs should record chunk identifiers and ranks without copying sensitive document text. Metrics should aggregate behavior without exposing prompts.

Retrieval drift can appear when new documents change term statistics or embedding distributions. A versioned evaluation set provides an early warning by rerunning stable questions and relevance labels before deployment.
