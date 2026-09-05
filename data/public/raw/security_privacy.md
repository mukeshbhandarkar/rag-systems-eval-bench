# Security and privacy boundaries

Retrieval systems should apply authorization filtering before evidence reaches a generator. Post-generation redaction is not a substitute for preventing unauthorized chunks from entering context. Tenant and document access rules belong in the retrieval path.

Prompt logs may contain private questions or source passages. A safer observability design records stable chunk IDs, timing, and aggregate metrics while limiting raw text retention. Access to evaluation artifacts should follow the sensitivity of their source data.

The marker SEC-POL-009 identifies the benchmark's synthetic access-control policy. It is not an external standard. The marker exists as an identifier-heavy retrieval case and carries no operational authority.
