# Chunk construction

Chunking divides source documents into retrieval units. Very large chunks can mix unrelated topics, while very small chunks may separate a claim from the details needed to understand it. A useful configuration balances evidence completeness with ranking precision.

Overlapping windows repeat boundary words in neighboring chunks. Overlap can preserve a sentence that crosses a boundary, but excessive overlap creates near-duplicate candidates that crowd the ranking. This benchmark uses deterministic word windows so identical inputs and settings always produce identical chunk identifiers.

Changing chunk size or overlap changes the evidence collection and therefore invalidates old relevance labels. Corpus versions must record these settings. Benchmark annotations should be reviewed whenever the source text or chunk configuration changes.
