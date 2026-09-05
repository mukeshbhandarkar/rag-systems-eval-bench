# Retrieval stages

A retrieval system accepts a query and returns ranked evidence units. In this benchmark, each evidence unit is a text chunk with a stable identifier. Recall at a cutoff measures how much labeled evidence appears near the top. Reciprocal rank instead rewards placing the first relevant chunk early.

Lexical retrieval matches query terms against document terms. It is often effective for exact product codes, error identifiers, and uncommon acronyms. Its weakness appears when a question and its evidence express the same idea with different vocabulary.

Dense retrieval maps queries and chunks into numeric vectors. Nearby vectors can represent similar meanings even without exact word overlap. Dense search can miss rare literal identifiers, so lexical and semantic approaches expose complementary behavior.
