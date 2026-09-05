# Hybrid rank fusion

Hybrid retrieval combines evidence from lexical and semantic rankers. Reciprocal Rank Fusion, abbreviated RRF, assigns each candidate a contribution of one divided by k plus its rank from every component list. The project uses k equal to 60.

RRF combines ordering rather than raw scores. That matters because a BM25 score and a cosine similarity have unrelated numeric scales. Rank fusion needs no learned calibration and gives candidates appearing in both lists a useful cumulative advantage.

Each component retrieves up to twenty candidates before fusion, even when the final request asks for only five. The final tie rule sorts by fused score, then best component rank, then stable chunk identifier. One MiniLM instance is reused inside the hybrid retriever.
