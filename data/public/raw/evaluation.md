# Evaluation discipline

Retrieval Recall at K measures the fraction of labeled relevant chunks found among the first K results. Mean Reciprocal Rank uses the reciprocal position of the first relevant result. These metrics answer different questions and should remain visible separately.

Normalized exact match is strict: a prediction must equal an accepted answer after documented text normalization. Token F1 gives partial credit for overlapping answer tokens. Neither metric proves that an answer is supported by retrieved evidence.

Failure decomposition distinguishes retrieval failure from generation failure. If labeled evidence was retrieved but the answer misses the reference, the case is a generation failure under the exact-match rule. An exact answer without labeled evidence is retained as a separate diagnostic category rather than silently called success.
