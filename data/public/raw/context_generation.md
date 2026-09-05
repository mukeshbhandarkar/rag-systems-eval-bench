# Context and generation

Retrieved chunks are placed into the generator context in rank order. The prompt tells the model to answer only from supplied evidence and to say it does not know when the evidence is insufficient. The complete prompt remains in the generation artifact for inspection.

A token-aware input budget protects the limited encoder window of FLAN-T5-small. Full chunks are added while they fit. If necessary, only the final included chunk is truncated, and later chunks are omitted.

Greedy decoding disables sampling and improves ordinary repeatability on the same CPU software stack. It does not guarantee that an answer is correct or grounded. Generation quality is evaluated separately from retrieval quality.
