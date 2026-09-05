# Reproducible ML systems

An experiment should record code, configuration, data versions, and model identity. A metric without those inputs is difficult to audit. Run artifacts should be immutable or use distinct run identifiers to prevent accidental comparison with overwritten settings.

Deterministic preprocessing means the same source files and configuration produce the same ordered chunks and stable IDs. Timestamps may record when a build occurred, but they must not influence content identifiers or ranking tie rules.

The environment lock resolves package versions, while corpus-v1 names the evidence release. The benchmark in this project instead uses the coordinated public label ragbench-v1 for both its corpus and question annotations.
