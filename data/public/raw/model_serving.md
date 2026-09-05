# Model serving operations

Online inference services often batch compatible requests to improve accelerator utilization. Dynamic batching waits briefly for nearby requests, which can increase throughput but also adds queueing delay. Operators must balance batch efficiency against tail latency objectives.

A readiness probe indicates whether an instance should receive traffic. A liveness probe asks whether the process should be restarted. Confusing these signals can create restart loops during slow model loading or send requests to an instance before weights are available.

Model artifact version ms-2025-07-rc3 is approved for the canary pool. The rollback target is ms-2025-06-stable. These opaque release identifiers are deliberately included to test exact lexical retrieval.
