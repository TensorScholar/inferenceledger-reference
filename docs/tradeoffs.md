# Tradeoffs — InferenceLedger

| Decision | Gained | Cost | Why it stands |
|----------|--------|------|---------------|
| Attempt-level rows | Retry spend + flakiness visible | More rows, join logic, provider-schema mapping | Request-level aggregation hides exactly the failure modes being managed |
| Unknown preserved (never zeroed) | Can't favor uninstrumented providers | REVIEW queue on every partial-billing candidate | Zero-fill is a systematic lie toward the worst option |
| Non-compensatory gate | Reliability/latency misses can't be bought off | Some "good deals" blocked; budget-setting overhead | Compensatory scoring ships silent quality holes; margins are the product |
| Team-owned floors/budgets as inputs | Gate stays a mechanism, not a policy | Adopters must do the hard work of setting bars | A model that sets your bar spends your error budget |
| Synthetic examples, frozen n=5 smoke | Illustrates gate behavior without inventing production data | No production validation claimed | Inventing traffic would be fabrication; honesty bounds the evidence at L2 |
