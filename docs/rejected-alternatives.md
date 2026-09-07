# Rejected alternatives — InferenceLedger

1. **Request-level cost rows (no attempt model).** Rejected: collapsing
   retries into one row hides both money (retry spend) and reliability
   (flakiness). The attempt is the unit of evidence; anything coarser lies
   by aggregation.
2. **Compensatory scoring (cost offsets reliability).** Rejected as the
   default: weighted scores let a cheap, slightly-worse model SHIP a silent
   quality hole. Teams that want compensatory tradeoffs can build them, but
   the reference gate stays non-compensatory — the margin structure is the
   product decision being demonstrated.
3. **Zero-filling unknown cost.** Rejected absolutely: `None → 0.0` makes
   missing billing look like free inference, which systematically favors the
   worst-instrumented provider. Unknown is preserved, counted, and propagated
   to REVIEW. This is the invariant the `unknown_cost_preserved` scenario pins.
4. **Single-provider cost tracking.** Rejected: migrations are comparisons.
   Tracking one provider's spend tells you nothing about whether a move is
   safe — the multi-provider aggregate + gate matrix is the minimum useful
   shape.
5. **Billing-grade claims from the n=5 smoke.** Rejected: five requests with
   zero retries cannot speak to retry economics or provider pricing. The smoke
   stays frozen as an instrumentation-existence artifact, never re-interpreted
   as a cost study.
