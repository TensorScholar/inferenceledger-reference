# Economics model — InferenceLedger (reference)

## Unit of evidence

An **attempt**, not a "request." A user-visible request may contain multiple provider attempts (timeouts, fallbacks, retries). Collapsing them hides both cost and reliability.

## Known vs unknown money

If the provider did not return usable usage data, cost is `None`. Aggregators must not coerce `None` to `0.0`. A candidate with unknown cost cannot SHIP; the gate returns REVIEW or NO-GO depending on reliability.

## Non-compensatory margins

A required reliability or latency miss cannot be offset by a cost win. This is a product decision: teams that want compensatory scoring can implement it, but it is not this model's default, because it is how "the cheaper model" ships a silent quality hole.

## Historical smoke vs current model

The 2026-07-03 five-request smoke is retained as a frozen L2 artifact. It had zero retries, so it does not speak to retry economics. The reference examples in this package illustrate retry/unknown/latency cases; they are **not** a re-analysis of that smoke.
