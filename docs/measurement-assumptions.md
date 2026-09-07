# Measurement assumptions

- Pricing tables are repository-local, not invoices.
- Latency is attempt-level wall time as recorded by the client, not provider-reported.
- p95 in the reference examples is supplied as a scenario parameter, not computed from n=3 samples. Computing p95 on three points would be theatre.
- "Success" is a deterministic validator pass in these examples, not a human preference score.
- Cross-provider semantic quality is out of scope unless a paired outcome contract is supplied.

If any of these is false for a real migration, the gate output is not interpretable as a ship decision.
