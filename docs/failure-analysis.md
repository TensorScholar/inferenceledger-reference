# How InferenceLedger can fail

- **Mis-paired items.** If baseline item 3 is not the same prompt as candidate item 3, deltas are fiction.
- **Hidden retries outside the ledger.** If the SDK retries without emitting attempts, cost and reliability are both understated.
- **Coercing unknown to zero** in a downstream dashboard. The model can be correct and still be laundered by a BI tool.
- **Compensatory human override.** A PM can still ship a NO-GO. The ledger cannot prevent that; it can only make the override visible.
- **n=5 historical smoke** used as if it were a capacity plan. Explicitly disallowed in the claims ledger.
