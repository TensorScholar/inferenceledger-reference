# Operational lessons — InferenceLedger

1. **Retries are where the money hides.** The first version tracked requests;
   retry spend was invisible until a flaky-provider week made the bill
   disagree with the ledger. Attempt-first modeling came from that gap, not
   from theory.
2. **Unknown billing correlates with the providers you'd most like to
   switch to.** New, cheap, poorly-instrumented endpoints return partial
   usage data. Zero-filling would have systematically recommended the
   least-observable option — the REVIEW quarantine exists because the
   attractive candidate is the suspicious one.
3. **Latency budgets must be pre-declared.** Post-hoc "that feels slower"
   blocks every migration and approves none consistently. The gate takes the
   budget as input so the argument happens once (when setting policy), not
   per migration. The paired tradeoff scenarios (breach → NO-GO, within
   margin → SHIP) pin that the boundary moves only with the budget.
4. **Floors are team-owned, not model-owned.** The reference gate takes
   `min_success_rate` as a parameter deliberately. A cost model that sets
   your reliability bar is a cost model that spends your error budget.
