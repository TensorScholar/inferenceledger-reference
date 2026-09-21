# Featured capsule — InferenceLedger reference

**Pin:** `d2f67e6`

## One-line

Bounded inference-change assurance: attempt-level evidence, non-compensatory gates, APPROVE / REJECT / ABSTAIN. Completeness is not sufficiency.

## Proof moment

IFStruct JSON-mode change: 100/100 pairs, improved p50 latency, structure unchanged, cost checks without variation → **INCONCLUSIVE**. Inspect without providers:

```bash
make check
python scripts/audit_decision_completeness.py \
  --decision-json benchmarks/reports/ifstruct-json-mode-20260911/decision.json
```

## Public evidence (retained)

| Experiment | Outcome |
|------------|---------|
| IFStruct JSON-mode | INCONCLUSIVE |
| Synthetic JSON-mode | INCONCLUSIVE |
| OpenRouter billing | VALIDATED_MISMATCH |

## External pilot

Not started. Provider/customer calls `0`. Authorized paid spend `$0`.

## Invariant

Every required check must independently support approval. Unknown cost is not zero. History does not rewrite itself.
