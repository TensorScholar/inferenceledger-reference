# Development and verification

Local validation in this snapshot is `make check`. Committed decision artifacts can be inspected
with no provider credentials and no paid calls.

Current canonical engineering and the current distribution CLI live in
[TensorScholar/inferenceledger](https://github.com/TensorScholar/inferenceledger). This file
describes only the historical snapshot in this repository.

## Supported Python

The package declares Python `>=3.11,<3.13`. CI is configured for standard ephemeral GitHub-hosted
Linux runners rather than a public self-hosted machine.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,providers]"
```

Provider extras are needed only for provider-backed execution. The default lint/type/test checks do
not make paid provider calls.

## Full check

```bash
make check
```

Equivalent commands:

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

## Audit a committed decision without provider credentials

The IFStruct decision is intentionally useful as an offline inspection target:

```bash
.venv/bin/python scripts/audit_decision_completeness.py \
  --decision-json benchmarks/reports/ifstruct-json-mode-20260911/decision.json
```

Expected output:

```text
contract=decision-evidence-completeness-v1 satisfied=true present=14 explicitly_unavailable=0 missing=0 missing_checks=-
```

This checks representational completeness only. Read the decision itself afterwards:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("benchmarks/reports/ifstruct-json-mode-20260911/decision.json")
d = json.loads(p.read_text())
print("experiment:", d["experiment_id"])
print("classification:", d["classification"])
print("decision:", d["final_decision"])
print("gate:", d["change_gate_result"]["decision"])
print("unsupported claims:")
for claim in d["unsupported_claims"]:
    print(" -", claim)
PY
```

The expected decision is still `inconclusive`. A complete bundle is not automatically sufficient
evidence for approval.

## Replaying benchmark execution

Scripts under `scripts/` separate the major phases:

- `run_benchmark.py`: reference OpenAI-compatible benchmark path;
- `run_litellm_workload.py`: controlled LiteLLM arm execution;
- `compare_paired.py`: paired run evidence;
- `decide_change.py`: stored baseline/candidate decision bundle;
- `audit_decision_completeness.py`: required evidence-surface audit;
- `audit_evidence_manifest.py`: deterministic pilot-bundle manifest audit;
- `run_openrouter_billing_validation.py`: frozen OpenRouter billing experiment tooling.

This snapshot preserves the historical reference CLI surface: `inferenceledger-smoke` and the
compatibility alias `inference-smoke`. They are reference-executor smoke utilities, not the
decision interface. The current canonical distribution CLI is `inferenceledger`, maintained in
[TensorScholar/inferenceledger](https://github.com/TensorScholar/inferenceledger). Do not treat
this snapshot's entry points as the current install identity. The decision path here remains the
evidence/analysis workflow described above.

Do not execute provider-backed scripts casually. They require an explicit experiment contract,
credentials, and—where applicable—a spend authorization. The repository's default verification path
must stay at `$0` provider spend.

## Historical evidence policy

Committed outcomes are not regenerated merely to make them look cleaner. In particular:

- `INCONCLUSIVE` remains `INCONCLUSIVE`;
- `VALIDATED_MISMATCH` remains `VALIDATED_MISMATCH`;
- thresholds are not changed after observing results;
- provider-reported or gateway-reported values are not promoted to invoice truth;
- unknown cost is not converted to zero;
- later semantic interpretation does not silently rewrite an earlier frozen rule.

Publication-only redactions are separately declared in `benchmarks/PUBLICATION_REDACTIONS.json`.
