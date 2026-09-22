# Documentation

This repository is a dated evidence and mechanism snapshot. Current engineering status, Task-5
acquisition materials, and living strategy documents are maintained in
[TensorScholar/inferenceledger](https://github.com/TensorScholar/inferenceledger). They are
intentionally not duplicated here so this tree is not a second source of truth. Their absence from
this snapshot does not mean those documents do not exist.

Read living docs **in this snapshot** in this order: `architecture.md` → `evidence.md` →
`development.md`.

## Public surface

Figures in [`public-surface/`](public-surface/) explain this snapshot's design and evidence
boundary. They are not a source of truth for the current private engineering state in
TensorScholar/inferenceledger.

| Artifact | Role |
| --- | --- |
| [`architecture.png`](public-surface/architecture.png) | Freeze → capture → ledger → verify → decide. |
| [`decision-flow.png`](public-surface/decision-flow.png) | Evidence → validation → gate → verdict. |
| [`evidence.png`](public-surface/evidence.png) | Retained outcomes from the committed experiments. |
| [`engineering-evidence-brief.pdf`](public-surface/engineering-evidence-brief.pdf) | One-page brief of the same boundary. |

## Living documentation in this snapshot

| Doc | Role |
| --- | --- |
| [`architecture.md`](architecture.md) | Conceptual path and the modules that implement it. |
| [`evidence.md`](evidence.md) | Claim-to-evidence index, decision vocabulary, and claim boundaries. |
| [`development.md`](development.md) | Install, `make check`, offline audit, and historical-evidence policy. |

## Historical / evidence-bound protocol material

These files are frozen protocols tied to committed experiments. They are not living product docs
and must not be read as a current roadmap.

- [`10_PROVIDER_BILLING_VALIDATION_PROTOCOL.md`](10_PROVIDER_BILLING_VALIDATION_PROTOCOL.md)
- [`11_OPENROUTER_GATEWAY_BILLING_VALIDATION.md`](11_OPENROUTER_GATEWAY_BILLING_VALIDATION.md)
- [`12_OPENROUTER_EVIDENCE_SEMANTICS.md`](12_OPENROUTER_EVIDENCE_SEMANTICS.md)

Numbering begins at 10 because earlier strategy, status, and process documents were omitted from
this public reference surface. The filenames are retained for historical provenance. Current
canonical status lives in TensorScholar/inferenceledger, not in reconstructed numbers 00–09.
