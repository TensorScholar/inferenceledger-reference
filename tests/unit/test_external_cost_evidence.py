from __future__ import annotations

from dataclasses import replace

import pytest

from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)


def _reported_attempt() -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=1,
        provider="external-provider",
        model="external-model",
        outcome=AttemptOutcome.SUCCEEDED,
        latency_ms=120,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        reported_cost_usd=0.0012,
        cost_evidence=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        cost_source="external_stack_standard_log",
        cost_source_record_id="log-123",
    )


def test_external_reported_cost_is_known_without_pricing_provenance() -> None:
    attempt = _reported_attempt()

    assert attempt.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
    assert attempt.reported_cost_usd == pytest.approx(0.0012)
    assert attempt.calculated_cost_usd is None
    assert attempt.cost_usd == pytest.approx(0.0012)
    assert attempt.cost_is_known is True
    assert attempt.pricing_record_id is None
    assert attempt.pricing_source_url is None


def test_external_reported_cost_requires_source_provenance() -> None:
    attempt = _reported_attempt()
    with pytest.raises(ValueError, match="requires source and source record id"):
        replace(attempt, cost_source=None)

    with pytest.raises(ValueError, match="requires source and source record id"):
        replace(attempt, cost_source_record_id="")


def test_external_reported_cost_requires_an_amount() -> None:
    with pytest.raises(ValueError, match="requires reported_cost_usd"):
        replace(_reported_attempt(), reported_cost_usd=None)


def test_external_reported_cost_rejects_pricing_provenance() -> None:
    with pytest.raises(ValueError, match="must not carry pricing provenance"):
        replace(_reported_attempt(), pricing_table_version="pricing-v1")


def test_external_reported_cost_rejects_calculated_amount() -> None:
    with pytest.raises(ValueError, match="both calculated and reported cost"):
        replace(_reported_attempt(), calculated_cost_usd=0.0012)


def test_calculated_cost_rejects_external_source_provenance() -> None:
    with pytest.raises(ValueError, match="must not carry external source provenance"):
        ProviderAttempt(
            attempt_index=1,
            provider="openai",
            model="gpt-4o-mini",
            outcome=AttemptOutcome.SUCCEEDED,
            latency_ms=100,
            calculated_cost_usd=0.001,
            cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
            pricing_table_version="test-v1",
            pricing_record_id="openai:gpt-4o-mini:2026-09-03",
            pricing_observed_at="2026-09-03",
            pricing_source_url="https://example.com/pricing",
            cost_source="external-stack",
        )


def test_unknown_cost_rejects_external_amount_and_provenance() -> None:
    with pytest.raises(ValueError, match="must not carry a monetary amount"):
        ProviderAttempt(
            attempt_index=1,
            provider="provider",
            model="model",
            outcome=AttemptOutcome.FAILED,
            latency_ms=20,
            reported_cost_usd=0.001,
        )

    with pytest.raises(ValueError, match="must not carry external source provenance"):
        ProviderAttempt(
            attempt_index=1,
            provider="provider",
            model="model",
            outcome=AttemptOutcome.FAILED,
            latency_ms=20,
            cost_source="external-stack",
        )


def test_negative_external_reported_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="reported_cost_usd must be non-negative"):
        replace(_reported_attempt(), reported_cost_usd=-0.01)
